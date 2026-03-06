from __future__ import annotations

import os
import re
import secrets
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from app.backend.state import GameState


PLAYABLE_SESSION_HEADER = "X-Playable-Session-Id"

_PLAYABLE_MAX_ACTIVE_ENV = "BW_PLAYABLE_MAX_ACTIVE_SESSIONS"
_PLAYABLE_MAX_ACTIVE_DEFAULT = 8
_PLAYABLE_MAX_ACTIVE_MIN = 1
_PLAYABLE_MAX_ACTIVE_MAX = 512

_PLAYABLE_TTL_MINUTES_ENV = "BW_PLAYABLE_SESSION_TTL_MINUTES"
_PLAYABLE_TTL_MINUTES_DEFAULT = 120
_PLAYABLE_TTL_MINUTES_MIN = 5
_PLAYABLE_TTL_MINUTES_MAX = 24 * 60

_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")

_BOUND_GAME_STATE_MODULES = (
    "app.backend.state",
    "app.backend.routes.playable_routes",
    "app.backend.routes.lifecycle_routes",
    "app.backend.observations",
    "app.backend.mcts",
    "app.backend.routes.media_routes",
)
_BOUND_GAME_STATE_LOCK = threading.RLock()


def _parse_env_int(name: str, default: int, *, min_value: int, max_value: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(str(raw).strip()) if raw is not None else int(default)
    except Exception:
        value = int(default)
    if value < min_value:
        return int(min_value)
    if value > max_value:
        return int(max_value)
    return int(value)


@dataclass
class _PlayableStateEntry:
    game_state: GameState
    created_at: float
    last_accessed_at: float


class PlayableCapacityError(RuntimeError):
    def __init__(self, max_active: int, active_now: int):
        super().__init__(f"Playable session capacity reached ({active_now}/{max_active}).")
        self.max_active = int(max_active)
        self.active_now = int(active_now)


class PlayableSessionStore:
    def __init__(self):
        self._sessions: dict[str, _PlayableStateEntry] = {}
        self._lock = threading.RLock()

    @staticmethod
    def parse_session_id(raw: str | None) -> str | None:
        token = str(raw or "").strip()
        if not token:
            return None
        if not _SESSION_ID_PATTERN.fullmatch(token):
            return None
        return token

    @staticmethod
    def max_active_sessions() -> int:
        return _parse_env_int(
            _PLAYABLE_MAX_ACTIVE_ENV,
            _PLAYABLE_MAX_ACTIVE_DEFAULT,
            min_value=_PLAYABLE_MAX_ACTIVE_MIN,
            max_value=_PLAYABLE_MAX_ACTIVE_MAX,
        )

    @staticmethod
    def session_ttl_minutes() -> int:
        return _parse_env_int(
            _PLAYABLE_TTL_MINUTES_ENV,
            _PLAYABLE_TTL_MINUTES_DEFAULT,
            min_value=_PLAYABLE_TTL_MINUTES_MIN,
            max_value=_PLAYABLE_TTL_MINUTES_MAX,
        )

    @staticmethod
    def _is_state_active(state: GameState) -> bool:
        session = getattr(state, "playable_session", None)
        return bool(isinstance(session, dict) and session.get("active"))

    @staticmethod
    def _new_session_id() -> str:
        # urlsafe output already follows [_A-Za-z0-9-] and is header-safe.
        return secrets.token_urlsafe(24)

    def _cleanup_expired_locked(self, now: float) -> None:
        ttl_seconds = float(self.session_ttl_minutes()) * 60.0
        stale_ids = [
            session_id
            for session_id, entry in self._sessions.items()
            if (now - float(entry.last_accessed_at)) >= ttl_seconds
        ]
        for session_id in stale_ids:
            self._sessions.pop(session_id, None)

    def _active_count_locked(self) -> int:
        return sum(1 for entry in self._sessions.values() if self._is_state_active(entry.game_state))

    def get(self, raw_session_id: str | None) -> tuple[str | None, GameState | None]:
        session_id = self.parse_session_id(raw_session_id)
        if not session_id:
            return None, None
        with self._lock:
            now = time.time()
            self._cleanup_expired_locked(now)
            entry = self._sessions.get(session_id)
            if entry is None:
                return session_id, None
            entry.last_accessed_at = now
            return session_id, entry.game_state

    def get_or_create_for_start(
        self,
        raw_session_id: str | None,
    ) -> tuple[str, GameState, bool]:
        with self._lock:
            now = time.time()
            self._cleanup_expired_locked(now)

            parsed = self.parse_session_id(raw_session_id)
            if parsed:
                existing = self._sessions.get(parsed)
                if existing is not None:
                    existing.last_accessed_at = now
                    return parsed, existing.game_state, False

            max_active = self.max_active_sessions()
            active_now = self._active_count_locked()
            if active_now >= max_active:
                raise PlayableCapacityError(max_active=max_active, active_now=active_now)

            while True:
                session_id = self._new_session_id()
                if session_id not in self._sessions:
                    break

            state = GameState()
            self._sessions[session_id] = _PlayableStateEntry(
                game_state=state,
                created_at=now,
                last_accessed_at=now,
            )
            return session_id, state, True

    def metrics(self) -> dict[str, int]:
        with self._lock:
            now = time.time()
            self._cleanup_expired_locked(now)
            return {
                "total_sessions": int(len(self._sessions)),
                "active_sessions": int(self._active_count_locked()),
                "max_active_sessions": int(self.max_active_sessions()),
                "session_ttl_minutes": int(self.session_ttl_minutes()),
            }


@contextmanager
def bind_game_state(target_state: GameState) -> Iterator[None]:
    previous_values: dict[str, object] = {}
    _BOUND_GAME_STATE_LOCK.acquire()
    try:
        for module_name in _BOUND_GAME_STATE_MODULES:
            module = sys.modules.get(module_name)
            if module is None or not hasattr(module, "game_state"):
                continue
            previous_values[module_name] = getattr(module, "game_state")
            setattr(module, "game_state", target_state)
        yield
    finally:
        for module_name, previous_value in previous_values.items():
            module = sys.modules.get(module_name)
            if module is not None:
                setattr(module, "game_state", previous_value)
        _BOUND_GAME_STATE_LOCK.release()


playable_session_store = PlayableSessionStore()

