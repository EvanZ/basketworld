import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure repository root is on sys.path so absolute imports work when launched as a script
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.backend.routes.evaluation_routes import router as evaluation_router
from app.backend.routes.analytics_routes import router as analytics_router
from app.backend.routes.media_routes import router as media_router
from app.backend.routes.admin_routes import router as admin_router
from app.backend.routes.lifecycle_routes import router as lifecycle_router
from app.backend.routes.policy_routes import router as policy_router
from app.backend.routes.playable_routes import router as playable_router
from app.backend.routes.training_routes import router as training_router
from app.backend.routes.skill_ratings_routes import router as skill_ratings_router
from app.backend.playable_analytics import playable_analytics_emitter
from dotenv import load_dotenv


def _is_truthy(raw: str | None) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_cors_origins(raw: str | None) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return ["*"]
    if text == "*":
        return ["*"]
    origins = [part.strip() for part in text.split(",")]
    return [origin for origin in origins if origin]


# Load backend env files while avoiding `.env` virtualenv directories.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_ENV_CANDIDATES = (
    PROJECT_ROOT / "app" / "backend" / ".env.app",
    PROJECT_ROOT / "app" / "backend" / ".env.local",
    PROJECT_ROOT / "app" / "backend" / ".env",
)
for env_path in _BACKEND_ENV_CANDIDATES:
    if env_path.is_file():
        load_dotenv(env_path, override=False)

public_mode = _is_truthy(os.getenv("BW_PUBLIC_MODE"))
cors_origins = _parse_cors_origins(os.getenv("BACKEND_CORS_ALLOW_ORIGINS", "*"))
allow_credentials = "*" not in cors_origins

# --- FastAPI App ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Playable routes are always mounted; public mode can expose only this subset.
app.include_router(playable_router)

if public_mode:
    # Keep only minimal routes needed by public playable clients.
    app.include_router(media_router)
else:
    app.include_router(evaluation_router)
    app.include_router(analytics_router)
    app.include_router(media_router)
    app.include_router(admin_router)
    app.include_router(lifecycle_router)
    app.include_router(policy_router)
    app.include_router(training_router)
    app.include_router(skill_ratings_router)


@app.on_event("shutdown")
def _shutdown_playable_analytics():
    playable_analytics_emitter.shutdown()
