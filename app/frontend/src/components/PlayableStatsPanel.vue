<script setup>
import { computed } from 'vue';

const props = defineProps({
  gameState: {
    type: Object,
    default: null,
  },
  userPlayerIds: {
    type: Array,
    default: () => [],
  },
  score: {
    type: Object,
    default: () => ({ user: 0, ai: 0 }),
  },
  stats: {
    type: Object,
    default: null,
  },
  playByPlay: {
    type: Array,
    default: () => [],
  },
  playerDisplayNames: {
    type: Object,
    default: () => ({}),
  },
  playerJerseyNumbers: {
    type: Object,
    default: () => ({}),
  },
});

function makeEmptyShotLine() {
  return { attempts: 0, made: 0 };
}

function makeEmptyTeamStats() {
  return {
    shots: {
      total: makeEmptyShotLine(),
      twoPt: makeEmptyShotLine(),
      threePt: makeEmptyShotLine(),
      dunk: makeEmptyShotLine(),
    },
    assists: 0,
    turnovers: 0,
    defensiveViolations: 0,
    offensiveViolations: 0,
    actions: {
      noop: 0,
      move: 0,
      shoot: 0,
      pass: 0,
      other: 0,
      total: 0,
    },
    perPlayer: {},
  };
}

function makeEmptyPlayerStats() {
  return {
    shots: {
      total: makeEmptyShotLine(),
      twoPt: makeEmptyShotLine(),
      threePt: makeEmptyShotLine(),
      dunk: makeEmptyShotLine(),
    },
    assists: 0,
    turnovers: 0,
    points: 0,
    actions: {
      noop: 0,
      move: 0,
      shoot: 0,
      pass: 0,
      other: 0,
      total: 0,
    },
  };
}

function makeEmptyStats() {
  return {
    turns: 0,
    possessions: 0,
    userOffensiveTurns: 0,
    userDefensiveTurns: 0,
    byTeam: {
      user: makeEmptyTeamStats(),
      ai: makeEmptyTeamStats(),
    },
    turnoverReasons: {},
  };
}

function asCount(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.max(0, parsed) : 0;
}

function formatShotCell(line) {
  const made = asCount(line?.made);
  const attempts = asCount(line?.attempts);
  return `${made}-${attempts}`;
}

function getPlayerSurname(playerId) {
  const id = Number(playerId);
  if (!Number.isFinite(id)) return '';
  const map = props.playerDisplayNames && typeof props.playerDisplayNames === 'object'
    ? props.playerDisplayNames
    : {};
  const raw = map[id] ?? map[String(id)];
  return typeof raw === 'string' ? raw.trim() : '';
}

function getPlayerJerseyNumber(playerId) {
  const id = Number(playerId);
  if (!Number.isFinite(id)) return '';
  const map = props.playerJerseyNumbers && typeof props.playerJerseyNumbers === 'object'
    ? props.playerJerseyNumbers
    : {};
  const raw = map[id] ?? map[String(id)];
  return typeof raw === 'string' ? raw.trim() : String(raw ?? '').trim();
}

function formatPlayerNameWithJersey(playerId) {
  const id = Number(playerId);
  if (!Number.isFinite(id)) return 'Unknown';
  const surname = getPlayerSurname(id);
  const jersey = getPlayerJerseyNumber(id);
  if (surname && jersey) return `${surname} #${jersey}`;
  if (surname) return `${surname} #${id}`;
  if (jersey) return `Player #${jersey}`;
  return `Player ${id}`;
}

const statsSafe = computed(() => {
  if (props.stats && typeof props.stats === 'object') return props.stats;
  return makeEmptyStats();
});

const userPlayerIdsForStats = computed(() => {
  const ids = Array.isArray(props.userPlayerIds) ? props.userPlayerIds : [];
  return ids.map((id) => Number(id)).filter((id) => Number.isFinite(id)).sort((a, b) => a - b);
});

const aiPlayerIdsForStats = computed(() => {
  const ids = props.gameState?.playable_ai_ids;
  if (!Array.isArray(ids)) return [];
  return ids.map((id) => Number(id)).filter((id) => Number.isFinite(id)).sort((a, b) => a - b);
});

const boxScoreRows = computed(() => {
  const byTeam = statsSafe.value?.byTeam || {};
  const getPlayerLine = (teamKey, playerId) => {
    const team = byTeam[teamKey] || {};
    const map = team.perPlayer && typeof team.perPlayer === 'object' ? team.perPlayer : {};
    return map[String(playerId)] || map[playerId] || makeEmptyPlayerStats();
  };

  const buildRows = (teamKey, playerIds) => {
    const line = byTeam[teamKey] || makeEmptyTeamStats();
    const shots = line.shots || {};
    const rows = playerIds.map((pid) => {
      const playerLine = getPlayerLine(teamKey, pid);
      const playerShots = playerLine.shots || {};
      return {
        key: `${teamKey}-player-${pid}`,
        player: formatPlayerNameWithJersey(pid),
        isTotal: false,
        points: asCount(playerLine.points),
        fg: formatShotCell(playerShots.total),
        twoPt: formatShotCell(playerShots.twoPt),
        threePt: formatShotCell(playerShots.threePt),
        dunk: formatShotCell(playerShots.dunk),
        assists: asCount(playerLine.assists),
        turnovers: asCount(playerLine.turnovers),
      };
    });

    rows.push({
      key: `${teamKey}-total`,
      player: teamKey === 'user' ? 'You' : 'AI',
      isTotal: true,
      points: asCount(props.score?.[teamKey]),
      fg: formatShotCell(shots.total),
      twoPt: formatShotCell(shots.twoPt),
      threePt: formatShotCell(shots.threePt),
      dunk: formatShotCell(shots.dunk),
      assists: asCount(line.assists),
      turnovers: asCount(line.turnovers),
    });

    return rows;
  };

  return [
    ...buildRows('user', userPlayerIdsForStats.value),
    ...buildRows('ai', aiPlayerIdsForStats.value),
  ];
});

const playByPlayRows = computed(() => {
  const rows = Array.isArray(props.playByPlay) ? props.playByPlay : [];
  return rows
    .map((row, idx) => ({
      key: row?.id ?? `pbp-${idx}`,
      period: String(row?.period || '-'),
      clock: String(row?.clock || '--:--'),
      team: String(row?.team || '-'),
      score: String(row?.score || '0-0'),
      event: String(row?.event || ''),
    }))
    .filter((row) => row.event.length > 0);
});
</script>

<template>
  <div class="playable-stats-panel">
    <section class="stats-card box-score-card">
      <h3>Box Score</h3>
      <div class="table-scroll">
        <table class="box-score-table">
          <thead>
            <tr>
              <th>Player</th>
              <th>PTS</th>
              <th>FG</th>
              <th>2PT</th>
              <th>3PT</th>
              <th>Dunk</th>
              <th>AST</th>
              <th>TOV</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="row in boxScoreRows"
              :key="row.key"
              :class="{ 'total-row': row.isTotal }"
            >
              <td class="player-cell">{{ row.player }}</td>
              <td>{{ row.points }}</td>
              <td>{{ row.fg }}</td>
              <td>{{ row.twoPt }}</td>
              <td>{{ row.threePt }}</td>
              <td>{{ row.dunk }}</td>
              <td>{{ row.assists }}</td>
              <td>{{ row.turnovers }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="stats-card play-by-play-card">
      <h3>Play-By-Play</h3>
      <div class="table-scroll play-by-play-scroll">
        <table class="play-by-play-table">
          <thead>
            <tr>
              <th>Period</th>
              <th>Clock</th>
              <th>Score</th>
              <th>Team</th>
              <th>Event</th>
            </tr>
          </thead>
          <tbody>
            <template v-if="playByPlayRows.length === 0">
              <tr>
                <td colspan="5" class="play-by-play-empty">No events yet.</td>
              </tr>
            </template>
            <template v-else>
              <tr
                v-for="row in playByPlayRows"
                :key="row.key"
              >
                <td>{{ row.period }}</td>
                <td>{{ row.clock }}</td>
                <td class="play-by-play-score">{{ row.score }}</td>
                <td class="play-by-play-team">{{ row.team }}</td>
                <td class="play-by-play-event">{{ row.event }}</td>
              </tr>
            </template>
          </tbody>
        </table>
      </div>
    </section>
  </div>
</template>

<style scoped>
.playable-stats-panel {
  padding: 0.2rem 0.4rem;
  display: flex;
  flex-direction: column;
  gap: 0.9rem;
}

.stats-card {
  border: 1px solid rgba(148, 163, 184, 0.28);
  border-radius: 12px;
  background: rgba(15, 23, 42, 0.2);
  padding: 0.55rem;
  display: flex;
  flex-direction: column;
  gap: 0.55rem;
}

h3 {
  margin: 0;
  color: var(--app-accent);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 0.95rem;
}

.table-scroll {
  width: 100%;
  overflow-x: auto;
}

.play-by-play-scroll {
  max-height: 270px;
  overflow-y: auto;
}

.box-score-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.74rem;
}

.play-by-play-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.72rem;
}

.box-score-table th,
.box-score-table td,
.play-by-play-table th,
.play-by-play-table td {
  border: 1px solid rgba(148, 163, 184, 0.22);
  padding: 0.34rem 0.4rem;
  text-align: right;
}

.box-score-table th:first-child,
.box-score-table td:first-child,
.play-by-play-table th:first-child,
.play-by-play-table td:first-child {
  text-align: left;
}

.box-score-table th {
  color: var(--app-text-muted);
  background: rgba(15, 23, 42, 0.45);
  font-weight: 600;
}

.play-by-play-table th {
  color: var(--app-text-muted);
  background: rgba(15, 23, 42, 0.45);
  font-weight: 600;
}

.box-score-table tbody tr:nth-child(even) {
  background: rgba(15, 23, 42, 0.25);
}

.play-by-play-table tbody tr:nth-child(even) {
  background: rgba(15, 23, 42, 0.25);
}

.player-cell {
  letter-spacing: 0.06em;
  font-weight: 500;
}

.box-score-table .total-row td {
  font-weight: 700;
  background: rgba(59, 130, 246, 0.1);
}

.play-by-play-table th:nth-child(1),
.play-by-play-table td:nth-child(1) {
  min-width: 86px;
}

.play-by-play-table th:nth-child(2),
.play-by-play-table td:nth-child(2) {
  min-width: 62px;
}

.play-by-play-table th:nth-child(3),
.play-by-play-table td:nth-child(3) {
  min-width: 90px;
  text-align: center;
}

.play-by-play-table th:nth-child(4),
.play-by-play-table td:nth-child(4) {
  min-width: 54px;
  text-align: center;
}

.play-by-play-table th:nth-child(5),
.play-by-play-table td:nth-child(5) {
  text-align: left;
}

.play-by-play-team {
  letter-spacing: 0.06em;
  font-weight: 600;
}

.play-by-play-event {
  white-space: normal;
  line-height: 1.3;
}

.play-by-play-score {
  font-variant-numeric: tabular-nums;
  letter-spacing: 0.04em;
}

.play-by-play-empty {
  text-align: center !important;
  color: var(--app-text-muted);
}
</style>
