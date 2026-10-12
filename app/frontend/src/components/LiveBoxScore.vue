<script setup>
import { computed } from 'vue';
import { playerSurnameForId } from '../utils/playerIdentity';

const props = defineProps({
  boxScore: {
    type: Object,
    default: null,
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

const STAT_COLUMNS = [
  ['pts', 'PTS'],
  ['fg', 'FG'],
  ['two', '2PT'],
  ['three', '3PT'],
  ['dunk', 'DUNK'],
  ['orb', 'ORB'],
  ['drb', 'DRB'],
  ['reb', 'REB'],
  ['ast', 'AST'],
  ['stl', 'STL'],
  ['tov', 'TOV'],
];

function count(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.max(0, Math.round(parsed)) : 0;
}

function formatStat(row, key) {
  if (key === 'fg') return `${count(row?.fgm)}-${count(row?.fga)}`;
  if (key === 'two') return `${count(row?.two_pm)}-${count(row?.two_pa)}`;
  if (key === 'three') return `${count(row?.three_pm)}-${count(row?.three_pa)}`;
  if (key === 'dunk') return `${count(row?.dunk_m)}-${count(row?.dunk_a)}`;
  return count(row?.[key]);
}

function formatPlayerName(playerId) {
  const names = props.playerDisplayNames || {};
  const numbers = props.playerJerseyNumbers || {};
  const rawName = names[playerId] ?? names[String(playerId)] ?? playerSurnameForId(playerId);
  const rawNumber = numbers[playerId] ?? numbers[String(playerId)] ?? playerId;
  const surname = String(rawName || '').trim().toUpperCase();
  const jersey = String(rawNumber ?? '').trim();
  if (surname && jersey) return `${surname} #${jersey}`;
  if (surname) return surname;
  return `Player #${jersey || playerId}`;
}

function normalizeTeam(teamKey, label) {
  const team = props.boxScore?.[teamKey];
  const players = Array.isArray(team?.players)
    ? team.players
      .map((row) => ({ ...row, player_id: Number(row?.player_id) }))
      .filter((row) => Number.isFinite(row.player_id))
      .sort((a, b) => a.player_id - b.player_id)
    : [];
  return {
    key: teamKey,
    label,
    players,
    totals: team?.totals || {},
    score: count(team?.score),
  };
}

const teams = computed(() => [
  normalizeTeam('player', 'Player'),
  normalizeTeam('ai', 'AI'),
]);
</script>

<template>
  <details class="live-box-score" open>
    <summary>
      <span>Box Score</span>
      <span class="box-score-summary-score">
        {{ teams[0].score }}–{{ teams[1].score }}
      </span>
    </summary>

    <div class="box-score-table-scroll">
      <table class="box-score-table">
        <thead>
          <tr>
            <th>Player</th>
            <th v-for="[, label] in STAT_COLUMNS" :key="label">{{ label }}</th>
          </tr>
        </thead>
        <tbody v-for="team in teams" :key="team.key" :class="`team-${team.key}`">
          <tr class="team-heading-row">
            <th :colspan="STAT_COLUMNS.length + 1">{{ team.label }}</th>
          </tr>
          <tr v-for="row in team.players" :key="`${team.key}-${row.player_id}`">
            <td class="player-cell">{{ formatPlayerName(row.player_id) }}</td>
            <td v-for="[key] in STAT_COLUMNS" :key="key">{{ formatStat(row, key) }}</td>
          </tr>
          <tr class="team-total-row">
            <td class="player-cell">TOTAL</td>
            <td v-for="[key] in STAT_COLUMNS" :key="key">{{ formatStat(team.totals, key) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </details>
</template>

<style scoped>
.live-box-score {
  border: 1px solid rgba(148, 163, 184, 0.32);
  border-radius: 12px;
  background: rgba(15, 23, 42, 0.35);
  overflow: hidden;
}

.live-box-score summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.7rem;
  padding: 0.62rem 0.78rem;
  color: var(--app-accent);
  cursor: pointer;
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.box-score-summary-score {
  color: var(--app-text);
  font-variant-numeric: tabular-nums;
}

.box-score-table-scroll {
  width: 100%;
  overflow-x: auto;
  border-top: 1px solid rgba(148, 163, 184, 0.22);
}

.box-score-table {
  width: 100%;
  min-width: 650px;
  border-collapse: collapse;
  font-size: 0.68rem;
  font-variant-numeric: tabular-nums;
}

.box-score-table th,
.box-score-table td {
  border-right: 1px solid rgba(148, 163, 184, 0.17);
  border-bottom: 1px solid rgba(148, 163, 184, 0.17);
  padding: 0.3rem 0.34rem;
  text-align: right;
  white-space: nowrap;
}

.box-score-table thead th {
  color: var(--app-text-muted);
  background: rgba(2, 6, 23, 0.5);
  font-weight: 600;
}

.box-score-table th:first-child,
.box-score-table td:first-child {
  position: sticky;
  left: 0;
  z-index: 1;
  text-align: left;
  background: rgb(19, 28, 48);
}

.team-heading-row th {
  position: static !important;
  text-align: left !important;
  padding: 0.34rem 0.48rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.team-player .team-heading-row th {
  color: #7dd3fc;
  background: rgba(14, 116, 204, 0.24);
}

.team-ai .team-heading-row th {
  color: #fda4af;
  background: rgba(225, 29, 72, 0.2);
}

.team-total-row td {
  font-weight: 700;
  background: rgba(148, 163, 184, 0.1);
}

.player-cell {
  font-weight: 600;
}
</style>
