<script setup>
import { computed } from 'vue';

const props = defineProps({
  gameState: {
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

function asNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatPercent(value, digits = 1) {
  const parsed = asNumber(value);
  if (parsed === null) return 'N/A';
  return `${(parsed * 100).toFixed(digits)}%`;
}

function formatValue(value, fallback = 'N/A') {
  if (value === null || value === undefined || value === '') return fallback;
  return value;
}

function normalizeSkillSet(raw) {
  const source = raw && typeof raw === 'object' ? raw : {};
  return {
    layup: Array.isArray(source.layup) ? source.layup.map((v) => asNumber(v)).map((v) => (v === null ? null : v)) : [],
    three_pt: Array.isArray(source.three_pt) ? source.three_pt.map((v) => asNumber(v)).map((v) => (v === null ? null : v)) : [],
    dunk: Array.isArray(source.dunk) ? source.dunk.map((v) => asNumber(v)).map((v) => (v === null ? null : v)) : [],
  };
}

function hasAnySkillValues(skillSet) {
  return ['layup', 'three_pt', 'dunk'].some((key) => Array.isArray(skillSet[key]) && skillSet[key].length > 0);
}

const userIds = computed(() => {
  const ids = props.gameState?.playable_user_ids;
  if (!Array.isArray(ids)) return [];
  return ids.map((id) => asNumber(id)).filter((id) => id !== null);
});

const aiIds = computed(() => {
  const ids = props.gameState?.playable_ai_ids;
  if (!Array.isArray(ids)) return [];
  return ids.map((id) => asNumber(id)).filter((id) => id !== null);
});

const sampledSideSkills = computed(() => {
  const sideSkills = props.gameState?.playable_side_skills;
  const fallbackCurrent = normalizeSkillSet(
    props.gameState?.offense_shooting_pct_sampled || props.gameState?.offense_shooting_pct_by_player,
  );
  const currentOffenseIsUser = Boolean(props.gameState?.playable_user_on_offense);

  const rawUser = normalizeSkillSet(sideSkills?.user);
  const rawAi = normalizeSkillSet(sideSkills?.ai);

  const userSkills = hasAnySkillValues(rawUser)
    ? rawUser
    : (currentOffenseIsUser ? fallbackCurrent : normalizeSkillSet(null));
  const aiSkills = hasAnySkillValues(rawAi)
    ? rawAi
    : (!currentOffenseIsUser ? fallbackCurrent : normalizeSkillSet(null));

  return { user: userSkills, ai: aiSkills };
});

function buildSkillRows(ids, skillSet) {
  if (!Array.isArray(ids) || ids.length === 0) return [];
  return ids.map((playerId, idx) => ({
    playerId,
    layup: skillSet.layup[idx] ?? null,
    threePt: skillSet.three_pt[idx] ?? null,
    dunk: skillSet.dunk[idx] ?? null,
  }));
}

function getPlayerSurname(playerId) {
  const id = asNumber(playerId);
  if (id === null) return '';
  const map = props.playerDisplayNames && typeof props.playerDisplayNames === 'object'
    ? props.playerDisplayNames
    : {};
  const raw = map[id] ?? map[String(id)];
  return typeof raw === 'string' ? raw.trim() : '';
}

function getPlayerJerseyNumber(playerId) {
  const id = asNumber(playerId);
  if (id === null) return '';
  const map = props.playerJerseyNumbers && typeof props.playerJerseyNumbers === 'object'
    ? props.playerJerseyNumbers
    : {};
  const raw = map[id] ?? map[String(id)];
  const jersey = typeof raw === 'string' ? raw.trim() : String(raw ?? '').trim();
  return jersey;
}

function formatPlayerLabel(playerId) {
  const id = asNumber(playerId);
  if (id === null) return 'Unknown';
  const surname = getPlayerSurname(id);
  const jersey = getPlayerJerseyNumber(id);
  if (surname && jersey) return `${surname} #${jersey}`;
  if (surname) return `${surname} #${id}`;
  if (jersey) return `Player #${jersey}`;
  return `Player ${id}`;
}

const userSkillRows = computed(() => buildSkillRows(userIds.value, sampledSideSkills.value.user));
const aiSkillRows = computed(() => buildSkillRows(aiIds.value, sampledSideSkills.value.ai));
</script>

<template>
  <section class="environment-tab">
    <div class="env-header">
      <h3>Environment</h3>
      <span>Read-only game parameters</span>
    </div>

    <div v-if="!gameState" class="env-empty">No environment data available.</div>

    <div v-else class="env-grid">
      <article class="env-card env-card-wide">
        <h4>Sampled Shooting Skills (Game Start)</h4>
        <div class="skills-grid">
          <div class="skill-team">
            <h5>You (Blue)</h5>
            <div class="skill-table">
              <div class="skill-row header">
                <span>Player</span>
                <span>Layup</span>
                <span>3PT</span>
                <span>Dunk</span>
              </div>
              <div
                v-for="row in userSkillRows"
                :key="`user-skill-${row.playerId}`"
                class="skill-row"
              >
                <span>{{ formatPlayerLabel(row.playerId) }}</span>
                <span>{{ formatPercent(row.layup) }}</span>
                <span>{{ formatPercent(row.threePt) }}</span>
                <span>{{ formatPercent(row.dunk) }}</span>
              </div>
            </div>
          </div>

          <div class="skill-team">
            <h5>AI (Red)</h5>
            <div class="skill-table">
              <div class="skill-row header">
                <span>Player</span>
                <span>Layup</span>
                <span>3PT</span>
                <span>Dunk</span>
              </div>
              <div
                v-for="row in aiSkillRows"
                :key="`ai-skill-${row.playerId}`"
                class="skill-row"
              >
                <span>{{ formatPlayerLabel(row.playerId) }}</span>
                <span>{{ formatPercent(row.layup) }}</span>
                <span>{{ formatPercent(row.threePt) }}</span>
                <span>{{ formatPercent(row.dunk) }}</span>
              </div>
            </div>
          </div>
        </div>
      </article>

      <article class="env-card">
        <h4>Defender Turnover Pressure</h4>
        <div class="param-row"><span>Pressure distance</span><strong>{{ formatValue(gameState.defender_pressure_distance) }}</strong></div>
        <div class="param-row"><span>Turnover chance</span><strong>{{ formatPercent(gameState.defender_pressure_turnover_chance, 2) }}</strong></div>
        <div class="param-row"><span>Decay lambda</span><strong>{{ formatValue(gameState.defender_pressure_decay_lambda) }}</strong></div>
      </article>

      <article class="env-card">
        <h4>Pass Interception</h4>
        <div class="param-row"><span>Base steal rate</span><strong>{{ formatPercent(gameState.base_steal_rate, 2) }}</strong></div>
        <div class="param-row"><span>Perpendicular decay</span><strong>{{ formatValue(gameState.steal_perp_decay) }}</strong></div>
        <div class="param-row"><span>Distance factor</span><strong>{{ formatValue(gameState.steal_distance_factor) }}</strong></div>
        <div class="param-row"><span>Position weight min</span><strong>{{ formatValue(gameState.steal_position_weight_min) }}</strong></div>
      </article>

      <article class="env-card">
        <h4>Spawn Distance</h4>
        <div class="param-row"><span>Min spawn distance</span><strong>{{ formatValue(gameState.spawn_distance) }}</strong></div>
        <div class="param-row"><span>Max spawn distance</span><strong>{{ gameState.max_spawn_distance ?? 'Unlimited' }}</strong></div>
        <div class="param-row"><span>Defender spawn distance</span><strong>{{ formatValue(gameState.defender_spawn_distance) }}</strong></div>
        <div class="param-row"><span>Offense boundary margin</span><strong>{{ formatValue(gameState.offense_spawn_boundary_margin) }}</strong></div>
        <div class="param-row"><span>Defender guard distance</span><strong>{{ formatValue(gameState.defender_guard_distance) }}</strong></div>
      </article>

      <article class="env-card">
        <h4>Shot Pressure</h4>
        <div class="param-row"><span>3PT extra decay / hex</span><strong>{{ formatPercent(gameState.three_pt_extra_hex_decay, 2) }}</strong></div>
        <div class="param-row"><span>Pressure enabled</span><strong>{{ gameState.shot_pressure_enabled ? 'Yes' : 'No' }}</strong></div>
        <div class="param-row"><span>Max pressure</span><strong>{{ formatPercent(gameState.shot_pressure_max, 2) }}</strong></div>
        <div class="param-row"><span>Pressure lambda</span><strong>{{ formatValue(gameState.shot_pressure_lambda) }}</strong></div>
        <div class="param-row"><span>Pressure arc degrees</span><strong>{{ formatValue(gameState.shot_pressure_arc_degrees) }}°</strong></div>
      </article>

      <article class="env-card">
        <h4>3-Second Violation Rules</h4>
        <div class="param-row"><span>Lane width</span><strong>{{ formatValue(gameState.three_second_lane_width) }}</strong></div>
        <div class="param-row"><span>Lane height</span><strong>{{ formatValue(gameState.three_second_lane_height) }}</strong></div>
        <div class="param-row"><span>Max steps in lane</span><strong>{{ formatValue(gameState.three_second_max_steps) }}</strong></div>
        <div class="param-row"><span>Offensive 3-sec enabled</span><strong>{{ gameState.offensive_three_seconds_enabled ? 'Yes' : 'No' }}</strong></div>
        <div class="param-row"><span>Illegal defense enabled</span><strong>{{ gameState.illegal_defense_enabled ? 'Yes' : 'No' }}</strong></div>
        <div class="param-row"><span>Lane hexes count</span><strong>{{ Array.isArray(gameState.offensive_lane_hexes) ? gameState.offensive_lane_hexes.length : 0 }}</strong></div>
      </article>

      <article class="env-card">
        <h4>Environment Settings</h4>
        <div class="param-row"><span>Players per side</span><strong>{{ gameState.players_per_side ?? userIds.length }}</strong></div>
        <div class="param-row"><span>Court dimensions</span><strong>{{ gameState.court_width }}×{{ gameState.court_height }}</strong></div>
        <div class="param-row"><span>Ball holder</span><strong>Player {{ formatValue(gameState.ball_holder) }}</strong></div>
        <div class="param-row"><span>Shot clock</span><strong>{{ gameState.shot_clock }}</strong></div>
        <div class="param-row"><span>Min shot clock at reset</span><strong>{{ formatValue(gameState.min_shot_clock) }}</strong></div>
        <div class="param-row"><span>Three-point distance</span><strong>{{ formatValue(gameState.three_point_distance) }}</strong></div>
        <div class="param-row"><span>Three-point short distance</span><strong>{{ formatValue(gameState.three_point_short_distance) }}</strong></div>
      </article>

      <article class="env-card">
        <h4>Policies</h4>
        <div class="param-row"><span>Run ID</span><strong class="mono">{{ formatValue(gameState.run_id) }}</strong></div>
        <div class="param-row"><span>Unified policy</span><strong class="mono">{{ formatValue(gameState.unified_policy_name) }}</strong></div>
        <div class="param-row"><span>Pass mode</span><strong>{{ formatValue(gameState.pass_mode) }}</strong></div>
        <div class="param-row"><span>Pass target strategy</span><strong>{{ formatValue(gameState.pass_target_strategy) }}</strong></div>
        <div class="param-row"><span>Illegal action policy</span><strong>{{ formatValue(gameState.illegal_action_policy) }}</strong></div>
      </article>

      <article class="env-card">
        <h4>Shot Parameters</h4>
        <div class="param-row"><span>Layup μ</span><strong>{{ formatPercent(gameState.shot_params?.layup_pct) }}</strong></div>
        <div class="param-row"><span>Layup σ</span><strong>{{ formatPercent(gameState.shot_params?.layup_std) }}</strong></div>
        <div class="param-row"><span>Three-point μ</span><strong>{{ formatPercent(gameState.shot_params?.three_pt_pct) }}</strong></div>
        <div class="param-row"><span>Three-point σ</span><strong>{{ formatPercent(gameState.shot_params?.three_pt_std) }}</strong></div>
        <div class="param-row"><span>Dunk μ</span><strong>{{ formatPercent(gameState.shot_params?.dunk_pct) }}</strong></div>
        <div class="param-row"><span>Dunk σ</span><strong>{{ formatPercent(gameState.shot_params?.dunk_std) }}</strong></div>
        <div class="param-row"><span>Dunks allowed</span><strong>{{ gameState.shot_params?.allow_dunks ? 'Yes' : 'No' }}</strong></div>
      </article>
    </div>
  </section>
</template>

<style scoped>
.environment-tab {
  margin-top: 0.9rem;
  border-top: 1px solid rgba(148, 163, 184, 0.25);
  padding-top: 0.9rem;
}

.env-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.6rem;
  margin-bottom: 0.75rem;
}

.env-header h3 {
  margin: 0;
  font-size: 0.92rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--app-accent);
}

.env-header span {
  font-size: 0.75rem;
  color: var(--app-text-muted);
}

.env-empty {
  color: var(--app-text-muted);
  font-size: 0.84rem;
}

.env-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 0.7rem;
}

.env-card {
  border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 12px;
  background: rgba(15, 23, 42, 0.35);
  padding: 0.65rem 0.75rem;
}

.env-card-wide {
  grid-column: 1 / -1;
}

.env-card h4 {
  margin: 0 0 0.5rem;
  font-size: 0.8rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--app-text-muted);
}

.param-row {
  display: flex;
  justify-content: space-between;
  gap: 0.6rem;
  font-size: 0.8rem;
  padding: 0.16rem 0;
}

.param-row span {
  color: var(--app-text-muted);
}

.param-row strong {
  color: var(--app-text);
  font-weight: 600;
  text-align: right;
}

.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 0.75rem;
}

.skills-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 0.65rem;
}

.skill-team h5 {
  margin: 0 0 0.4rem;
  font-size: 0.78rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.skill-table {
  border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 10px;
  overflow: hidden;
}

.skill-row {
  display: grid;
  grid-template-columns: 72px repeat(3, 1fr);
  gap: 0.35rem;
  align-items: center;
  padding: 0.33rem 0.45rem;
  font-size: 0.78rem;
}

.skill-row.header {
  background: rgba(148, 163, 184, 0.12);
  font-size: 0.72rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--app-text-muted);
}

.skill-row:not(.header):nth-child(even) {
  background: rgba(15, 23, 42, 0.25);
}

@media (max-width: 900px) {
  .env-grid {
    grid-template-columns: 1fr;
  }
}
</style>
