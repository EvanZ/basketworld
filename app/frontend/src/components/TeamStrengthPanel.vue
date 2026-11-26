<script setup>
import { computed } from 'vue';
import { calculateTeamStrength } from '../utils/teamStrength';
import { playerSurnameForId } from '../utils/playerIdentity';

const props = defineProps({
  report: { type: Object, default: null },
  gameState: { type: Object, default: null },
  playerDisplayNames: { type: Object, default: () => ({}) },
  playerDeterministic: { type: Boolean, default: true },
  opponentDeterministic: { type: Boolean, default: true },
});
const strength = computed(() => calculateTeamStrength(props.report, props.gameState, props));
const signed = (value) => `${value > 0 ? '+' : ''}${value.toFixed(1)}`;
const advantage = computed(() => {
  const value = strength.value.advantage;
  if (Math.abs(value) < 0.05) return 'Even skill matchup';
  return `${value > 0 ? 'Blue' : 'Red'} skill advantage: +${Math.abs(value).toFixed(1)}`;
});
</script>

<template>
  <section class="team-strength" aria-labelledby="team-strength-title">
    <h3 id="team-strength-title">Current team strength</h3>
    <p v-if="!strength.available" class="notice" role="status">{{ strength.reason }}</p>
    <template v-else>
      <div class="totals">
        <div class="blue"><span>Blue</span><strong>{{ signed(strength.blue) }}</strong></div>
        <div class="red"><span>Red</span><strong>{{ signed(strength.red) }}</strong></div>
        <div class="advantage"><strong>{{ advantage }}</strong><span>points per 100 possessions</span></div>
      </div>
      <p class="note">Sum of current players’ skill ratings, relative to the study’s average player profile. These are skill estimates, not a predicted score or win probability. Updates with the current roster.</p>
      <p class="note">Using the selected study below · checkpoint {{ report.summary.checkpoint_update }} · {{ report.reproducibility.team_size }}-on-{{ report.reproducibility.team_size }} · {{ report.summary.action_mode }} actions.</p>
      <p class="note">Checkpoint path and update match. This report cannot verify changed game rules or replaced checkpoint files; use it with the same model and rules as the study.</p>
      <p v-for="warning in strength.warnings" :key="warning" class="notice">{{ warning }}</p>
      <details>
        <summary>Player contributions</summary>
        <div class="table-scroll">
          <table>
            <thead><tr><th>Team</th><th>Player</th><th>Skill rating</th><th>Skill contributions</th></tr></thead>
            <tbody>
              <tr v-for="player in strength.players" :key="player.id">
                <td :class="player.team.toLowerCase()">{{ player.team }}</td>
                <td>#{{ player.id }} {{ playerDisplayNames[player.id] || playerSurnameForId(player.id) }}</td>
                <td>{{ signed(player.rating) }}</td>
                <td class="breakdown"><span v-for="skill in player.contributions" :key="skill.key">{{ skill.label }} {{ signed(skill.value) }}</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </details>
    </template>
  </section>
</template>

<style scoped>
.team-strength { margin: 1rem 0; padding: 1rem; border: 1px solid #415c80; border-radius: 10px; background: #101e32; color: #edf3ff; }
h3 { margin: 0 0 .8rem; font-size: 1.05rem; text-transform: none; letter-spacing: normal; }
.totals { display: flex; flex-wrap: wrap; gap: 1rem 2.5rem; align-items: center; }
.totals > div { display: grid; gap: .25rem; }
.totals strong { font-size: 1.5rem; font-variant-numeric: tabular-nums; }
.totals .advantage strong { font-size: 1.1rem; }
.blue { color: #8bc7ff; } .red { color: #ffa5ac; }
.note, .advantage span { color: #b4c5df; font-size: .82rem; line-height: 1.5; }
.notice { color: #ffe1a8; line-height: 1.5; }
summary { cursor: pointer; color: #c8dcfc; }
.table-scroll { overflow: auto; margin-top: .6rem; }
table { width: 100%; border-collapse: collapse; font-size: .83rem; }
th, td { padding: .55rem; text-align: left; vertical-align: top; border-bottom: 1px solid #283d59; }
th { color: #c8dcfc; } td:not(.breakdown) { white-space: nowrap; font-variant-numeric: tabular-nums; }
.breakdown { min-width: 220px; } .breakdown span { display: inline-block; margin: 0 .8rem .3rem 0; }
</style>
