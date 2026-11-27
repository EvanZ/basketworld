<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { api } from './api'

const leagues = ref([])
const options = ref(null)
const selected = ref(null)
const draft = ref(null)
const snapshot = ref(null)
const teamId = ref('')
const playerId = ref('')
const movingPlayer = ref(false)
const moveTarget = ref('')
const deletingLeague = ref(false)
const deleteName = ref('')
const error = ref('')
const status = ref('')
const busy = ref(false)
const showArchived = ref(false)
const creating = ref(false)
const profileIndex = ref(0)
const creation = ref({name: '', team_count: 64, team_size: 5, seed: 0, rules: {}, sampling: {}})
const playerQuery = ref('')
const playerPage = ref(0)
const dirty = ref(false)
let timer = null
let saving = null
const copy = value => JSON.parse(JSON.stringify(value))
const shown = computed(() => snapshot.value || draft.value)
const readOnly = computed(() => Boolean(snapshot.value || selected.value?.archived))
const roster = computed(() => shown.value?.rosters.find(t => t.id === teamId.value))
const player = computed(() => shown.value?.players.find(p => p.id === playerId.value))
watch([playerId, () => shown.value?.league_id, readOnly], () => { movingPlayer.value = false })
function teamName(id) {
  return shown.value?.rosters.find(t => t.id === id)?.name || 'Unassigned'
}
function startMove() {
  moveTarget.value = members.value.get(playerId.value)?.team_id || ''
  movingPlayer.value = true
}
function confirmMove() {
  if (readOnly.value || busy.value || !player.value) return
  if (moveTarget.value === (members.value.get(playerId.value)?.team_id || '')) return
  movePlayer(playerId.value, moveTarget.value)
  movingPlayer.value = false
}
const members = computed(() => new Map((shown.value?.rosters || []).flatMap(t => t.members.map(m => [m.player_id, {...m, team_id: t.id}]))))
const filteredPlayers = computed(() => (shown.value?.players || []).filter(p => {
  const member = members.value.get(p.id)
  return (teamId.value === 'all' || (teamId.value === 'unassigned' ? !member : member?.team_id === teamId.value)) &&
    (!playerQuery.value || p.name.toLowerCase().includes(playerQuery.value.toLowerCase()))
}))
const displayedPlayers = computed(() => filteredPlayers.value.slice(playerPage.value * 25, (playerPage.value + 1) * 25))
watch([teamId, playerQuery, () => shown.value?.league_id], () => { playerPage.value = 0 })
watch(() => filteredPlayers.value.length, count => { playerPage.value = Math.min(playerPage.value, Math.max(0, Math.ceil(count / 25) - 1)) })
const unassignedCount = computed(() => (shown.value?.players || []).filter(p => !members.value.has(p.id)).length)
const activeSkills = computed(() => shown.value?.skill_descriptors || [])
const profile = computed(() => options.value?.profiles[profileIndex.value])
function inactiveSkill(key, rules) {
  return (key === 'rim_protection' && !rules.rim_protection_enabled) || (key === 'contested_shooting' && !rules.contested_shooting_enabled)
}
function useProfile() {
  creation.value.rules = copy(profile.value.rules)
  creation.value.sampling = copy(profile.value.sampling)
}
async function refresh() {
  try { leagues.value = await api.listLeagues(showArchived.value) } catch (e) { error.value = e.message }
}
function payload() {
  return {revision: selected.value.revision, name: draft.value.name,
    players: draft.value.players.map(({id, name, skills}) => ({id, name, skills})),
    rosters: draft.value.rosters.map(({id, name, members}) => ({id, name, members}))}
}
function scheduleSave() {
  dirty.value = true
  status.value = 'Unsaved changes'
  window.clearTimeout(timer)
  timer = window.setTimeout(saveDraft, 350)
}
async function saveDraft() {
  window.clearTimeout(timer)
  if (saving) return saving
  if (!dirty.value || readOnly.value) return !dirty.value
  saving = (async () => {
    try {
      while (dirty.value) {
        dirty.value = false
        status.value = 'Saving…'
        const result = await api.saveLeagueDraft(draft.value.league_id, copy(payload()))
        selected.value.revision = result.revision
        // Preserve edits made while this request was in flight.
        if (!dirty.value) draft.value = result.draft
      }
      status.value = 'All changes saved'
      error.value = ''
      await refresh()
      return true
    } catch (e) {
      dirty.value = true
      status.value = 'Changes not saved'
      error.value = e.message
      return false
    } finally { saving = null }
  })()
  return saving
}
async function openLeague(id) {
  if (!(await saveDraft())) return
  busy.value = true
  try {
    selected.value = await api.getLeague(id)
    deletingLeague.value = false
    deleteName.value = ''
    draft.value = copy(selected.value.draft)
    snapshot.value = null
    teamId.value = draft.value.rosters[0]?.id || 'all'
    playerId.value = ''
    error.value = ''; status.value = 'All changes saved'; creating.value = false
  } catch (e) { error.value = e.message } finally { busy.value = false }
}
async function create() {
  busy.value = true
  try {
    const result = await api.createLeague(creation.value)
    await refresh()
    await openLeague(result.draft.league_id)
  } catch (e) { error.value = e.message } finally { busy.value = false }
}
async function publish() {
  if (!(await saveDraft())) return
  busy.value = true
  try {
    const result = await api.publishLeague(draft.value.league_id, selected.value.revision)
    selected.value = await api.getLeague(draft.value.league_id)
    status.value = `Published v${result.version}. Draft changes will require another version.`
    error.value = ''
  } catch (e) { error.value = e.message } finally { busy.value = false }
}
async function inspectVersion(version) {
  if (!(await saveDraft())) return
  busy.value = true
  try {
    snapshot.value = version === null ? null : await api.getLeagueVersion(draft.value.league_id, version)
    playerId.value = ''
    error.value = ''
  } catch (e) { error.value = e.message } finally { busy.value = false }
}
async function archive(archived, version = null) {
  if (!(await saveDraft())) return
  busy.value = true
  try {
    selected.value = await api.archiveLeague(draft.value.league_id, archived, version)
    await refresh()
    error.value = ''
  } catch (e) { error.value = e.message } finally { busy.value = false }
}
async function startDelete() {
  if (!(await saveDraft())) return
  deleteName.value = ''
  deletingLeague.value = true
}
async function deleteLeague() {
  if (busy.value || !deletingLeague.value || deleteName.value !== draft.value.name) return
  if (!(await saveDraft())) return
  busy.value = true
  try {
    await api.deleteLeague(draft.value.league_id, selected.value.revision, deleteName.value)
    selected.value = null; draft.value = null; snapshot.value = null
    playerId.value = ''; deletingLeague.value = false; deleteName.value = ''
    error.value = ''; status.value = ''
    await refresh()
  } catch (e) { error.value = e.message } finally { busy.value = false }
}
function movePlayer(id, target) {
  let jersey = 0
  for (const team of draft.value.rosters) {
    const existing = team.members.find(m => m.player_id === id)
    if (existing) jersey = existing.jersey_number
    team.members = team.members.filter(m => m.player_id !== id)
  }
  const destination = draft.value.rosters.find(t => t.id === target)
  if (destination) {
    const used = new Set(destination.members.map(m => m.jersey_number))
    if (used.has(jersey)) jersey = Array.from({length: 100}, (_, i) => i).find(n => !used.has(n)) ?? jersey
    destination.members.push({player_id: id, jersey_number: jersey})
  }
  scheduleSave()
}
function setJersey(id, value) {
  const member = draft.value.rosters.flatMap(t => t.members).find(m => m.player_id === id)
  if (member) { member.jersey_number = value === '' ? null : Number(value); scheduleSave() }
}
async function startCreation() {
  if (!(await saveDraft())) return
  creating.value = true
  error.value = ''
}
function preventLostEdits(event) {
  if (dirty.value || saving) { event.preventDefault(); event.returnValue = '' }
}
onMounted(async () => {
  window.addEventListener('beforeunload', preventLostEdits)
  try { options.value = await api.leagueOptions(); useProfile(); await refresh() } catch (e) { error.value = e.message }
})
onBeforeUnmount(() => {
  window.clearTimeout(timer)
  window.removeEventListener('beforeunload', preventLostEdits)
  if (dirty.value) saveDraft()
})
</script>

<template>
  <section class="league-manager">
    <p class="league-notice">Create reusable leagues here. League-backed training will be available in #85 slice 2. Season records and player statistics will come with competition evaluation.</p>
    <div v-if="error" role="alert" class="error-banner">{{ error }} <button v-if="dirty" @click="saveDraft">Retry save</button></div>
    <div class="league-toolbar">
      <label class="toggle"><input v-model="showArchived" type="checkbox" @change="refresh" /> Show archived leagues and versions</label>
      <button class="primary" :disabled="busy || !options" @click="startCreation">Create league</button>
    </div>
    <form v-if="creating && options" class="panel league-create" @submit.prevent="create">
      <h2>Create league</h2>
      <div class="league-fields">
        <label>League name<input v-model="creation.name" required maxlength="120" /></label>
        <label>Teams<input v-model.number="creation.team_count" type="number" min="2" max="512" required /></label>
        <label>Players per team<select v-model.number="creation.team_size"><option v-for="n in options.team_sizes" :value="n" :key="n">{{ n }}v{{ n }}</option></select></label>
        <label>Generation seed<input v-model.number="creation.seed" type="number" min="0" max="2147483647" required /></label>
        <label>Skill rules<select v-model.number="profileIndex" @change="useProfile"><option v-for="(p, i) in options.profiles" :key="i" :value="i">{{ p.label }}</option></select></label>
      </div>
      <details><summary>Player generation settings</summary>
        <p>Uses the existing skill sampler. Bounded shooting skills are clipped to 0–1. Names and jersey numbers use a separate seeded random stream.</p>
        <div v-for="skill in profile.skills" :key="skill.key" class="league-sampling">
          <strong>{{ skill.display_label }} <small v-if="inactiveSkill(skill.key, creation.rules)">(inactive under these rules)</small></strong>
          <label v-for="(_, key) in creation.sampling[skill.key]" :key="key">{{ key }}<input v-model.number="creation.sampling[skill.key][key]" type="number" step="any" required :aria-label="`${skill.display_label} ${key}`" /></label>
        </div>
      </details>
      <details><summary>Rule compatibility settings</summary><p>Runs must match these saved shot-defense rule settings. These do not change generated skills; generation distributions are configured above.</p>
        <div class="league-fields"><label v-for="key in Object.keys(creation.rules).filter(k => !k.endsWith('_enabled'))" :key="key">{{ key }}<input v-model.number="creation.rules[key]" type="number" step="any" required /></label></div>
      </details>
      <div class="league-toolbar"><span>{{ creation.team_count * creation.team_size }} players, randomly assigned to complete rosters.</span><button class="primary" :disabled="busy">{{ busy ? 'Generating…' : 'Generate league' }}</button><button type="button" @click="creating = false">Cancel</button></div>
    </form>
    <div class="league-layout">
      <aside class="panel league-list"><h2>Leagues</h2><p v-if="!leagues.length">No leagues yet.</p>
        <button v-for="league in leagues" :key="league.league_id" class="league-list-item" :class="{active: draft?.league_id === league.league_id}" :disabled="busy" @click="openLeague(league.league_id)">
          <strong>{{ league.name }}</strong><small>{{ league.team_count }} teams · {{ league.team_size }}v{{ league.team_size }}{{ league.archived ? ' · Archived' : '' }}</small>
        </button>
      </aside>
      <div v-if="shown" class="panel league-content">
        <div class="league-toolbar">
          <label>League name<input v-model="draft.name" :disabled="readOnly || busy" v-if="!snapshot" @input="scheduleSave" /><strong v-else>{{ snapshot.name }}</strong></label>
          <span role="status">{{ snapshot ? `Published v${snapshot.version} · Read only` : status }}</span>
          <button :disabled="busy || readOnly" @click="publish">Publish version</button>
          <button :disabled="busy" @click="archive(!selected.archived)">{{ selected.archived ? 'Restore league' : 'Archive league' }}</button>
          <button :disabled="busy" @click="startDelete">Delete league…</button>
        </div>
        <div v-if="deletingLeague" class="league-delete" role="group" aria-label="Delete league confirmation">
          <p>Permanently delete <strong>{{ draft.name }}</strong>, all its players, rosters, and published versions? This cannot be undone. Leagues used by training runs cannot be deleted.</p>
          <label>Type the league name to confirm<input v-model="deleteName" :disabled="busy" aria-label="League name to delete" /></label>
          <button :disabled="busy || deleteName !== draft.name" @click="deleteLeague">Permanently delete league</button>
          <button :disabled="busy" @click="deletingLeague = false">Cancel deletion</button>
        </div>
        <p>{{ shown.team_count }} teams · {{ shown.team_size }} players per team · Skill schema {{ shown.skill_schema_version }} · Seed {{ shown.seed }}</p>
        <details class="league-versions"><summary>Published versions ({{ selected.versions.length }})</summary>
          <button :disabled="busy" @click="inspectVersion(null)">Edit current draft</button>
          <div v-for="v in selected.versions.filter(v => showArchived || !v.archived)" :key="v.version" class="league-toolbar">
            <button :disabled="busy" @click="inspectVersion(v.version)">Inspect v{{ v.version }}{{ v.archived ? ' (archived)' : '' }}</button>
            <small>{{ v.published_at }} · {{ v.sha256.slice(0, 12) }}</small>
            <button :disabled="busy" @click="archive(!v.archived, v.version)">{{ v.archived ? 'Restore version' : 'Archive version' }}</button>
          </div>
        </details>
        <div class="league-toolbar">
          <label>Roster<select aria-label="Roster" v-model="teamId" @change="playerId = ''"><option value="all">All players</option><option value="unassigned">Unassigned ({{ unassignedCount }})</option><option v-for="team in shown.rosters" :key="team.id" :value="team.id">{{ team.name }} ({{ team.members.length }}/{{ shown.team_size }})</option></select></label>
          <label v-if="roster">Team name<input v-model="roster.name" :disabled="readOnly || busy" @input="scheduleSave" /></label>
          <label>Find player<input v-model="playerQuery" placeholder="Player name" /></label>
        </div>
        <div class="league-table-wrap"><table class="league-players"><thead><tr><th>Player</th><th>Jersey</th><th>Team</th></tr></thead><tbody>
          <tr v-for="p in displayedPlayers" :key="p.id" :class="{active: playerId === p.id}">
            <td><button @click="playerId = p.id">{{ p.name }}</button></td>
            <td><input v-if="members.has(p.id)" type="number" min="0" max="99" :value="members.get(p.id).jersey_number" :aria-label="`Jersey for ${p.name}`" :disabled="readOnly || busy" @change="setJersey(p.id, $event.target.value)" /><span v-else>—</span></td>
            <td>{{ teamName(members.get(p.id)?.team_id) }}</td>
          </tr>
        </tbody></table></div>
        <div v-if="filteredPlayers.length > 25" class="league-toolbar">
          <button :disabled="playerPage === 0" @click="playerPage--">Previous players</button>
          <span>Players {{ playerPage * 25 + 1 }}–{{ Math.min((playerPage + 1) * 25, filteredPlayers.length) }} of {{ filteredPlayers.length }}</span>
          <button :disabled="(playerPage + 1) * 25 >= filteredPlayers.length" @click="playerPage++">Next players</button>
        </div>
        <p v-if="!displayedPlayers.length">No players match this selection.</p>
        <section v-if="player" class="league-player-detail">
          <h3>Player skills</h3><label>Player name<input v-model="player.name" :disabled="readOnly || busy" @input="scheduleSave" /></label>
          <p>Team: {{ teamName(members.get(player.id)?.team_id) }}</p>
          <button v-if="!readOnly && !movingPlayer" :disabled="busy" @click="startMove">Move player…</button>
          <div v-if="movingPlayer && !readOnly" class="league-move">
            <label>Destination team<select aria-label="Destination team" v-model="moveTarget" :disabled="busy"><option value="">Unassigned</option><option v-for="team in shown.rosters" :key="team.id" :value="team.id">{{ team.name }}</option></select></label>
            <p>Move {{ player.name }} from {{ teamName(members.get(player.id)?.team_id) }} to {{ teamName(moveTarget) }}? This changes the draft roster.</p>
            <button :disabled="busy || moveTarget === (members.get(player.id)?.team_id || '')" @click="confirmMove">Confirm move</button>
            <button @click="movingPlayer = false">Cancel</button>
          </div>
          <small class="league-identity">Player ID: {{ player.id }}</small>
          <div class="league-fields"><label v-for="skill in activeSkills" :key="skill.key">{{ skill.display_label }}<span v-if="inactiveSkill(skill.key, shown.rules.parameters)"> (inactive)</span>
            <input v-model.number="player.skills[skill.key]" type="number" step="any" :min="skill.minimum ?? undefined" :max="skill.maximum ?? undefined" :disabled="readOnly || busy" @input="scheduleSave" :aria-label="`Player ${skill.display_label}`" />
            <small v-if="player.skills[skill.key] !== player.generated_skills[skill.key]">Override · generated {{ player.generated_skills[skill.key] }}</small>
          </label></div>
          <details><summary>Generation provenance</summary><pre>{{ JSON.stringify({provenance: player.provenance, sampling: shown.sampling, rules: shown.rules}, null, 2) }}</pre></details>
        </section>
        <p v-else>Select a player to inspect or edit their full skill profile.</p>
      </div>
    </div>
  </section>
</template>

<style scoped>
button:not(.primary) { background: #12213a; border: 1px solid #304764; border-radius: 8px; padding: 8px 12px; color: #e6edf7; cursor: pointer; }
button:not(.primary):hover:not(:disabled) { background: #1c3454; border-color: #58ccec; }
button:disabled { opacity: .5; cursor: default; }
.league-players td button { background: transparent; border-color: transparent; color: #59d5fa; text-align: left; }
.league-players tr.active { background: #13263e; }

.league-toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin: 12px 0; }
.league-layout { display: grid; grid-template-columns: 230px minmax(0, 1fr); gap: 18px; }
.league-list, .league-content, .league-create { padding: 20px; }
.league-content { min-width: 0; }
.league-list-item { display: flex; flex-direction: column; text-align: left; width: 100%; gap: 6px; margin: 8px 0; }
.league-list-item.active { border-color: #f7994a; }
.league-fields { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin: 16px 0; }
label { display: flex; flex-direction: column; gap: 6px; }
label.toggle { flex-direction: row; align-items: center; }
.league-notice { color: #b6c2d2; line-height: 1.5; }
.league-sampling { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; margin: 12px 0; }
.league-sampling strong { min-width: 170px; }
.league-sampling input { width: 110px; }
.league-table-wrap { overflow: auto; max-height: 420px; }
.league-players { width: 100%; }
.league-players input { max-width: 85px; }
.league-players select { max-width: 250px; }
.league-player-detail { margin-top: 20px; border-top: 1px solid #344158; padding-top: 14px; }
.league-identity { display: block; margin-top: 10px; overflow-wrap: anywhere; }
summary { cursor: pointer; margin: 16px 0; }
pre { max-height: 320px; overflow: auto; }
@media (max-width: 1050px) { .league-layout { grid-template-columns: 1fr; } }
</style>
