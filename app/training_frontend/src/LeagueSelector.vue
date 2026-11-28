<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { api } from './api'
const props = defineProps({ modelValue: {type: Object, default: null} })
const emit = defineEmits(['update:modelValue'])
const versions = ref([])
const snapshot = ref(null)
const error = ref('')
const listError = ref('')
const notice = ref('')
const key = ref('')
const refKey = value => value ? `${value.league_id}:${value.version}` : ''
let selectionRequest = 0
const choices = computed(() => {
  if (!snapshot.value || !props.modelValue || versions.value.some(v => refKey(v) === refKey(props.modelValue))) return versions.value
  return [...versions.value, {...props.modelValue, name: `${snapshot.value.name} (archived; restore before a new run)`}]
})
watch(() => props.modelValue, async value => {
  const request = ++selectionRequest
  key.value = refKey(value)
  snapshot.value = null
  error.value = ''
  if (!value) return
  notice.value = ''
  try {
    const loaded = await api.getLeagueVersion(value.league_id, value.version)
    if (request === selectionRequest) snapshot.value = loaded
  } catch (e) {
    if (request !== selectionRequest) return
    if (e.status === 404) {
      notice.value = 'The previously selected league or version is no longer available. Select a published league version below; your other run settings are preserved.'
      emit('update:modelValue', null)
    } else error.value = e.message
  }
}, {deep: true, immediate: true})
onMounted(async () => {
  try {
    const leagues = await api.listLeagues()
    const details = await Promise.all(leagues.map(l => api.getLeague(l.league_id)))
    versions.value = details.flatMap(l => l.versions.filter(v => !v.archived).map(v => ({...v, name: l.draft.name})))
    key.value = refKey(props.modelValue)
  } catch (e) { listError.value = e.message }
})
function choose() {
  notice.value = ''
  const v = choices.value.find(v => refKey(v) === key.value)
  emit('update:modelValue', v ? {league_id: v.league_id, version: v.version, sha256: v.sha256} : null)
}
</script>
<template>
  <section class="league-selection wide">
    <p v-if="notice" role="status">{{ notice }}</p>
    <label>Published league version<select aria-label="Published league version" v-model="key" @change="choose"><option value="">Select a published league version</option><option v-for="v in choices" :key="refKey(v)" :value="refKey(v)">{{ v.name }} · v{{ v.version }}</option></select></label>
    <p>The selected published version supplies the persistent teams and fresh-team skill distributions.</p>
    <p v-if="modelValue">Selected version {{ modelValue.version }} remains pinned; later league edits will not change it.</p>
    <details v-if="snapshot"><summary>Fresh-team skill distributions</summary><p v-for="skill in snapshot.skill_descriptors" :key="skill.key"><strong>{{ skill.display_label }}:</strong> {{ Object.entries(snapshot.sampling[skill.key]).map(([k, v]) => `${k} = ${v}`).join(', ') }}</p></details>
    <p v-if="error" role="alert">{{ error }}</p>
    <p v-if="listError" role="alert">{{ listError }}</p>
  </section>
</template>
<style scoped>
.league-selection { border: 1px solid #344158; border-radius: 10px; padding: 16px; margin: 14px 0; }
label { display: flex; flex-direction: column; gap: 8px; }
p { line-height: 1.5; color: #b6c2d2; }
</style>
