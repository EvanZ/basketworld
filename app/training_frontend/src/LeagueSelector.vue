<script setup>
import { onMounted, ref, watch } from 'vue'
import { api } from './api'
const props = defineProps({ modelValue: {type: Object, default: null} })
const emit = defineEmits(['update:modelValue'])
const versions = ref([])
const error = ref('')
const key = ref('')
const refKey = value => value ? `${value.league_id}:${value.version}` : ''
watch(() => props.modelValue, value => { key.value = refKey(value) }, {deep: true})
onMounted(async () => {
  try {
    const leagues = await api.listLeagues()
    const details = await Promise.all(leagues.map(l => api.getLeague(l.league_id)))
    versions.value = details.flatMap(l => l.versions.filter(v => !v.archived).map(v => ({...v, name: l.draft.name})))
    if (props.modelValue && !versions.value.some(v => refKey(v) === refKey(props.modelValue))) {
      const saved = await api.getLeagueVersion(props.modelValue.league_id, props.modelValue.version)
      versions.value.push({...props.modelValue, name: `${saved.name} (archived; restore before a new run)`})
    }
    key.value = refKey(props.modelValue)
  } catch (e) { error.value = e.message }
})
function choose() {
  const v = versions.value.find(v => refKey(v) === key.value)
  emit('update:modelValue', v ? {league_id: v.league_id, version: v.version, sha256: v.sha256} : null)
}
</script>
<template>
  <section class="league-selection wide">
    <label>Published league version<select aria-label="Published league version" v-model="key" @change="choose"><option value="">No league — current random skill sampling</option><option v-for="v in versions" :key="refKey(v)" :value="refKey(v)">{{ v.name }} · v{{ v.version }}</option></select></label>
    <p>League-backed training is unavailable until #85 slice 2. You can inspect a saved selection now; clear it to launch with current random sampling.</p>
    <p v-if="modelValue"><strong>Launch disabled:</strong> selected version {{ modelValue.version }} remains pinned; later league edits will not change it.</p>
    <p v-if="error" role="alert">{{ error }}</p>
  </section>
</template>
<style scoped>
.league-selection { border: 1px solid #344158; border-radius: 10px; padding: 16px; margin: 14px 0; }
label { display: flex; flex-direction: column; gap: 8px; }
p { line-height: 1.5; color: #b6c2d2; }
</style>
