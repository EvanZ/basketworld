<script setup>
import { computed, onMounted, onBeforeUnmount, ref } from 'vue'
import { opponentChartData } from './pfsp'
import { opponentUsagePlot } from './opponentUsagePlot'

const props = defineProps({ metrics: { type: Array, default: () => [] } })
const emit = defineEmits(['select-metric'])
const chart = ref(null)
const width = ref(640)
const hoveredId = ref(null)
const view = ref('current')
let observer
onMounted(() => {
  observer = new ResizeObserver(entries => { width.value = Math.max(280, entries[0].contentRect.width) })
  observer.observe(chart.value)
})
onBeforeUnmount(() => observer?.disconnect())
const data = computed(() => opponentChartData(props.metrics, { basis: view.value }))
const hovered = computed(() => data.value.rows.find(row => row.opponent === hoveredId.value))
const plot = computed(() => opponentUsagePlot(data.value.rows, width.value))
const percent = value => `${Number(value).toFixed(value < 1 ? 2 : 1)}%`
const ppp = value => Number.isFinite(value) ? value.toFixed(2) : '—'
const count = value => Number.isFinite(value) ? Number(value).toLocaleString() : '—'
const description = row => {
  const assignmentBasis = data.value.update !== null ? 'newly assigned games' : 'assigned games'
  return `${row.opponent} · U${row.update} · usage ${percent(row.usage)} (${count(row.assignedGames)} of ${count(data.value.games)} ${assignmentBasis})` +
    ` · PPP scored ${ppp(row.pppScored)} (${count(row.offensivePossessions)} learner offensive possessions)` +
    ` · PPP allowed ${ppp(row.pppAllowed)} (${count(row.defensivePossessions)} opponent offensive possessions)`
}
const pppBasis = computed(() => view.value === 'current'
  ? 'Current training update only'
  : 'Pooled training history since tracking began')
</script>

<template>
  <section ref="chart" class="opponent-usage-chart">
    <div class="detail-section-heading">
      <h3>Opponent usage and PPP</h3>
      <span>{{ data.rows.length }} opponents<span v-if="data.capacity"> / {{ data.capacity }} slots</span></span>
    </div>
    <div class="opponent-chart-controls">
      <label>PPP view
        <select v-model="view" aria-label="Opponent chart view">
          <option value="current">Current update</option>
          <option value="pooled">Pooled (cumulative)</option>
        </select>
      </label>
    </div>
    <p v-if="data.update !== null">Training U{{ data.update.toLocaleString() }} · {{ data.games.toLocaleString() }} newly assigned games</p>
    <p v-else>{{ data.basis }}</p>
    <p v-if="view === 'current' && !data.currentAvailable" class="usage-notice" role="status">
      Current-update PPP is unavailable. Pause and resume the run with the updated trainer to collect it; pooled PPP remains available.
    </p>
    <div class="usage-legend" aria-label="Chart legend">
      <span><i class="usage-swatch" />Usage % · left axis</span>
      <span><i class="ppp-swatch scored" />PPP scored · right axis</span>
      <span><i class="ppp-swatch allowed" />PPP allowed · right axis</span>
    </div>
    <svg v-if="data.rows.length && (data.games > 0 || plot.pppTicks.length)" :class="{ dense: data.rows.length > 100 }" :viewBox="`0 0 ${width} 258`" role="img" :aria-label="`Opponent snapshot update versus usage percentage and PPP scored and allowed. ${pppBasis}.`">
      <title>Opponent usage and PPP by snapshot update · {{ pppBasis }}</title>
      <desc>Bars: observed game assignments, {{ data.basis }}, left percentage axis. Lines: {{ pppBasis }}, modes combined by possession counts, right PPP axis. Missing or zero-possession PPP is not plotted.</desc>
      <g v-for="tick in plot.yTicks" :key="`y-${tick}`">
        <line :x1="plot.left" :x2="plot.right" :y1="plot.y(tick)" :y2="plot.y(tick)" class="usage-grid" />
        <text :x="plot.left - 8" :y="plot.y(tick) + 4" text-anchor="end">{{ percent(tick) }}</text>
      </g>
      <g v-for="tick in plot.xTicks" :key="`x-${tick}`">
        <text :x="plot.x(tick)" :y="plot.bottom + 20" text-anchor="middle">{{ tick.toLocaleString() }}</text>
      </g>
      <g v-for="tick in plot.pppTicks" :key="`ppp-${tick}`">
        <text :x="plot.right + 8" :y="plot.pppY(tick) + 4">{{ tick.toFixed(1) }}</text>
      </g>
      <rect v-for="bar in plot.bars" :key="bar.opponent" class="usage-bar"
        :x="bar.x - bar.width / 2" :y="bar.y" :width="bar.width" :height="Math.max(0, plot.bottom - bar.y)"
        tabindex="0" role="button" :aria-label="`${description(bar)}; open usage history`"
        @mouseenter="hoveredId = bar.opponent" @focus="hoveredId = bar.opponent" @click="emit('select-metric', bar.name)"
        @keydown.enter="emit('select-metric', bar.name)">
        <title>{{ description(bar) }}</title>
      </rect>
      <g v-for="series in plot.series" :key="series.key" :class="['ppp-series', series.style]">
        <path :d="series.path" class="ppp-line" />
        <circle v-for="point in series.points" :key="point.opponent" :cx="point.x" :cy="point.y" r="3"
          tabindex="0" role="img" :aria-label="description(point)"
          @mouseenter="hoveredId = point.opponent" @focus="hoveredId = point.opponent">
          <title>{{ description(point) }}</title>
        </circle>
      </g>
      <g v-if="plot.bars.length <= 6 && plot.uniqueUpdates">
        <text v-for="bar in plot.bars" :key="`label-${bar.opponent}`" :x="bar.x" :y="bar.y - 6" text-anchor="middle">{{ percent(bar.usage) }}</text>
      </g>
      <text :x="(plot.left + plot.right) / 2" y="248" text-anchor="middle">Opponent snapshot update</text>
      <text transform="translate(13 110) rotate(-90)" text-anchor="middle">Usage (%)</text>
      <text v-if="plot.pppTicks.length" :transform="`translate(${width - 10} 110) rotate(90)`" text-anchor="middle">PPP</text>
    </svg>
    <p v-else>No opponent games assigned in this sample yet.</p>
    <p v-if="hovered" class="usage-caption">
      {{ description(hovered) }}
      <template v-if="view === 'pooled' && hovered.pppStartUpdate != null"> · PPP tracked since U{{ hovered.pppStartUpdate.toLocaleString() }}</template>
    </p>
    <p class="usage-caption">Observed assignments · {{ data.basis }} · excludes legal-random games</p>
    <p class="usage-caption">{{ pppBasis }} · sampled/deterministic modes combined. No completed possessions means a gap, not zero PPP.</p>
  </section>
</template>

<style scoped>
.opponent-usage-chart { min-width: 0; margin-bottom: 1rem; }
.opponent-usage-chart svg { display: block; width: 100%; height: auto; overflow: visible; }
.opponent-usage-chart text { fill: currentColor; font-size: 12px; }
.opponent-usage-chart p { margin: 0.4rem 0; }
.opponent-chart-controls { display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0; }
.opponent-chart-controls label { display: grid; gap: 5px; font-size: 12px; }
.opponent-chart-controls select { padding: 6px 9px; border: 1px solid #31547b; border-radius: 6px; color: inherit; background: #102039; max-width: 100%; }
.usage-notice { color: #fbbf24; font-size: 13px; }
.usage-grid { stroke: currentColor; opacity: 0.15; }
.usage-bar { fill: #60a5fa; cursor: pointer; }
.usage-bar:hover, .usage-bar:focus { fill: #93c5fd; }
.usage-legend { display: flex; flex-wrap: wrap; gap: 8px 18px; margin: 12px 0; font-size: 12px; }
.usage-legend span { display: inline-flex; align-items: center; gap: 7px; }
.usage-swatch { width: 12px; height: 10px; background: #60a5fa; }
.ppp-swatch { width: 20px; border-top: 3px solid currentColor; }
.scored { color: #6ee7b7; }
.allowed { color: #fbbf24; }
.ppp-swatch.allowed { border-top-style: dashed; }
.ppp-line { fill: none; stroke: currentColor; stroke-width: 2; pointer-events: none; }
.allowed .ppp-line { stroke-dasharray: 6 4; }
.ppp-series circle { fill: currentColor; stroke: #060b18; stroke-width: 1; }
.dense .ppp-series circle { fill: transparent; stroke: transparent; }
.ppp-series circle:hover, .ppp-series circle:focus { stroke: #fff; stroke-width: 2; }
.dense .ppp-series circle:hover, .dense .ppp-series circle:focus { fill: currentColor; }
.usage-caption { font-size: 0.85rem; opacity: 0.85; }
</style>
