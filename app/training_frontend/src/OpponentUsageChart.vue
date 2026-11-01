<script setup>
import { computed, onMounted, onBeforeUnmount, ref } from 'vue'
import { opponentUsage } from './pfsp'

const props = defineProps({ metrics: { type: Array, default: () => [] } })
const emit = defineEmits(['select-metric'])
const chart = ref(null)
const width = ref(640)
const hoveredId = ref(null)
let observer
onMounted(() => {
  observer = new ResizeObserver(entries => { width.value = Math.max(280, entries[0].contentRect.width) })
  observer.observe(chart.value)
})
onBeforeUnmount(() => observer?.disconnect())
const data = computed(() => opponentUsage(props.metrics))
const hovered = computed(() => data.value.rows.find(row => row.opponent === hoveredId.value))
const plot = computed(() => {
  const rows = data.value.rows
  const left = 62, right = width.value - 16, bottom = 202, top = 18
  const first = rows[0]?.update ?? 0, last = rows.at(-1)?.update ?? 1
  const span = Math.max(1, last - first)
  const gap = rows.length > 1 ? Math.min(...rows.slice(1).map((row, i) => Math.max(1, row.update - rows[i].update))) : span
  const lo = first - gap * 0.6, hi = last + gap * 0.6
  const x = value => left + (value - lo) / (hi - lo) * (right - left)
  const rawMaximum = Math.min(100, Math.max(0.1, ...rows.map(row => row.usage * 1.15)))
  const magnitude = 10 ** Math.floor(Math.log10(rawMaximum / 4))
  const ratio = rawMaximum / (4 * magnitude)
  const tickStep = (ratio <= 1 ? 1 : ratio <= 2 ? 2 : ratio <= 5 ? 5 : 10) * magnitude
  const maximum = Math.min(100, Math.ceil(rawMaximum / tickStep) * tickStep)
  const y = value => bottom - value / maximum * (bottom - top)
  const barWidth = Math.max(1, Math.min(44, gap / (hi - lo) * (right - left) * 0.7))
  const tickCount = width.value < 450 ? 3 : 5
  const xTicks = [...new Set(Array.from({ length: tickCount }, (_, i) => rows[Math.round((rows.length - 1) * i / (tickCount - 1))]?.update ?? 0))]
  const groups = new Map()
  for (const row of rows) groups.set(row.update, [...(groups.get(row.update) || []), row.opponent])
  return { left, right, bottom, top, x, y, xTicks, uniqueUpdates: groups.size === rows.length,
    yTicks: Array.from({ length: Math.round(maximum / tickStep) + 1 }, (_, i) => i * tickStep),
    bars: rows.map(row => {
      const group = groups.get(row.update)
      return { ...row, x: x(row.update) + ((group.indexOf(row.opponent) + 0.5) / group.length - 0.5) * barWidth,
        y: y(row.usage), width: Math.max(1, barWidth / group.length * (group.length > 1 ? 0.9 : 1)) }
    }) }
})
const percent = value => `${Number(value).toFixed(value < 1 ? 2 : 1)}%`
</script>

<template>
  <section ref="chart" class="opponent-usage-chart">
    <div class="detail-section-heading">
      <h3>Opponent usage</h3>
      <span>{{ data.rows.length }} opponents<span v-if="data.capacity"> / {{ data.capacity }} slots</span></span>
    </div>
    <p v-if="data.update !== null">Training U{{ data.update.toLocaleString() }} · {{ data.games.toLocaleString() }} newly assigned games</p>
    <p v-else>{{ data.basis }}</p>
    <svg v-if="data.rows.length && data.games > 0" :viewBox="`0 0 ${width} 258`" role="img" aria-label="Opponent snapshot update versus percentage of assigned games">
      <title>Opponent usage by snapshot update</title>
      <desc>Observed game assignments, not PFSP selection probabilities. {{ data.basis }}.</desc>
      <g v-for="tick in plot.yTicks" :key="`y-${tick}`">
        <line :x1="plot.left" :x2="plot.right" :y1="plot.y(tick)" :y2="plot.y(tick)" class="usage-grid" />
        <text :x="plot.left - 8" :y="plot.y(tick) + 4" text-anchor="end">{{ percent(tick) }}</text>
      </g>
      <g v-for="tick in plot.xTicks" :key="`x-${tick}`">
        <text :x="plot.x(tick)" :y="plot.bottom + 20" text-anchor="middle">{{ tick.toLocaleString() }}</text>
      </g>
      <rect v-for="bar in plot.bars" :key="bar.opponent" class="usage-bar"
        :x="bar.x - bar.width / 2" :y="bar.y" :width="bar.width" :height="Math.max(0, plot.bottom - bar.y)"
        tabindex="0" role="button" :aria-label="`${bar.opponent}: ${percent(bar.usage)}; open history`"
        @mouseenter="hoveredId = bar.opponent" @focus="hoveredId = bar.opponent" @click="emit('select-metric', bar.name)"
        @keydown.enter="emit('select-metric', bar.name)">
        <title>{{ bar.opponent }} · U{{ bar.update }} · {{ percent(bar.usage) }}</title>
      </rect>
      <g v-if="plot.bars.length <= 6 && plot.uniqueUpdates">
        <text v-for="bar in plot.bars" :key="`label-${bar.opponent}`" :x="bar.x" :y="bar.y - 6" text-anchor="middle">{{ percent(bar.usage) }}</text>
      </g>
      <text :x="(plot.left + plot.right) / 2" y="248" text-anchor="middle">Opponent snapshot update</text>
      <text transform="translate(13 110) rotate(-90)" text-anchor="middle">Usage (%)</text>
    </svg>
    <p v-else>No opponent games assigned in this sample yet.</p>
    <p v-if="hovered" class="usage-caption">{{ hovered.opponent }} · {{ percent(hovered.usage) }} of assigned games</p>
    <p class="usage-caption">Observed assignments · {{ data.basis }} · excludes legal-random games</p>
  </section>
</template>

<style scoped>
.opponent-usage-chart { min-width: 0; margin-bottom: 1rem; }
.opponent-usage-chart svg { display: block; width: 100%; height: auto; overflow: visible; }
.opponent-usage-chart text { fill: currentColor; font-size: 12px; }
.opponent-usage-chart p { margin: 0.4rem 0; }
.usage-grid { stroke: currentColor; opacity: 0.15; }
.usage-bar { fill: #60a5fa; cursor: pointer; }
.usage-bar:hover, .usage-bar:focus { fill: #93c5fd; }
.usage-caption { font-size: 0.85rem; opacity: 0.85; }
</style>
