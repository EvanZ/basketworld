<script setup>
import { computed, ref } from 'vue';

const props = defineProps({
  links: {
    type: Array,
    default: () => [],
  },
  playerIds: {
    type: Array,
    default: () => [],
  },
  title: {
    type: String,
    default: 'Assist Sankey',
  },
  totalLabel: {
    type: String,
    default: 'Total',
  },
});

const CANVAS_WIDTH = 920;
const CANVAS_HEIGHT = 360;
const NODE_WIDTH = 24;
const LEFT_X = 140;
const RIGHT_X = CANVAS_WIDTH - LEFT_X - NODE_WIDTH;
const CURVE_SIZE = 240;
const TOP_PADDING = 28;
const BOTTOM_PADDING = 24;
const EXPORT_SCALE = 2;
const EXPORT_BACKGROUND = '#0b1222';

const svgRef = ref(null);
const isExporting = ref(false);

function toFiniteNumber(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function sourceColorForId(playerId) {
  const hue = ((Number(playerId) || 0) * 53 + 17) % 360;
  return `hsl(${hue} 74% 58%)`;
}

const normalizedLinks = computed(() =>
  (props.links || [])
    .map((raw) => {
      const source = toFiniteNumber(raw?.source);
      const target = toFiniteNumber(raw?.target);
      const count = toFiniteNumber(raw?.count);
      if (source === null || target === null || count === null) return null;
      return {
        source: Number(source),
        target: Number(target),
        count: Number(count),
      };
    })
    .filter((link) => link && link.count > 0 && link.source !== link.target),
);

const nodeIds = computed(() => {
  const fromProps = (props.playerIds || [])
    .map((id) => toFiniteNumber(id))
    .filter((id) => id !== null)
    .map((id) => Number(id));
  if (fromProps.length > 0) {
    return Array.from(new Set(fromProps)).sort((a, b) => a - b);
  }
  const set = new Set();
  for (const link of normalizedLinks.value) {
    set.add(link.source);
    set.add(link.target);
  }
  return Array.from(set).sort((a, b) => a - b);
});

const totalsByPlayer = computed(() => {
  const out = {};
  const incoming = {};
  for (const id of nodeIds.value) {
    out[id] = 0;
    incoming[id] = 0;
  }
  for (const link of normalizedLinks.value) {
    out[link.source] = (out[link.source] || 0) + link.count;
    incoming[link.target] = (incoming[link.target] || 0) + link.count;
  }
  return { out, incoming };
});

const totalAssists = computed(() =>
  normalizedLinks.value.reduce((sum, link) => sum + link.count, 0),
);

const layout = computed(() => {
  const ids = nodeIds.value;
  const n = ids.length;
  if (n === 0) {
    return {
      leftNodes: {},
      rightNodes: {},
      leftScale: 1,
      rightScale: 1,
      linkScale: 1,
    };
  }

  const gap = n > 1 ? 14 : 0;
  const availableHeight = Math.max(
    40,
    CANVAS_HEIGHT - TOP_PADDING - BOTTOM_PADDING - gap * (n - 1),
  );
  const minWeightUnits = 0.35;

  const { out, incoming } = totalsByPlayer.value;
  const leftWeightSum = ids.reduce(
    (sum, id) => sum + Math.max(minWeightUnits, Number(out[id] || 0)),
    0,
  );
  const rightWeightSum = ids.reduce(
    (sum, id) => sum + Math.max(minWeightUnits, Number(incoming[id] || 0)),
    0,
  );

  const leftScale = availableHeight / Math.max(1e-9, leftWeightSum);
  const rightScale = availableHeight / Math.max(1e-9, rightWeightSum);
  const linkScale = Math.min(leftScale, rightScale);

  const leftNodes = {};
  const rightNodes = {};

  let yCursor = TOP_PADDING;
  for (const id of ids) {
    const h = Math.max(minWeightUnits, Number(out[id] || 0)) * leftScale;
    leftNodes[id] = { y: yCursor, h };
    yCursor += h + gap;
  }

  yCursor = TOP_PADDING;
  for (const id of ids) {
    const h = Math.max(minWeightUnits, Number(incoming[id] || 0)) * rightScale;
    rightNodes[id] = { y: yCursor, h };
    yCursor += h + gap;
  }

  return { leftNodes, rightNodes, leftScale, rightScale, linkScale };
});

const leftNodesForRender = computed(() => {
  const { out } = totalsByPlayer.value;
  const leftNodes = layout.value.leftNodes;
  return nodeIds.value.map((id) => ({
    id,
    y: leftNodes[id]?.y || TOP_PADDING,
    h: leftNodes[id]?.h || 18,
    total: Number(out[id] || 0),
  }));
});

const rightNodesForRender = computed(() => {
  const { incoming } = totalsByPlayer.value;
  const rightNodes = layout.value.rightNodes;
  return nodeIds.value.map((id) => ({
    id,
    y: rightNodes[id]?.y || TOP_PADDING,
    h: rightNodes[id]?.h || 18,
    total: Number(incoming[id] || 0),
  }));
});

const renderedLinks = computed(() => {
  const { leftNodes, rightNodes, leftScale, rightScale, linkScale } = layout.value;
  const leftUsedUnits = {};
  const rightUsedUnits = {};
  for (const id of nodeIds.value) {
    leftUsedUnits[id] = 0;
    rightUsedUnits[id] = 0;
  }

  const sorted = [...normalizedLinks.value].sort(
    (a, b) =>
      a.source - b.source
      || a.target - b.target
      || b.count - a.count,
  );

  return sorted
    .map((link, idx) => {
      const leftNode = leftNodes[link.source];
      const rightNode = rightNodes[link.target];
      if (!leftNode || !rightNode) return null;

      const y1 = leftNode.y + (leftUsedUnits[link.source] + link.count / 2) * leftScale;
      const y2 = rightNode.y + (rightUsedUnits[link.target] + link.count / 2) * rightScale;
      leftUsedUnits[link.source] += link.count;
      rightUsedUnits[link.target] += link.count;

      const x1 = LEFT_X + NODE_WIDTH;
      const x2 = RIGHT_X;
      const c1 = x1 + CURVE_SIZE;
      const c2 = x2 - CURVE_SIZE;
      const path = `M ${x1} ${y1} C ${c1} ${y1}, ${c2} ${y2}, ${x2} ${y2}`;

      return {
        key: `${link.source}-${link.target}-${idx}`,
        source: link.source,
        target: link.target,
        count: link.count,
        thickness: Math.max(2, link.count * linkScale),
        color: sourceColorForId(link.source),
        path,
      };
    })
    .filter(Boolean);
});

const hasData = computed(() => totalAssists.value > 0 && renderedLinks.value.length > 0);

function buildExportFilename() {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `assist-sankey-${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}.png`;
}

function loadSvgImage(blobUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Failed to load SVG image for export'));
    img.src = blobUrl;
  });
}

function canvasToBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
        return;
      }
      reject(new Error('Failed to encode PNG'));
    }, 'image/png');
  });
}

async function downloadPng() {
  if (!hasData.value || isExporting.value) return;
  const svgEl = svgRef.value;
  if (!svgEl) return;

  isExporting.value = true;
  let svgUrl = null;
  let pngUrl = null;
  try {
    const serializer = new XMLSerializer();
    const svgMarkup = serializer.serializeToString(svgEl);
    const svgBlob = new Blob([svgMarkup], { type: 'image/svg+xml;charset=utf-8' });
    svgUrl = URL.createObjectURL(svgBlob);
    const image = await loadSvgImage(svgUrl);

    const canvas = document.createElement('canvas');
    canvas.width = CANVAS_WIDTH * EXPORT_SCALE;
    canvas.height = CANVAS_HEIGHT * EXPORT_SCALE;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Failed to acquire canvas context');
    ctx.scale(EXPORT_SCALE, EXPORT_SCALE);
    ctx.fillStyle = EXPORT_BACKGROUND;
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    ctx.drawImage(image, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    const pngBlob = await canvasToBlob(canvas);
    pngUrl = URL.createObjectURL(pngBlob);
    const anchor = document.createElement('a');
    anchor.href = pngUrl;
    anchor.download = buildExportFilename();
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
  } catch (err) {
    console.error('[AssistSankey] PNG export failed:', err);
  } finally {
    if (svgUrl) URL.revokeObjectURL(svgUrl);
    if (pngUrl) URL.revokeObjectURL(pngUrl);
    isExporting.value = false;
  }
}
</script>

<template>
  <div class="assist-sankey">
    <div class="assist-sankey-header">
      <h3>{{ title }}</h3>
      <div class="assist-sankey-header-right">
        <span class="assist-sankey-total">{{ totalLabel }}: {{ totalAssists }}</span>
        <button
          class="download-button"
          type="button"
          :disabled="!hasData || isExporting"
          @click="downloadPng"
          title="Download Sankey as PNG"
          aria-label="Download Sankey as PNG"
        >
          📥
        </button>
      </div>
    </div>

    <div v-if="!hasData" class="assist-sankey-empty">
      No assisted makes recorded in this evaluation.
    </div>

    <svg
      v-else
      ref="svgRef"
      class="assist-sankey-svg"
      :viewBox="`0 0 ${CANVAS_WIDTH} ${CANVAS_HEIGHT}`"
      role="img"
      aria-label="Assist Sankey diagram"
    >
      <text class="axis-label" :x="LEFT_X - 44" :y="18" fill="#94a3b8" font-size="12">Passer</text>
      <text
        class="axis-label"
        :x="RIGHT_X + NODE_WIDTH + 44"
        :y="18"
        text-anchor="end"
        fill="#94a3b8"
        font-size="12"
      >
        Shooter
      </text>

      <g class="link-group">
        <path
          v-for="link in renderedLinks"
          :key="link.key"
          :d="link.path"
          :stroke="link.color"
          :stroke-width="link.thickness"
          fill="none"
          stroke-linecap="round"
          opacity="0.55"
        >
          <title>Player {{ link.source }} to Player {{ link.target }}: {{ link.count }}</title>
        </path>
      </g>

      <g class="node-group">
        <g v-for="node in leftNodesForRender" :key="`left-${node.id}`">
          <rect
            :x="LEFT_X"
            :y="node.y"
            :width="NODE_WIDTH"
            :height="node.h"
            :fill="sourceColorForId(node.id)"
            rx="4"
          />
          <text
            :x="LEFT_X - 12"
            :y="node.y + (node.h / 2) + 4"
            class="node-label"
            text-anchor="end"
            fill="#dbe3f0"
            font-size="12"
          >
            P{{ node.id }} ({{ node.total }})
          </text>
        </g>

        <g v-for="node in rightNodesForRender" :key="`right-${node.id}`">
          <rect
            :x="RIGHT_X"
            :y="node.y"
            :width="NODE_WIDTH"
            :height="node.h"
            :fill="sourceColorForId(node.id)"
            rx="4"
          />
          <text
            :x="RIGHT_X + NODE_WIDTH + 12"
            :y="node.y + (node.h / 2) + 4"
            class="node-label"
            text-anchor="start"
            fill="#dbe3f0"
            font-size="12"
          >
            P{{ node.id }} ({{ node.total }})
          </text>
        </g>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.assist-sankey {
  width: min(1000px, 100%);
  background: rgba(10, 18, 34, 0.56);
  border: 1px solid rgba(148, 163, 184, 0.28);
  border-radius: 14px;
  padding: 0.75rem 0.9rem 0.8rem;
}

.assist-sankey-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 0.45rem;
}

.assist-sankey-header-right {
  display: flex;
  align-items: center;
  gap: 0.6rem;
}

.assist-sankey-header h3 {
  margin: 0;
  color: #cbd5e1;
  font-size: 0.95rem;
  letter-spacing: 0.03em;
  text-transform: uppercase;
}

.assist-sankey-total {
  color: #94a3b8;
  font-size: 0.84rem;
}

.download-button {
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid #ddd;
  border-radius: 6px;
  padding: 8px 12px;
  cursor: pointer;
  font-size: 2rem;
  color: #333;
  transition: all 0.2s ease;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  line-height: 1;
}

.download-button:hover {
  background: rgb(13, 9, 223);
  box-shadow: 0 4px 8px rgba(0,0,0,0.15);
  transform: translateY(-1px);
}

.download-button:active {
  transform: translateY(0);
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.download-button:disabled {
  opacity: 0.45;
  cursor: not-allowed;
  transform: none;
}

.assist-sankey-empty {
  color: #94a3b8;
  font-size: 0.9rem;
  text-align: center;
  padding: 0.85rem 0.2rem;
}

.assist-sankey-svg {
  width: 100%;
  height: auto;
  display: block;
  overflow: visible;
}

.axis-label {
  fill: #94a3b8;
  font-size: 12px;
  letter-spacing: 0.03em;
}

.node-label {
  fill: #dbe3f0;
  font-size: 12px;
}
</style>
