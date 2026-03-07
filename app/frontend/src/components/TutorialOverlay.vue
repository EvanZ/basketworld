<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';

const props = defineProps({
  active: {
    type: Boolean,
    default: false,
  },
  step: {
    type: Object,
    default: null,
  },
  stepIndex: {
    type: Number,
    default: 0,
  },
  totalSteps: {
    type: Number,
    default: 0,
  },
  canGoBack: {
    type: Boolean,
    default: false,
  },
  statusText: {
    type: String,
    default: '',
  },
  canAdvance: {
    type: Boolean,
    default: true,
  },
});

const emit = defineEmits(['next', 'back', 'skip']);

const targetRect = ref(null);
const viewport = ref({ width: 1280, height: 720 });

function safeSelectorValue(raw) {
  return String(raw || '')
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"');
}

function refreshViewport() {
  if (typeof window === 'undefined') return;
  viewport.value = {
    width: Number(window.innerWidth || 1280),
    height: Number(window.innerHeight || 720),
  };
}

function updateTargetRect() {
  if (!props.active) {
    targetRect.value = null;
    return;
  }
  const targetId = String(props.step?.targetId || '');
  if (!targetId) {
    targetRect.value = null;
    return;
  }
  const selector = `[data-tutorial-id="${safeSelectorValue(targetId)}"]`;
  const el = document.querySelector(selector);
  if (!el || typeof el.getBoundingClientRect !== 'function') {
    targetRect.value = null;
    return;
  }
  const rect = el.getBoundingClientRect();
  const pad = 8;
  targetRect.value = {
    top: Math.max(8, rect.top - pad),
    left: Math.max(8, rect.left - pad),
    width: Math.max(20, rect.width + (pad * 2)),
    height: Math.max(20, rect.height + (pad * 2)),
    right: rect.right + pad,
    bottom: rect.bottom + pad,
  };
}

function scheduleTargetRectUpdate() {
  nextTick(() => {
    refreshViewport();
    updateTargetRect();
  });
}

const highlightStyle = computed(() => {
  const rect = targetRect.value;
  if (!rect) return {};
  return {
    top: `${rect.top}px`,
    left: `${rect.left}px`,
    width: `${rect.width}px`,
    height: `${rect.height}px`,
  };
});

const cardStyle = computed(() => {
  const rect = targetRect.value;
  const panelWidth = Math.min(400, Math.max(300, viewport.value.width - 24));
  const panelHeight = 190;
  const maxLeft = Math.max(12, viewport.value.width - panelWidth - 12);

  if (!rect) {
    return {
      width: `${panelWidth}px`,
      left: `${Math.max(12, Math.min(maxLeft, (viewport.value.width - panelWidth) / 2))}px`,
      bottom: '12px',
    };
  }

  let top = rect.bottom + 12;
  if (top + panelHeight > viewport.value.height - 10) {
    top = Math.max(12, rect.top - panelHeight - 12);
  }
  const left = Math.max(12, Math.min(maxLeft, rect.left));

  return {
    width: `${panelWidth}px`,
    top: `${top}px`,
    left: `${left}px`,
  };
});

function onViewportChanged() {
  if (!props.active) return;
  scheduleTargetRectUpdate();
}

watch(
  () => props.active,
  () => {
    scheduleTargetRectUpdate();
  },
  { immediate: true },
);

watch(
  () => props.step?.targetId,
  () => {
    scheduleTargetRectUpdate();
  },
);

watch(
  () => props.stepIndex,
  () => {
    scheduleTargetRectUpdate();
  },
);

onMounted(() => {
  refreshViewport();
  window.addEventListener('resize', onViewportChanged);
  window.addEventListener('scroll', onViewportChanged, true);
});

onBeforeUnmount(() => {
  window.removeEventListener('resize', onViewportChanged);
  window.removeEventListener('scroll', onViewportChanged, true);
});
</script>

<template>
  <div v-if="active" class="tutorial-overlay" aria-live="polite" role="dialog" aria-label="Playable tutorial">
    <div class="tutorial-overlay-dim" />

    <section class="tutorial-card" :style="cardStyle">
      <p class="tutorial-step-counter">Step {{ Math.max(0, stepIndex) + 1 }} / {{ Math.max(1, totalSteps) }}</p>
      <h4 class="tutorial-title">{{ step?.title || 'Tutorial' }}</h4>
      <p class="tutorial-body">{{ step?.body || '' }}</p>
      <p v-if="statusText" class="tutorial-status">{{ statusText }}</p>
      <div class="tutorial-actions">
        <button type="button" class="tutorial-btn ghost" @click="emit('skip')">Skip</button>
        <button type="button" class="tutorial-btn ghost" :disabled="!canGoBack" @click="emit('back')">Back</button>
        <button type="button" class="tutorial-btn primary" :disabled="!canAdvance" @click="emit('next')">Next</button>
      </div>
    </section>
  </div>
</template>

<style scoped>
.tutorial-overlay {
  position: fixed;
  inset: 0;
  z-index: 90;
  pointer-events: none;
}

.tutorial-overlay-dim {
  position: absolute;
  inset: 0;
  background: rgba(2, 6, 23, 0.32);
}

.tutorial-card {
  position: absolute;
  border-radius: 12px;
  border: 1px solid var(--app-accent-strong);
  background: rgba(15, 23, 42, 0.95);
  box-shadow: 0 16px 34px rgba(2, 6, 23, 0.5);
  padding: 0.75rem 0.82rem;
  display: flex;
  flex-direction: column;
  gap: 0.42rem;
  pointer-events: auto;
}

.tutorial-step-counter {
  margin: 0;
  font-size: 0.66rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--app-text-muted);
}

.tutorial-title {
  margin: 0;
  font-size: 0.94rem;
  color: var(--app-accent);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.tutorial-body {
  margin: 0;
  font-size: 0.82rem;
  color: var(--app-text);
  line-height: 1.4;
}

.tutorial-status {
  margin: 0;
  font-size: 0.73rem;
  color: #93c5fd;
}


.tutorial-actions {
  display: flex;
  gap: 0.42rem;
  justify-content: flex-end;
}

.tutorial-btn {
  pointer-events: auto;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.45);
  background: transparent;
  color: var(--app-text);
  font-size: 0.73rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 0.3rem 0.7rem;
}

.tutorial-btn.primary {
  border-color: var(--app-accent-strong);
  color: var(--app-accent);
}

.tutorial-btn:disabled {
  opacity: 0.4;
}
</style>
