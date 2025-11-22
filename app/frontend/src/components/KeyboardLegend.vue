<script setup>
const shortcuts = [
  { key: 'N', label: 'New Game' },
  { key: 'P', label: 'Self-Play' },
  { key: 'T', label: 'Submit Turn' },
  { key: 'S', label: 'Save Episode' },
  { key: 'Y', label: 'Replay Episode' },
  { key: 'R', label: 'Reset Stats' },
  { key: 'C', label: 'Copy Stats to Clipboard' },
  { key: '0-9', label: 'Select Player (by ID)' },
  { key: '←', label: 'Step Back (manual replay)', isArrow: true },
  { key: '→', label: 'Step Forward (manual replay)', isArrow: true },
];
</script>

<template>
  <div class="kb-legend">
    <div class="kb-legend-title">
      <font-awesome-icon :icon="['fas','keyboard']" /> Shortcuts
    </div>
    <ul class="kb-legend-list">
      <li v-for="item in shortcuts" :key="item.key" class="kb-item">
        <template v-if="item.key === '0-9'">
          <span class="keycap small" aria-hidden="true">0</span>
          <span class="keycap-sep" aria-hidden="true">–</span>
          <span class="keycap small" aria-hidden="true">9</span>
        </template>
        <template v-else-if="item.isArrow">
          <span class="keycap arrow-key" aria-hidden="true">{{ item.key }}</span>
        </template>
        <template v-else>
          <span class="keycap" aria-hidden="true">{{ item.key }}</span>
        </template>
        <span class="kb-label">{{ item.label }}</span>
      </li>
    </ul>
  </div>
  
</template>

<style scoped>
.kb-legend {
  margin-top: 12px;
  background: rgba(255,255,255,0.9);
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 8px 10px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.1);
  font-size: 12px;
  color: #333;
}
.kb-legend-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 600;
  margin-bottom: 6px;
}
.kb-legend-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-wrap: wrap;
  gap: 4px 12px;
  align-items: center;
}
.kb-item { display: flex; align-items: center; gap: 6px; }
.kb-label { white-space: nowrap; }

/* Keycap styling */
.keycap {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 22px;
  height: 22px;
  padding: 0 6px;
  border: 1px solid #bbb;
  border-bottom-width: 3px;
  border-radius: 4px;
  background: linear-gradient(#f7f7f7, #eaeaea);
  box-shadow: inset 0 1px 0 #fff, 0 1px 0 rgba(0,0,0,0.06);
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
  font-weight: 700;
}
.keycap.small { min-width: 18px; height: 20px; padding: 0 4px; font-size: 11px; }
.keycap.arrow-key { font-size: 14px; min-width: 24px; }
.keycap-sep { margin: 0 4px; color: #666; }
</style>


