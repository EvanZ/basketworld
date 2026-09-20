import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import { fileURLToPath } from 'node:url';

// npm run dev -- --config tests/capture-vite.config.js --port 5174
// Open /tests/capture.html; uses a read-only recorded replay, without resetting
// or stepping the user's game. No test fixtures or routes enter production.
export default defineConfig({
  plugins: [
    {
      name: 'seed-recorded-episode-for-save-test',
      enforce: 'pre',
      transform(code, id) {
        if (process.env.BW_CAPTURE_SEED_REPLAY !== '1' || !id.endsWith('/src/App.vue')) return;
        // Test-only bootstrap: restore an existing replay, then exercise the
        // unmodified App Save Episode button and all its interpolation frames.
        // This never invokes init_game, reset, step, or start_self_play.
        return code.replace('</script>', `
onMounted(async () => {
  try {
    const response = await fetch('/api/replay_last_episode', { method: 'POST' });
    const payload = await response.json();
    if (!payload.states?.length) throw new Error('A recorded episode is required');
    initialSetup.value = {};
    replayStates.value = payload.states;
    gameHistory.value = payload.states.slice();
    currentStepIndex.value = payload.states.length - 1;
    gameState.value = payload.states.at(-1);
    isManualStepping.value = true;
    canReplay.value = true;
    document.querySelector('#test-export-result').textContent = 'Ready: ' + payload.states.length + ' recorded states';
  } catch (error) {
    document.querySelector('#test-export-result').textContent = 'FAIL: ' + error.message;
  }
});
</script>`);
      },
    },
    vue(),
  ],
  resolve: { alias: { '@': fileURLToPath(new URL('../src', import.meta.url)) } },
  server: { proxy: {
    '/api/render_gif_from_pngs': process.env.BW_CAPTURE_MEDIA_URL || 'http://localhost:8080',
    '/api/episode_exports': process.env.BW_CAPTURE_MEDIA_URL || 'http://localhost:8080',
    '/test/': process.env.BW_CAPTURE_MEDIA_URL || 'http://localhost:8080',
    '/api': 'http://localhost:8080',
  } },
});
