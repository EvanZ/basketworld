import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(
  new URL('../src/utils/modelBackend.js', import.meta.url),
  'utf8',
);
const { isJaxModelState } = await import(
  `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`,
);

test('recognizes the explicit JAX backend', () => {
  assert.equal(isJaxModelState({ model_backend: 'jax' }), true);
  assert.equal(isJaxModelState({ model_backend: 'JAX-native' }), true);
});

test('recognizes a loaded JAX checkpoint when a compact update omits its backend', () => {
  assert.equal(isJaxModelState({
    model_metadata: {
      checkpoint_path: '/runs/latest/checkpoints/latest',
      policy_spec: { model_type: 'attention', action_dim_per_player: 9 },
    },
  }), true);
});

test('does not expose JAX-only controls for unrelated model metadata', () => {
  assert.equal(isJaxModelState({ model_backend: 'sb3' }), false);
  assert.equal(isJaxModelState({
    model_metadata: { checkpoint_path: '/runs/legacy/model.zip' },
  }), false);
});
