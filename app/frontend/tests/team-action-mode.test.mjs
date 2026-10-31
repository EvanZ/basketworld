import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(
  new URL('../src/utils/teamActionMode.js', import.meta.url),
  'utf8',
);
const {
  policyTeamActionMode,
  usesBackendJointPolicyActions,
} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('autoregressive checkpoint metadata selects backend joint actions', () => {
  const state = {
    model_metadata: {
      policy_spec: {
        team_action_mode: 'autoregressive',
      },
    },
  };

  assert.equal(policyTeamActionMode(state), 'autoregressive');
  assert.equal(usesBackendJointPolicyActions(state), true);
});

test('independent is the legacy default and saved config is a fallback', () => {
  assert.equal(policyTeamActionMode({}), 'independent');
  assert.equal(usesBackendJointPolicyActions({}), false);
  assert.equal(
    policyTeamActionMode({ training_params: { team_action_mode: 'autoregressive' } }),
    'autoregressive',
  );
});

test('explicit state mode takes precedence over metadata fallbacks', () => {
  const state = {
    team_action_mode: 'independent',
    model_metadata: {
      policy_spec: {
        team_action_mode: 'autoregressive',
      },
    },
  };

  assert.equal(policyTeamActionMode(state), 'independent');
  assert.equal(usesBackendJointPolicyActions(state), false);
});
