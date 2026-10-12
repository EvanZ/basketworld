import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const source = await readFile(
  new URL('../src/utils/skillRatingsWorkbench.js', import.meta.url),
  'utf8',
);
const {
  latestReportedProgress,
  reportedProgressPercent,
  selectedAnalysisIdentity,
  skillRatingsReportOptionLabel,
} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('progress advances only from evaluator-reported completed counts', () => {
  const events = [
    { phase: 'batch_completed', completed: 64, total: 128, batch: 1 },
    { phase: 'batch_running', elapsed_sec: 20 },
  ];

  const progress = latestReportedProgress(events);

  assert.equal(progress.completed, 64);
  assert.equal(progress.total, 128);
  assert.equal(reportedProgressPercent(progress), 50);
  assert.equal(latestReportedProgress([{ phase: 'batch_running', elapsed_sec: 10 }]), null);
});

test('report selection names the exact analysis revision and path', () => {
  const item = {
    study_id: 'study-a',
    checkpoint_update: 42,
    analysis_label: 'Reanalysis v3',
    path: '/repo/artifacts/fixed_pool_reports/study-a/analysis_revisions/v3.json',
  };

  assert.equal(
    skillRatingsReportOptionLabel(item),
    'study-a · update 42 · Reanalysis v3 · /repo/artifacts/fixed_pool_reports/study-a/analysis_revisions/v3.json',
  );
});

test('selected analysis retains every raw-study dependency for inspection', () => {
  const identity = selectedAnalysisIdentity({
    summary: { analysis_label: 'Original report schema v4', path: '/reports/checkpoint_report.json' },
    raw_artifact_dependencies: {
      study_id: 'study-a',
      pool_manifest: '/raw/pool_manifest.json',
      schedule_manifest: '/raw/schedule_manifest.json',
      mlflow_artifact_path: null,
    },
  });

  assert.equal(identity.label, 'Original report schema v4');
  assert.equal(identity.path, '/reports/checkpoint_report.json');
  assert.deepEqual(identity.dependencies, [
    { key: 'study_id', value: 'study-a' },
    { key: 'pool_manifest', value: '/raw/pool_manifest.json' },
    { key: 'schedule_manifest', value: '/raw/schedule_manifest.json' },
  ]);
});
