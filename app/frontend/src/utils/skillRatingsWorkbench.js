export function latestReportedProgress(events = []) {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (
      Number.isFinite(event?.completed)
      && Number.isFinite(event?.total)
      && event.total > 0
    ) {
      return event;
    }
  }
  return null;
}

export function reportedProgressPercent(progress) {
  if (!progress) return null;
  return Math.min(100, Math.max(0, (100 * progress.completed) / progress.total));
}

export function skillRatingsReportOptionLabel(item) {
  const study = item?.study_id || 'Study';
  const update = item?.checkpoint_update ?? '—';
  const analysis = item?.analysis_label || 'Original report';
  return `${study} · update ${update} · ${analysis} · ${item?.path || 'unknown path'}`;
}

export function selectedAnalysisIdentity(report) {
  const summary = report?.summary || {};
  const dependencies = Object.entries(report?.raw_artifact_dependencies || {})
    .filter(([, value]) => value !== null && value !== undefined && value !== '')
    .map(([key, value]) => ({ key, value: String(value) }));
  return {
    label: summary.analysis_label || 'Original report',
    path: summary.path || '',
    dependencies,
  };
}

export function skillEffectInterval(skill) {
  const low = skill?.interval_p05;
  const high = skill?.interval_p95;
  if (low == null || high == null || low === '' || high === ''
      || !Number.isFinite(Number(low)) || !Number.isFinite(Number(high))
      || Number(low) > Number(high)) return null;
  return {
    label: `${Number(low).toFixed(2)} to ${Number(high).toFixed(2)}`,
    includesZero: Number(low) <= 0 && Number(high) >= 0,
  };
}
