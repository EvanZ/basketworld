export function latestLaunchSourceProvenance(run) {
  const launches = run?.provenance?.launches
  if (!Array.isArray(launches) || !launches.length) return null
  const source = launches.at(-1)?.source_provenance
  return source && typeof source === 'object' ? source : null
}

export function launchGitCommit(run) {
  const commit = latestLaunchSourceProvenance(run)?.commit_sha
  return typeof commit === 'string' && commit ? commit : 'Unavailable'
}

export function launchWorktreeState(run) {
  const source = latestLaunchSourceProvenance(run)
  if (!source || source.available === false) return 'Unavailable'
  return source.worktree_dirty ? 'Dirty tracked changes' : 'Clean'
}
