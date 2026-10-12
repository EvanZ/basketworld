import test from 'node:test'
import assert from 'node:assert/strict'

import {
  latestLaunchSourceProvenance,
  launchGitCommit,
  launchWorktreeState,
} from '../src/runProvenance.js'

test('shows the source commit from the most recent launch', () => {
  const run = {
    provenance: {
      launches: [
        { source_provenance: { available: true, commit_sha: 'a'.repeat(40), worktree_dirty: false } },
        { source_provenance: { available: true, commit_sha: 'b'.repeat(40), worktree_dirty: true } },
      ],
    },
  }

  assert.equal(latestLaunchSourceProvenance(run).commit_sha, 'b'.repeat(40))
  assert.equal(launchGitCommit(run), 'b'.repeat(40))
  assert.equal(launchWorktreeState(run), 'Dirty tracked changes')
})

test('handles clean and legacy run provenance', () => {
  assert.equal(launchWorktreeState({
    provenance: {
      launches: [{ source_provenance: { available: true, worktree_dirty: false } }],
    },
  }), 'Clean')
  assert.equal(launchGitCommit({ provenance: {} }), 'Unavailable')
  assert.equal(launchWorktreeState({ provenance: {} }), 'Unavailable')
})
