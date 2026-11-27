# League authoring

The trainer's **Leagues** section maintains reusable player pools independently
of training runs. Create multiple leagues with 64 teams by default, select a
3v3/4v4/5v5 format and skill rules, and configure generation distributions and a
seed. Players are generated once, then randomly assigned to complete rosters.

Navigate **Leagues → Rosters → Players** to inspect all supported skills, rename
teams/players, edit skills and jersey numbers, or reassign a player. To reassign,
open the player's details, choose **Move player…**, select a destination, and
click **Confirm move**. Selecting a destination alone does not change the roster;
**Cancel** discards the proposed move. Confirmed moves and other edits
automatically save to the trainer database. Wait for **All changes saved** before
refreshing. Invalid edits remain unsaved until corrected. Incomplete rosters and
unassigned players are allowed in drafts, but publication requires every roster
to be complete and every player assigned once, with unique jerseys per team.

**Publish version** creates an immutable snapshot. Later edits affect the draft
only; earlier snapshots remain unchanged. Inspect published versions read-only.
Archive/restore controls hide items without deleting their historical artifacts.
To remove a test league permanently, choose **Delete league…**, type its exact
name, and confirm. This deletes its draft, players, rosters, and published
versions. Leagues referenced by training runs or saved run snapshots cannot be
deleted; archive them instead.

**League-backed training is not enabled in this slice.** The new-run form can
select an exact published version, but Launch remains disabled while a league
is selected. The API enforces the same restriction. Clear the selection to use
current random skill sampling. Training integration follows in #85 slice 2;
season/tournament records follow in slices 3 and 5. This feature adds no training
metrics or per-game training records.

Drafts and metadata use the configured trainer database. Published JSON files
use `<trainer data directory>/leagues/<league-id>/v<version>.json`. Future runs
pin `{league_id, version, sha256}` and an exact `runs/<run-id>/league.json` copy.
Copies retain that reference; resume/extension load the pinned snapshot without
regenerating. The shared `basketworld_jax.leagues.load_league_snapshot` validates
identities, membership, skills, schema, checksum, team size and shot-defense
rules for future consumers. See [the trainer app guide](https://github.com/EvanZ/basketworld/blob/codex/halfcourt-multi-possession/docs/training_app.md#leagues-rosters-and-players-86)
for the full storage/API contract.
