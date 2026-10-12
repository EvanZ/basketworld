import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const source = await readFile(new URL('../src/utils/playerIdentity.js', import.meta.url), 'utf8');
const {
  PLAYER_SURNAME_POOL,
  playerSurnameForId,
} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('dev player identities use the production surname pool deterministically', () => {
  assert.equal(playerSurnameForId(0), PLAYER_SURNAME_POOL[0].toUpperCase());
  assert.equal(playerSurnameForId(9), PLAYER_SURNAME_POOL[9].toUpperCase());
  assert.equal(
    playerSurnameForId(PLAYER_SURNAME_POOL.length),
    PLAYER_SURNAME_POOL[0].toUpperCase(),
  );
  assert.equal(playerSurnameForId('not-a-player'), '');
});

test('dev jerseys and environment rows render names and numbers without live EV labels', async () => {
  const boardSource = await readFile(
    new URL('../src/components/GameBoard.vue', import.meta.url),
    'utf8',
  );
  const controlsSource = await readFile(
    new URL('../src/components/PlayerControls.vue', import.meta.url),
    'utf8',
  );
  const appSource = await readFile(new URL('../src/App.vue', import.meta.url), 'utf8');

  assert.match(boardSource, /v-if="getPlayerDisplayName\(player\.id\)"/);
  assert.doesNotMatch(boardSource, /currentGameState\.ep_by_player\[player\.id\]/);
  assert.match(boardSource, /\.ball-indicator\s*\{[^}]*stroke:\s*#facc15;/s);
  assert.match(boardSource, /\.ball-indicator\s*\{[^}]*stroke-width:\s*0\.09rem;/s);
  assert.match(boardSource, /animation:\s*ball-handler-glow-pulse var\(--dribble-period, 0\.5s\)/);
  assert.match(boardSource, /animation-delay:\s*var\(--dribble-delay, 0s\)/);
  assert.doesNotMatch(boardSource, /\.ball-indicator\s*\{[^}]*stroke-dasharray:/s);
  assert.match(controlsSource, /formatPlayerIdentity\(row\.playerId\)/);
  assert.match(appSource, /:player-display-names="playerDisplayNames"/);
  assert.match(appSource, /:player-jersey-numbers="playerJerseyNumbers"/);
});
