import { createApp, ref, h, nextTick, onMounted } from 'vue';
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome';
import { library } from '@fortawesome/fontawesome-svg-core';
import { faToggleOff, faToggleOn } from '@fortawesome/free-solid-svg-icons';
import GameBoard from '../src/components/GameBoard.vue';
import '../src/assets/main.css';

library.add(faToggleOff, faToggleOn);

function syntheticBaseState() {
  return {
    players_per_side: 3,
    positions: [[5, 7], [3, 6], [7, 6], [5, 3], [3, 4], [7, 4]],
    ball_holder: 0,
    shot_clock: 18,
    shot_clock_steps: 24,
    min_shot_clock: 10,
    done: false,
    enable_multi_possession: true,
    episode_end_mode: 'possessions',
    multi_possession_limit: 5,
    completed_possessions: 3,
    user_completed_possessions: 2,
    ai_completed_possessions: 1,
    user_score: 12,
    ai_score: 10,
    team_a_score: 12,
    team_b_score: 10,
    team_a_ids: [0, 1, 2],
    team_b_ids: [3, 4, 5],
    user_player_ids: [0, 1, 2],
    ai_player_ids: [3, 4, 5],
    offense_ids: [0, 1, 2],
    defense_ids: [3, 4, 5],
    offense_label: 'user',
    offense_team: 'team_a',
    team_ownership: { team_a: 'user', team_b: 'ai' },
    game_phase: 'live',
    in_overtime: false,
    overtime_round: 0,
    overtime_possessions_completed: 0,
    inbound_steps_remaining: 0,
    inbound_deadline_steps: 5,
    check_setup_steps_remaining: 0,
    check_setup_steps: 5,
    check_steps_remaining: 0,
    check_deadline_steps: 5,
    clearance_required: false,
    basket_position: [5, 1],
    court_width: 11,
    court_height: 11,
    three_point_distance: 4,
    three_point_short_distance: 3,
    three_point_hexes: [],
    offensive_lane_hexes: [],
    defensive_lane_hexes: [],
    offensive_lane_steps: { 0: 0, 1: 0, 2: 0 },
    defensive_lane_steps: { 3: 0, 4: 0, 5: 0 },
    three_second_max_steps: 3,
    action_mask: Array.from({ length: 6 }, () => Array(18).fill(1)),
    last_action_results: {},
    player_shooting_skills: {},
    player_episode_skills: {},
    player_rebound_skills: {},
    ep_by_player: Array(6).fill(0),
  };
}

const app = createApp({
  setup() {
    const board = ref(null);
    const history = ref([]);
    const step = ref(0);
    const progress = ref(1);
    const status = ref('Loading recorded episode…');
    const png = ref('');
    const gif = ref('');
    const busy = ref(false);
    const compact = ref(false);
    const captureMode = ref(true);
    const showEndGameOutcome = ref(true);
    let states = [];
    const overlay = ref(null);
    const dimensions = ref([]);
    const cases = ref('');
    const variant = ref('recorded');

    async function select(index) {
      variant.value = 'recorded';
      step.value = index;
      progress.value = 1;
      showEndGameOutcome.value = true;
      history.value = states.slice(0, index + 1);
      const results = states[index]?.last_action_results;
      const rebound = results?.rebound || results?.rebounds?.[0];
      overlay.value = rebound?.attempt ? {
        source: 'live_rebound_step',
        sampled_winner: { player_id: rebound.winner, team: rebound.winner_team },
        sampled_target: { q: rebound.target[0], r: rebound.target[1] },
        target_cells: rebound.target_cells,
      } : null;
      await nextTick();
      // Allow CSS transitions on the live scoreboard to settle before comparing.
      await new Promise((resolve) => setTimeout(resolve, 200));
    }

    async function selectVariant(value) {
      busy.value = true;
      png.value = '';
      gif.value = '';
      await select(step.value);
      variant.value = value;
      if (value === 'recorded') { busy.value = false; return; }
      const state = structuredClone(states[step.value]);
      state.done = false;
      state.last_action_results = {};
      state.clearance_required = false;
      state.game_phase = ['inbound', 'legacy'].includes(value) ? 'awaiting_inbound' : 'live';
      state.inbound_steps_remaining = 3;
      state.inbound_deadline_steps = 5;
      state.shot_clock = 24;
      state.user_score = 12;
      state.ai_score = 23;
      state.offensive_lane_steps = Object.fromEntries(state.offense_ids.map((id) => [id, 2]));
      state.defensive_lane_steps = Object.fromEntries(state.defense_ids.map((id) => [id, 3]));
      state.enable_multi_possession = value !== 'legacy';
      if (value === 'legacy') delete state.episode_end_mode;
      if (value.startsWith('clock-')) {
        state.enable_multi_possession = true;
        state.episode_end_mode = 'game_clock';
        state.game_clock_seconds = 300;
        state.overtime_seconds = 120;
        state.game_clock_remaining = value === 'clock-overtime' ? 83 : 247;
        state.overtime_round = value === 'clock-overtime' ? 2 : 0;
        state.game_phase = value === 'clock-setup'
          ? 'check_setup'
          : (value === 'clock-inbound' ? 'awaiting_inbound' : 'live');
        state.check_setup_steps_remaining = value === 'clock-setup' ? 3 : 0;
        state.inbound_steps_remaining = value === 'clock-inbound' ? 4 : 0;
      }
      const shooter = state.offense_ids[0];
      const receiver = state.offense_ids[1];
      const defender = state.defense_ids[0];
      const preActionPositions = structuredClone(state.positions);
      if (value === 'clearance') state.clearance_required = true;
      if (value === 'made') state.last_action_results = { shots: { [shooter]: { success: true, is_three: false } } };
      if (value === 'pass') {
        progress.value = 0.5;
        state.ball_holder = receiver;
        state.last_action_results = {
          passes: { [shooter]: { success: true, target: receiver } },
          pre_action_positions: preActionPositions,
        };
      }
      if (value === 'rebound') {
        progress.value = 0.75;
        state.clearance_required = true;
        state.ball_holder = defender;
        state.last_action_results = {
          shots: { [shooter]: { success: false, is_three: true } },
          rebound: {
            attempt: true,
            defensive: true,
            winner: defender,
            winner_team: 'defense',
            target: preActionPositions[defender],
          },
          pre_action_positions: preActionPositions,
        };
        overlay.value = {
          source: 'live_rebound_step',
          sampled_winner: { player_id: defender, team: 'defense' },
          sampled_target: {
            q: preActionPositions[defender][0],
            r: preActionPositions[defender][1],
          },
          target_cells: [],
        };
      }
      if (value === 'steal') {
        progress.value = 0.72;
        state.ball_holder = defender;
        state.last_action_results = {
          passes: {
            [shooter]: {
              success: false,
              target: receiver,
              turnover: true,
              reason: 'steal',
              stolen_by: defender,
            },
          },
          turnovers: [{ reason: 'steal', player_id: shooter, stolen_by: defender }],
          pre_action_positions: preActionPositions,
        };
      }
      if (value === 'turnover') {
        state.last_action_results = { turnovers: [{ reason: 'defender_pressure' }] };
      }
      if (value === 'end') state.done = true;
      if (['terminal-shot-action', 'terminal-shot-title'].includes(value)) {
        state.done = true;
        state.last_action_results = { shots: { [shooter]: { success: false, is_three: true } } };
        // This is the middle of the flight during GIF capture, where the ball
        // and projectile should be visibly distinct. The title fixture then
        // verifies the separate final held frame.
        progress.value = value === 'terminal-shot-action' ? 0.5 : 1;
        showEndGameOutcome.value = value === 'terminal-shot-title';
      }
      if (value !== 'rebound') overlay.value = null;
      history.value = [state];
      await nextTick();
      await new Promise((resolve) => setTimeout(resolve, 200));
      busy.value = false;
    }

    function boardGeometry() {
      const root = document.querySelector('.game-board-container');
      const boardRect = root.getBoundingClientRect();
      const court = root.querySelector(':scope > svg').getBoundingClientRect();
      const scoreboard = root.querySelector('.game-scoreboard').getBoundingClientRect();
      if (scoreboard.left < boardRect.left || scoreboard.right > boardRect.right) {
        throw new Error('Scoreboard overflows the board');
      }
      return [boardRect.width, boardRect.height, court.top - boardRect.top, court.width, court.height];
    }

    async function checkStableLayout() {
      busy.value = true;
      const originalCompact = compact.value;
      const originalCaptureMode = captureMode.value;
      let checked = 0;
      try {
        await document.fonts.ready;
        const scoreboardWidths = [];
        for (const exportMode of [true, false]) {
          captureMode.value = exportMode;
          for (const narrow of [false, true]) {
            compact.value = narrow;
            await select(0);
            const baseline = boardGeometry();
            if (exportMode) scoreboardWidths.push(document.querySelector('.game-scoreboard').getBoundingClientRect().width);
            const check = (label) => {
              const current = boardGeometry();
              if (current.some((value, index) => Math.abs(value - baseline[index]) > 0.1)) {
                throw new Error(`${label}: geometry changed from ${baseline} to ${current}`);
              }
              const banner = document.querySelector('.shot-attempt-banner');
              if (exportMode && banner && getComputedStyle(banner).opacity !== '1') throw new Error('Export banner is faded');
              checked += 1;
            };
            for (let index = 0; index < states.length; index += 1) {
              await select(index);
              check(`Step ${index}, width ${narrow ? 440 : 640}`);
            }
            for (const fixture of [
              'clearance', 'made', 'pass', 'rebound', 'steal', 'turnover', 'inbound',
              'clock-live', 'clock-setup', 'clock-inbound', 'clock-overtime',
              'terminal-shot-action', 'terminal-shot-title', 'end',
            ]) {
              await selectVariant(fixture);
              busy.value = true;
              check(`Fixture ${fixture}, width ${narrow ? 440 : 640}`);
            }
          }
        }
        if (scoreboardWidths[0] < scoreboardWidths[1] * 1.15) throw new Error('Scoreboard does not scale with board width');
        status.value = `PASS: ${checked} live/export layout checks; court stays fixed at each width; scoreboard scales (${scoreboardWidths.map(Math.round).join(' / ')}px); export banners fully visible`;
      } catch (error) { status.value = `FAIL: ${error.message}`; }
      finally {
        compact.value = originalCompact;
        captureMode.value = originalCaptureMode;
        await select(0);
        busy.value = false;
      }
    }

    async function capture() {
      const frame = await board.value.renderStateToPng();
      if (!frame?.startsWith('data:image/png')) throw new Error(`No PNG at step ${step.value}`);
      const img = new Image();
      img.src = frame;
      await img.decode();
      dimensions.value.push([img.width, img.height]);
      return frame;
    }

    async function captureSelected() {
      busy.value = true;
      try {
        png.value = await capture();
        const response = await fetch('/api/render_gif_from_pngs', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ frames: [png.value], durations: [1] }),
        });
        if (!response.ok) throw new Error(await response.text());
        const blob = await response.blob();
        const reader = new FileReader();
        gif.value = await new Promise((resolve, reject) => {
          reader.onload = () => resolve(reader.result);
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        });
        await nextTick();
        await document.querySelector('#gif-preview').decode();
        status.value = `PASS: PNG and decoded GIF, step ${step.value}; ${dimensions.value.at(-1).join('×')}`;
      } catch (error) { status.value = `FAIL: ${error.message}`; }
      finally { busy.value = false; }
    }

    async function exportEpisode() {
      busy.value = true;
      try {
        const frames = [];
        dimensions.value = [];
        for (let index = 0; index < states.length; index += 1) {
          await select(index);
          frames.push(await capture());
          status.value = `Captured ${index + 1}/${states.length} recorded states`;
        }
        // Exercise the same encoder as Save Episode with all recorded states.
        const response = await fetch('/api/render_gif_from_pngs', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ frames, durations: frames.map((_, i) => i === frames.length - 1 ? 1 : 0.2) }),
        });
        if (!response.ok) throw new Error(await response.text());
        const blob = await response.blob();
        const reader = new FileReader();
        gif.value = await new Promise((resolve) => {
          reader.onload = () => resolve(reader.result);
          reader.readAsDataURL(blob);
        });
        await nextTick();
        const decoded = document.querySelector('#gif-preview');
        await decoded.decode();
        const expectedWidth = Math.max(...dimensions.value.map(([width]) => width));
        const expectedHeight = Math.max(...dimensions.value.map(([, height]) => height));
        if (decoded.naturalWidth !== expectedWidth || decoded.naturalHeight !== expectedHeight) {
          throw new Error(`GIF canvas ${decoded.naturalWidth}×${decoded.naturalHeight} crops frames; expected ${expectedWidth}×${expectedHeight}`);
        }
        png.value = frames.at(-1);
        status.value = `PASS: ${frames.length} recorded states encoded as ${Math.round(blob.size / 1024)} KiB GIF; dimensions ${JSON.stringify([...new Set(dimensions.value.map(String))])}`;
      } catch (error) { status.value = `FAIL: ${error.message}`; }
      finally { busy.value = false; }
    }

    onMounted(async () => {
      try {
        const response = await fetch('/api/replay_last_episode', { method: 'POST' });
        const payload = await response.json();
        states = payload.states || [];
        if (!states.length) throw new Error('A recorded episode is required');
        cases.value = states.map((state, index) => `${index}: ${state.game_phase}, score ${state.user_score}-${state.ai_score}, ${state.clearance_required ? 'clearance' : ''} ${Object.keys(state.last_action_results?.shots || {}).length ? 'shot' : ''} ${state.done ? 'END' : ''}`).join(' | ');
        await select(0);
        status.value = `Ready: ${states.length} recorded states`;
      } catch (error) {
        states = [syntheticBaseState()];
        cases.value = 'Synthetic display fixture (backend replay unavailable)';
        await select(0);
        status.value = `Ready: synthetic fixture; ${error.message}`;
      }
    });

    return () => h('div', [
      h('p', { id: 'capture-status' }, status.value),
      h('p', { id: 'capture-cases' }, cases.value),
      h('label', ['Step ', h('select', { value: step.value, disabled: busy.value, onChange: (event) => select(Number(event.target.value)) }, states.map((_, index) => h('option', { value: index }, String(index))))]),
      h('label', ['Fixture ', h('select', { value: variant.value, disabled: busy.value, onChange: (event) => selectVariant(event.target.value) }, [
        h('option', { value: 'recorded' }, 'Recorded state'),
        h('option', { value: 'inbound' }, 'Inbound / double-digit scores / lane lights'),
        h('option', { value: 'clearance' }, 'Clearance only'),
        h('option', { value: 'made' }, 'Made 2pt'),
        h('option', { value: 'pass' }, 'Completed pass hands'),
        h('option', { value: 'rebound' }, 'Missed 3pt / rebound / clearance'),
        h('option', { value: 'steal' }, 'Steal'),
        h('option', { value: 'turnover' }, 'Turnover'),
        h('option', { value: 'terminal-shot-action' }, 'Terminal missed 3pt action'),
        h('option', { value: 'terminal-shot-title' }, 'Terminal End Game hold'),
        h('option', { value: 'end' }, 'End Game'),
        h('option', { value: 'legacy' }, 'Legacy shot clock'),
        h('option', { value: 'clock-live' }, 'Game clock / regulation / running'),
        h('option', { value: 'clock-setup' }, 'Game clock / check setup / stopped'),
        h('option', { value: 'clock-inbound' }, 'Game clock / inbound / stopped'),
        h('option', { value: 'clock-overtime' }, 'Game clock / overtime'),
      ])]),
      h('button', { onClick: captureSelected, disabled: busy.value }, 'Capture PNG and GIF'),
      h('button', { onClick: exportEpisode, disabled: busy.value }, 'Export full recorded episode'),
      h('button', { onClick: checkStableLayout, disabled: busy.value }, 'Check stable banner layout'),
      h('button', { onClick: () => { compact.value = !compact.value; }, disabled: busy.value }, 'Toggle compact width'),
      h('button', {
        onClick: () => { captureMode.value = !captureMode.value; },
        disabled: busy.value,
      }, captureMode.value ? 'Use live transitions' : 'Use deterministic capture'),
      h('div', { style: 'display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap' }, [
        h('div', { style: `width:${compact.value ? 440 : 640}px;flex:none` }, [
          h('p', 'Live board'),
          history.value.length ? h(GameBoard, { key: variant.value, ref: board, gameHistory: history.value, disableBackendValueFetches: true, disableTransitions: captureMode.value, moveProgress: progress.value, reboundProgress: progress.value, showEndGameOutcome: showEndGameOutcome.value, reboundTargetOverlay: overlay.value }) : null,
        ]),
        h('div', { style: `width:${compact.value ? 440 : 640}px;flex:none` }, [
          h('p', 'Saved GIF (decoded by browser)'),
          gif.value ? h('img', { id: 'gif-preview', src: gif.value, style: 'width:100%' }) : null,
        ]),
      ]),
      png.value ? h('img', { id: 'png-preview', src: png.value, style: 'display:none' }) : null,
    ]);
  },
});
app.component('font-awesome-icon', FontAwesomeIcon);
app.mount('#app');
