export const playableTutorialSteps = [
  {
    id: 'start-game',
    targetId: 'setup-start-game-btn',
    title: 'Start A Game',
    body: 'Set players, difficulty, and format, then click Start Game.',
    completeWhen: 'has_game',
  },
  {
    id: 'open-rules-physics',
    targetId: 'setup-rules-physics-btn',
    title: 'Open Rules & Physics',
    body: 'Click Rules & Physics to review gameplay rules, lane violations, and model details.',
    completeWhen: 'rules_modal_open',
  },
  {
    id: 'select-player-zero',
    targetId: 'controls-player-zero-btn',
    title: 'Select Player 0',
    body: 'Pick Player 0 from the tabs (or press 0).',
    completeWhen: 'active_player_zero',
  },
  {
    id: 'select-actions',
    targetId: 'controls-action-pad',
    title: 'Choose Team Actions',
    body: 'Pick actions for your players. Selection auto-advances through remaining IDs.',
    completeWhen: 'all_actions_selected',
  },
  {
    id: 'submit-turn',
    targetId: 'controls-submit-turn',
    title: 'Submit The Turn',
    body: 'Click Submit Turn (or press T) to execute your team actions.',
    completeWhen: 'turn_submitted',
  },
  {
    id: 'read-scoreboard',
    targetId: 'playable-scoreboard',
    title: 'Read The Scoreboard',
    body: 'Track score, period clock, lane lights, and live TOV pressure around the ball handler.',
    completeWhen: 'manual',
  },
];
