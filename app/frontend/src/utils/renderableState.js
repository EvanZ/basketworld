/**
 * Return the canonical state to render on the board.
 *
 * Action results may retain a pre-action snapshot for animation origins, but
 * the board itself must always show the state after the completed transition.
 */
export function resolveRenderableState(gameState) {
  const positions = Array.isArray(gameState?.positions) ? gameState.positions : [];
  const rawBallHolder = gameState?.ball_holder;
  const ballHolder = rawBallHolder !== null
    && rawBallHolder !== undefined
    && Number.isFinite(Number(rawBallHolder))
    ? Number(rawBallHolder)
    : null;
  return { positions, ballHolder };
}
