export function stableBoardViewBox({
  courtLayout,
  minimalChrome,
  hexRadius,
  multiPossession,
}) {
  if (!Array.isArray(courtLayout) || courtLayout.length === 0) {
    return '-100 -100 200 200';
  }

  const courtX = courtLayout.map((hex) => Number(hex.x));
  const courtY = courtLayout.map((hex) => Number(hex.y));
  const allX = [...courtX];
  const allY = [...courtY];

  if (multiPossession) {
    const minCourtX = Math.min(...courtX);
    const minCourtY = Math.min(...courtY);
    const maxCourtY = Math.max(...courtY);

    // Reserve every restart area on every frame. Baseline inbounds sit one
    // axial west step beyond the court, while technical inbounds sit one
    // offset row beyond either sideline. Keeping both sideline rows in the
    // bounds prevents the SVG from recentering when the selected side changes.
    allX.push(minCourtX - (Math.sqrt(3) * hexRadius));
    allY.push(
      minCourtY - (1.5 * hexRadius),
      maxCourtY + (1.5 * hexRadius),
    );
  }

  const margin = minimalChrome ? hexRadius * 1.8 : hexRadius * 3;
  const minX = Math.min(...allX) - margin;
  const maxX = Math.max(...allX) + margin;
  const minY = Math.min(...allY) - margin;
  const maxY = Math.max(...allY) + margin;

  return `${minX} ${minY} ${maxX - minX} ${maxY - minY}`;
}
