function normalizedFrameCount(frameCount) {
  const numeric = Math.floor(Number(frameCount));
  return Number.isFinite(numeric) ? Math.max(0, numeric) : 0;
}

function clampFrameNumber(value, frameCount, fallback) {
  const numeric = Math.round(Number(value));
  const resolved = Number.isFinite(numeric) ? numeric : fallback;
  return Math.min(frameCount, Math.max(1, resolved));
}

export function fullEpisodeFrameRange(frameCount) {
  const count = normalizedFrameCount(frameCount);
  if (count === 0) {
    return { start: 0, end: 0, frameCount: 0, selectedFrameCount: 0 };
  }
  return { start: 1, end: count, frameCount: count, selectedFrameCount: count };
}

export function normalizeEpisodeFrameRange(start, end, frameCount) {
  const count = normalizedFrameCount(frameCount);
  if (count === 0) return fullEpisodeFrameRange(0);

  const normalizedStart = clampFrameNumber(start, count, 1);
  const normalizedEnd = clampFrameNumber(end, count, count);
  const low = Math.min(normalizedStart, normalizedEnd);
  const high = Math.max(normalizedStart, normalizedEnd);
  return {
    start: low,
    end: high,
    frameCount: count,
    selectedFrameCount: high - low + 1,
  };
}

export function updateEpisodeFrameBoundary(range, boundary, value, frameCount) {
  const current = normalizeEpisodeFrameRange(
    range?.start,
    range?.end,
    frameCount,
  );
  if (current.frameCount === 0) return current;

  const nextValue = clampFrameNumber(
    value,
    current.frameCount,
    boundary === 'start' ? current.start : current.end,
  );
  const start = boundary === 'start'
    ? Math.min(nextValue, current.end)
    : current.start;
  const end = boundary === 'end'
    ? Math.max(nextValue, current.start)
    : current.end;
  return {
    start,
    end,
    frameCount: current.frameCount,
    selectedFrameCount: end - start + 1,
  };
}

export function buildEpisodeFrameSelection(states, start, end) {
  const episodeStates = Array.isArray(states) ? states : [];
  const range = normalizeEpisodeFrameRange(start, end, episodeStates.length);
  if (range.frameCount === 0) return { range, entries: [] };

  const entries = [];
  for (let index = range.start - 1; index < range.end; index += 1) {
    entries.push({
      index,
      frameNumber: index + 1,
      state: episodeStates[index],
      previousState: index > 0 ? episodeStates[index - 1] : episodeStates[index],
    });
  }
  return { range, entries };
}
