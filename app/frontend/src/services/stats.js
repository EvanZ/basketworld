// Simple localStorage-backed stats store for episode aggregates

const STORAGE_KEY = 'bw_stats_v1';

export function getDefaultStats() {
  return {
    episodes: 0,
    dunk: { attempts: 0, made: 0, assists: 0 },
    twoPt: { attempts: 0, made: 0, assists: 0 },
    threePt: { attempts: 0, made: 0, assists: 0 },
    turnovers: 0,
    points: 0,
    rewardSum: 0,
    episodeStepsSum: 0,
  };
}

export function loadStats() {
  try {
    const raw = typeof localStorage !== 'undefined' ? localStorage.getItem(STORAGE_KEY) : null;
    if (!raw) return getDefaultStats();
    const parsed = JSON.parse(raw);
    // Basic shape validation with fallbacks
    return {
      episodes: Number(parsed.episodes) || 0,
      dunk: {
        attempts: Number(parsed?.dunk?.attempts) || 0,
        made: Number(parsed?.dunk?.made) || 0,
        assists: Number(parsed?.dunk?.assists) || 0,
      },
      twoPt: {
        attempts: Number(parsed?.twoPt?.attempts) || 0,
        made: Number(parsed?.twoPt?.made) || 0,
        assists: Number(parsed?.twoPt?.assists) || 0,
      },
      threePt: {
        attempts: Number(parsed?.threePt?.attempts) || 0,
        made: Number(parsed?.threePt?.made) || 0,
        assists: Number(parsed?.threePt?.assists) || 0,
      },
      turnovers: Number(parsed.turnovers) || 0,
      points: Number(parsed.points) || 0,
      rewardSum: Number(parsed.rewardSum) || 0,
      episodeStepsSum: Number(parsed.episodeStepsSum) || 0,
    };
  } catch (e) {
    // Corrupt storage; reset
    return getDefaultStats();
  }
}

export function saveStats(stats) {
  try {
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(stats));
    }
  } catch (e) {
    // ignore
  }
}

export function resetStatsStorage() {
  const fresh = getDefaultStats();
  saveStats(fresh);
  return fresh;
}


