// Simple localStorage-backed stats store for episode aggregates

const STORAGE_KEY = 'bw_stats_v1';

function normalizeNumberRecord(raw) {
  const out = {};
  if (!raw || typeof raw !== 'object') return out;
  for (const [key, val] of Object.entries(raw)) {
    const num = Number(val);
    out[String(key)] = Number.isFinite(num) ? num : 0;
  }
  return out;
}

export function getDefaultStats() {
  return {
    episodes: 0,
    dunk: { attempts: 0, made: 0, assists: 0, potentialAssists: 0 },
    twoPt: { attempts: 0, made: 0, assists: 0, potentialAssists: 0 },
    threePt: { attempts: 0, made: 0, assists: 0, potentialAssists: 0 },
    turnovers: 0,
    rebounds: {
      offensive: 0,
      defensive: 0,
      byPlayer: {},
      byPlayerOffensive: {},
      byPlayerDefensive: {},
      targetDistanceSumOffense: 0,
      targetDistanceSumDefense: 0,
      targetDistanceCount: 0,
      postOrbSamples: 0,
      postOrbPoints: 0,
      postOrbValueSamples: 0,
      postOrbConsensusValue: 0,
      postOrbOffenseValue: 0,
      postOrbDefenseValue: 0,
    },
    violations: {
      defensiveLane: 0,
      offensiveThreeSeconds: 0,
    },
    points: 0,
    rewardSum: 0,
    episodeStepsSum: 0,
    intentSelectionCounts: {},
    intentInactiveCount: 0,
    turnoverReasons: {},
    actionMix: {
      noop: 0,
      move: 0,
      shoot: 0,
      pass: 0,
      other: 0,
      total: 0,
    },
    actionMixHolder: {
      noop: 0,
      move: 0,
      shoot: 0,
      pass: 0,
      other: 0,
      total: 0,
    },
    rewardBreakdown: {
      totalReward: 0,
      expectedPoints: 0,
      passReward: 0,
      violationReward: 0,
      assistPotential: 0,
      assistFullBonus: 0,
      phiShaping: 0,
      unexplained: 0,
    },
    selectorDiagnostics: {
      source: "intent_fallback",
      enabled: false,
      specEnabled: false,
      configEnabled: false,
      configMode: "",
      disabledReason: "",
      mode: "",
      appliedEpisodeStarts: 0,
      rawProbCount: 0,
      meanRawMaxProb: 0,
      meanRawProbs: [],
      sampleCounts: {},
      argmaxCounts: {},
    },
    reboundDiagnostics: {
      globalContestRate: 0,
      totalGlobalContests: 0,
      eligibility: {},
      resolvedParams: {},
    },
    valueDiagnostics: {
      discount_gamma: 0,
      sample_count: 0,
      completed_episode_count: 0,
      offense_value_mean: 0,
      defense_value_mean: 0,
      value_sum_mean: 0,
      value_sum_abs_mean: 0,
      offense_return_mean: 0,
      defense_return_mean: 0,
      return_sum_mean: 0,
      return_sum_abs_mean: 0,
      offense_value_bias_mean: 0,
      defense_value_bias_mean: 0,
      offense_value_mae: 0,
      defense_value_mae: 0,
    },
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
        potentialAssists: Number(parsed?.dunk?.potentialAssists) || 0,
      },
      twoPt: {
        attempts: Number(parsed?.twoPt?.attempts) || 0,
        made: Number(parsed?.twoPt?.made) || 0,
        assists: Number(parsed?.twoPt?.assists) || 0,
        potentialAssists: Number(parsed?.twoPt?.potentialAssists) || 0,
      },
      threePt: {
        attempts: Number(parsed?.threePt?.attempts) || 0,
        made: Number(parsed?.threePt?.made) || 0,
        assists: Number(parsed?.threePt?.assists) || 0,
        potentialAssists: Number(parsed?.threePt?.potentialAssists) || 0,
      },
      turnovers: Number(parsed.turnovers) || 0,
      rebounds: {
        offensive: Number(parsed?.rebounds?.offensive) || 0,
        defensive: Number(parsed?.rebounds?.defensive) || 0,
        byPlayer: normalizeNumberRecord(parsed?.rebounds?.byPlayer || parsed?.rebounds?.byPlayerOffensive),
        byPlayerOffensive: normalizeNumberRecord(parsed?.rebounds?.byPlayerOffensive || parsed?.rebounds?.byPlayer),
        byPlayerDefensive: normalizeNumberRecord(parsed?.rebounds?.byPlayerDefensive),
        targetDistanceSumOffense: Number(parsed?.rebounds?.targetDistanceSumOffense) || 0,
        targetDistanceSumDefense: Number(parsed?.rebounds?.targetDistanceSumDefense) || 0,
        targetDistanceCount: Number(parsed?.rebounds?.targetDistanceCount) || 0,
        postOrbSamples: Number(parsed?.rebounds?.postOrbSamples) || 0,
        postOrbPoints: Number(parsed?.rebounds?.postOrbPoints) || 0,
        postOrbValueSamples: Number(parsed?.rebounds?.postOrbValueSamples) || 0,
        postOrbConsensusValue: Number(parsed?.rebounds?.postOrbConsensusValue) || 0,
        postOrbOffenseValue: Number(parsed?.rebounds?.postOrbOffenseValue) || 0,
        postOrbDefenseValue: Number(parsed?.rebounds?.postOrbDefenseValue) || 0,
      },
      violations: {
        defensiveLane: Number(parsed?.violations?.defensiveLane) || 0,
        offensiveThreeSeconds: Number(parsed?.violations?.offensiveThreeSeconds) || 0,
      },
      points: Number(parsed.points) || 0,
      rewardSum: Number(parsed.rewardSum) || 0,
      episodeStepsSum: Number(parsed.episodeStepsSum) || 0,
      intentSelectionCounts: normalizeNumberRecord(parsed.intentSelectionCounts),
      intentInactiveCount: Number(parsed.intentInactiveCount) || 0,
      turnoverReasons: normalizeNumberRecord(parsed.turnoverReasons),
      actionMix: {
        noop: Number(parsed?.actionMix?.noop) || 0,
        move: Number(parsed?.actionMix?.move) || 0,
        shoot: Number(parsed?.actionMix?.shoot) || 0,
        pass: Number(parsed?.actionMix?.pass) || 0,
        other: Number(parsed?.actionMix?.other) || 0,
        total: Number(parsed?.actionMix?.total) || 0,
      },
      actionMixHolder: {
        noop: Number(parsed?.actionMixHolder?.noop) || 0,
        move: Number(parsed?.actionMixHolder?.move) || 0,
        shoot: Number(parsed?.actionMixHolder?.shoot) || 0,
        pass: Number(parsed?.actionMixHolder?.pass) || 0,
        other: Number(parsed?.actionMixHolder?.other) || 0,
        total: Number(parsed?.actionMixHolder?.total) || 0,
      },
      rewardBreakdown: {
        totalReward: Number(parsed?.rewardBreakdown?.totalReward) || 0,
        expectedPoints: Number(parsed?.rewardBreakdown?.expectedPoints) || 0,
        passReward: Number(parsed?.rewardBreakdown?.passReward) || 0,
        violationReward: Number(parsed?.rewardBreakdown?.violationReward) || 0,
        assistPotential: Number(parsed?.rewardBreakdown?.assistPotential) || 0,
        assistFullBonus: Number(parsed?.rewardBreakdown?.assistFullBonus) || 0,
        phiShaping: Number(parsed?.rewardBreakdown?.phiShaping) || 0,
        unexplained: Number(parsed?.rewardBreakdown?.unexplained) || 0,
      },
      selectorDiagnostics: {
        source: String(parsed?.selectorDiagnostics?.source || "intent_fallback"),
        enabled: Boolean(parsed?.selectorDiagnostics?.enabled),
        specEnabled: Boolean(parsed?.selectorDiagnostics?.specEnabled),
        configEnabled: Boolean(parsed?.selectorDiagnostics?.configEnabled),
        configMode: String(parsed?.selectorDiagnostics?.configMode || ""),
        disabledReason: String(parsed?.selectorDiagnostics?.disabledReason || ""),
        mode: String(parsed?.selectorDiagnostics?.mode || ""),
        appliedEpisodeStarts: Number(parsed?.selectorDiagnostics?.appliedEpisodeStarts) || 0,
        rawProbCount: Number(parsed?.selectorDiagnostics?.rawProbCount) || 0,
        meanRawMaxProb: Number(parsed?.selectorDiagnostics?.meanRawMaxProb) || 0,
        meanRawProbs: Array.isArray(parsed?.selectorDiagnostics?.meanRawProbs)
          ? parsed.selectorDiagnostics.meanRawProbs.map((v) => Number(v) || 0)
          : [],
        sampleCounts: normalizeNumberRecord(parsed?.selectorDiagnostics?.sampleCounts),
        argmaxCounts: normalizeNumberRecord(parsed?.selectorDiagnostics?.argmaxCounts),
      },
      reboundDiagnostics: {
        globalContestRate: Number(parsed?.reboundDiagnostics?.globalContestRate) || 0,
        totalGlobalContests: Number(parsed?.reboundDiagnostics?.totalGlobalContests) || 0,
        eligibility: (parsed?.reboundDiagnostics?.eligibility && typeof parsed.reboundDiagnostics.eligibility === 'object')
          ? { ...parsed.reboundDiagnostics.eligibility }
          : {},
        resolvedParams: (parsed?.reboundDiagnostics?.resolvedParams && typeof parsed.reboundDiagnostics.resolvedParams === 'object')
          ? { ...parsed.reboundDiagnostics.resolvedParams }
          : {},
      },
      valueDiagnostics: (parsed?.valueDiagnostics && typeof parsed.valueDiagnostics === 'object')
        ? { ...parsed.valueDiagnostics }
        : getDefaultStats().valueDiagnostics,
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
