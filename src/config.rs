//! The 19 engine parameters (SPEC §6).

/// Largest value of a parameter that counts traces, cues or calls.
const COUNT_MAX: usize = 1 << 20;
const DAY: f64 = 86_400.0;
const YEAR: f64 = 365.0 * DAY;

/// ENGRAM v2.2 parameters (SPEC §6). Only `cosine_floor`, `theta_related` and
/// `gist_min_cosine` depend on the embedding model (pre-set for EmbeddingGemma).
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Upper bound on the number of traces.
    pub capacity: usize,
    /// S0: stability of a new trace, seconds.
    pub initial_stability: f64,
    /// Retrieval gain: `S ← S·(1 + gain·a·(1−R))`.
    pub spacing_gain: f64,
    /// S_max: no immortal memory.
    pub max_stability: f64,
    /// New traces are protected from eviction this long (hippocampal buffer).
    pub grace_period: f64,
    /// Cosine of unrelated text under the embedding model (activation baseline).
    pub cosine_floor: f64,
    /// Retrievability floor in the recall score.
    pub alpha: f64,
    /// Traces injected per recall.
    pub inject_n: usize,
    /// MMR diversity penalty.
    pub mmr_lambda: f64,
    /// Absolute minimum recall score.
    pub min_score: f64,
    /// Recall threshold relative to the best score.
    pub relative_score: f64,
    /// Character budget of the injected pack.
    pub budget_chars: usize,
    /// Upper bound on query cues.
    pub max_cues: usize,
    /// Reactivation band: how far a cue reaches into the past.
    pub theta_related: f64,
    /// LLM calls per dream.
    pub dream_budget: usize,
    /// Blast radius of one verdict.
    pub dream_max_members: usize,
    /// Confabulation guard: a gist must be this close to an absorbed member.
    pub gist_min_cosine: f64,
    /// Maximum characters of a trace text.
    pub text_max: usize,
    /// Write rate limit.
    pub writes_per_day: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            capacity: 10_000,
            initial_stability: DAY,
            spacing_gain: 3.0,
            max_stability: 10.0 * YEAR,
            grace_period: 3.0 * DAY,
            cosine_floor: 0.4,
            alpha: 0.35,
            inject_n: 5,
            mmr_lambda: 0.3,
            min_score: 0.1,
            relative_score: 0.6,
            budget_chars: 1024,
            max_cues: 8,
            theta_related: 0.55,
            dream_budget: 5,
            dream_max_members: 8,
            gist_min_cosine: 0.5,
            text_max: 170,
            writes_per_day: 1000,
        }
    }
}

impl Config {
    /// Rejects parameter combinations the engine cannot honour.
    pub fn validate(&self) -> Result<(), String> {
        let unit = |x: f64| (0.0..=1.0).contains(&x);
        let reals = [self.initial_stability, self.spacing_gain, self.max_stability, self.grace_period, self.cosine_floor,
                     self.alpha, self.mmr_lambda, self.min_score, self.relative_score, self.theta_related, self.gist_min_cosine];
        if reals.iter().any(|x| !x.is_finite()) {
            return Err("every real-valued parameter must be finite".into());
        }
        if self.max_stability < 1.0 || self.spacing_gain < 0.0 || self.grace_period < 0.0 || self.mmr_lambda < 0.0 {
            return Err("require max_stability >= 1 s and spacing_gain, grace_period, mmr_lambda >= 0".into());
        }
        if !unit(self.min_score) || !(-1.0..=1.0).contains(&self.gist_min_cosine) || self.max_cues == 0 {
            return Err("require min_score in [0, 1], gist_min_cosine in [-1, 1], max_cues >= 1".into());
        }
        if self.capacity == 0 {
            return Err("capacity must be positive".into());
        }
        if !(self.initial_stability > 0.0 && self.initial_stability <= self.max_stability) {
            return Err("require 0 < initial_stability <= max_stability".into());
        }
        if !(0.0..1.0).contains(&self.cosine_floor) {
            return Err("cosine_floor must be in [0, 1)".into());
        }
        if !unit(self.alpha) || !unit(self.relative_score) {
            return Err("alpha and relative_score must be in [0, 1]".into());
        }
        if !(self.theta_related > 0.0 && self.theta_related < 1.0) {
            return Err("theta_related must be in (0, 1)".into());
        }
        if self.dream_max_members < 2 || self.inject_n == 0 || self.budget_chars == 0 || self.text_max == 0 {
            return Err("dream_max_members ≥ 2; inject_n, budget_chars, text_max > 0".into());
        }
        // Counts that size a search or a batch must stay far from usize overflow.
        if [self.inject_n, self.max_cues, self.dream_max_members, self.dream_budget].iter().any(|&n| n > COUNT_MAX) {
            return Err("inject_n, max_cues, dream_max_members and dream_budget must be at most 2^20".into());
        }
        Ok(())
    }
}
