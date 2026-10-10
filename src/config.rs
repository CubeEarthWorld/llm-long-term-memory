//! The 19 engine parameters (SPEC §6) and the `params!` macro that gives a flat
//! parameter struct its defaults and its tolerant JSON form in one declaration.

use serde_json::Value;

/// A parameter value that round-trips through JSON. Setting is tolerant, like a
/// settings form: numbers may arrive as strings, and a value of the wrong kind is
/// ignored (the field keeps its current value).
pub trait Param {
    /// The JSON form.
    fn to_json(&self) -> Value;
    /// Overwrites `self` from `v` when `v` can be read as this type.
    fn set_json(&mut self, v: &Value);
}

fn number(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        Value::Bool(b) => Some(f64::from(u8::from(*b))),
        _ => None,
    }
}

impl Param for f64 {
    fn to_json(&self) -> Value {
        Value::from(*self)
    }
    fn set_json(&mut self, v: &Value) {
        if let Some(x) = number(v) {
            *self = x;
        }
    }
}

impl Param for usize {
    fn to_json(&self) -> Value {
        Value::from(*self)
    }
    /// Truncates like `int(float(x))`; a negative count reads as 0 (which validation
    /// rejects where 0 is meaningless).
    fn set_json(&mut self, v: &Value) {
        if let Some(x) = number(v).filter(|x| x.is_finite()) {
            *self = x.max(0.0) as usize;
        }
    }
}

impl Param for bool {
    fn to_json(&self) -> Value {
        Value::from(*self)
    }
    fn set_json(&mut self, v: &Value) {
        *self = match v {
            Value::Bool(b) => *b,
            Value::String(s) => matches!(s.as_str(), "true" | "True" | "1"),
            Value::Number(n) => n.as_f64() == Some(1.0),
            _ => false,
        };
    }
}

impl Param for String {
    fn to_json(&self) -> Value {
        Value::from(self.as_str())
    }
    fn set_json(&mut self, v: &Value) {
        *self = match v {
            Value::String(s) => s.clone(),
            Value::Null => return,
            other => other.to_string(),
        };
    }
}

/// Declares a flat parameter struct: fields with defaults, `Default`, and
/// `to_json` / `merge_json` (unknown keys are ignored, known ones coerced).
macro_rules! params {
    (
        $(#[$meta:meta])*
        pub struct $name:ident {
            $( $(#[$fmeta:meta])* $field:ident : $ty:ty = $default:expr ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Debug, PartialEq)]
        pub struct $name {
            $( $(#[$fmeta])* pub $field: $ty, )*
        }

        impl Default for $name {
            fn default() -> Self {
                Self { $( $field: $default, )* }
            }
        }

        impl $name {
            /// The parameters as a JSON object, in declaration order.
            pub fn to_json(&self) -> serde_json::Value {
                let mut m = serde_json::Map::new();
                $( m.insert(stringify!($field).to_owned(), $crate::Param::to_json(&self.$field)); )*
                serde_json::Value::Object(m)
            }

            /// Overwrites the fields present in the JSON object `v`.
            pub fn merge_json(&mut self, v: &serde_json::Value) {
                let Some(o) = v.as_object() else { return };
                $( if let Some(x) = o.get(stringify!($field)) {
                    $crate::Param::set_json(&mut self.$field, x);
                } )*
            }
        }
    };
}
#[cfg(feature = "app")]
pub(crate) use params;

const DAY: f64 = 86_400.0;
const YEAR: f64 = 365.0 * DAY;

params! {
    /// ENGRAM v2.1 parameters (SPEC §6). Only `cosine_floor`, `theta_related` and
    /// `gist_min_cosine` depend on the embedding model (pre-set for EmbeddingGemma).
    pub struct Config {
        /// Upper bound on the number of traces.
        capacity: usize = 10_000,
        /// S0: stability of a new trace, seconds.
        initial_stability: f64 = DAY,
        /// Retrieval gain: `S ← S·(1 + gain·a·(1−R))`.
        spacing_gain: f64 = 3.0,
        /// S_max: no immortal memory.
        max_stability: f64 = 10.0 * YEAR,
        /// New traces are protected from eviction this long (hippocampal buffer).
        grace_period: f64 = 3.0 * DAY,
        /// Cosine of unrelated text under the embedding model (activation baseline).
        cosine_floor: f64 = 0.4,
        /// Retrievability floor in the recall score.
        alpha: f64 = 0.35,
        /// Traces injected per recall.
        inject_n: usize = 5,
        /// MMR diversity penalty.
        mmr_lambda: f64 = 0.3,
        /// Absolute minimum recall score.
        min_score: f64 = 0.1,
        /// Recall threshold relative to the best score.
        relative_score: f64 = 0.6,
        /// Character budget of the injected pack.
        budget_chars: usize = 1024,
        /// Upper bound on query cues.
        max_cues: usize = 8,
        /// Reactivation band: how far a cue reaches into the past.
        theta_related: f64 = 0.55,
        /// LLM calls per dream.
        dream_budget: usize = 5,
        /// Blast radius of one verdict.
        dream_max_members: usize = 8,
        /// Confabulation guard: a gist must be this close to an absorbed member.
        gist_min_cosine: f64 = 0.5,
        /// Maximum characters of a trace text.
        text_max: usize = 170,
        /// Write rate limit.
        writes_per_day: usize = 1000,
    }
}

impl Config {
    /// Rejects parameter combinations the engine cannot honour.
    pub fn validate(&self) -> Result<(), String> {
        let unit = |x: f64| (0.0..=1.0).contains(&x);
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
        Ok(())
    }
}
