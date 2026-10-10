//! The memory trace — the canonical record (SPEC §2).

/// One trace. Text is canonical; `vector` is a derived index under `model_id`.
#[derive(Clone, Debug, PartialEq)]
pub struct Trace {
    /// ULID-shaped id (first 10 chars = millisecond timestamp, so ids sort by time).
    pub id: String,
    /// A self-contained proposition, ≤ `text_max` characters, no `《》`.
    pub text: String,
    /// Unix seconds when the proposition was stated.
    pub created_at: i64,
    /// `'IANA;+HH:MM'`.
    pub tz: String,
    /// Decay baseline, Unix seconds.
    pub last_recall: i64,
    /// Half-life of retrievability, seconds.
    pub stability: f64,
    /// Integrated by the dream phase (`false` = labile).
    pub consolidated: bool,
    /// The question this fact answers later; what the dream searches the past with.
    pub cue: String,
    /// Embedding model of `vector` (`""` = not indexed yet).
    pub model_id: String,
    /// Unit-norm vector under `model_id`.
    pub vector: Vec<f32>,
}

impl Trace {
    /// The stability band: half-octaves of `stability` (seconds). Within one band traces
    /// decay at nearly the same rate, so the one recalled longest ago is the band's
    /// weakest — an order that never changes with time, so a store can index it
    /// (`(band, last_recall)`); the weakest overall is the weakest of the band heads.
    pub fn band(&self) -> u8 {
        let s = if self.stability >= 1.0 { self.stability } else { 1.0 }; // NaN and junk: band 0
        (2.0 * s.log2()).floor().clamp(0.0, 255.0) as u8
    }
}
