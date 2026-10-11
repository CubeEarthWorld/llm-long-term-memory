//! The ENGRAM v2.2 engine (SPEC §3–§5): traces with a forgetting curve, the wake-phase
//! verbs remember / recall / cite / forget, and the sleep-phase dream.
//!
//! The engine holds no copy of the memory. Every question goes to the [`Storage`] — a
//! trace by id or text, an ordered view, a similarity search — so opening costs
//! nothing and no operation touches more than the traces it is about.

use crate::dream::{Adjudicator, DreamDecision, DreamMember, DreamRequest};
use crate::guard::Guarded;
use crate::id::{ulid_ids, IdGen};
use crate::storage::{Hit, Storage, View, Write};
use crate::text::{clean_text, cues, valid_id};
use crate::time::{clean_tz, fmt_local, system_clock, tz_field, Clock};
use crate::vector::{dot, normalized};
use crate::{Config, Embedder, Result, Trace};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Seeds examined per LLM call.
const SEEDS_PER_BUDGET: usize = 8;
/// Traces re-embedded per embedder call.
const REINDEX_BATCH: usize = 64;
/// Rows read per stability band beyond those needed, in the first window of an eviction
/// (the window then grows while a weaker row could still lie beyond it).
const BAND_DEPTH: usize = 8;
/// Most rows read from one stability band to prove which trace is the weakest.
const BAND_CAP: usize = 128;
/// `log2 strength` at or below which a trace is dead (strength ≤ 2⁻⁶⁴ s): which
/// of two dead traces goes first no longer matters.
const DEAD: f64 = -64.0;
/// Unusable rows read past before a read of an ordered view gives up widening.
const JUNK_READ: usize = 1024;
/// Most rows one similarity search asks a store for while widening past unusable ones.
const NEAR_REACH_MAX: usize = 1 << 20;
/// Labile rows read per page of the dream queue.
const SEED_PAGE: usize = 64;
/// How far from unit length a stored vector may be (‖v‖² − 1) before it is not trusted:
/// room for a store that quantises its vectors, none for a rescaled or zero one.
const VECTOR_DRIFT: f64 = 0.05;
/// Traces with one text read to find a usable one before a write counts as new.
const TEXT_PROBE: usize = 8;
/// Longest stored timezone field (`'IANA name;+HH:MM'`).
const TZ_MAX: usize = 64;

/// What [`Engram::remember`] did.
#[derive(Clone, Debug, PartialEq)]
pub enum Remembered {
    /// A new trace. `cosine` is how strongly its cue reactivated the nearest older
    /// trace (0 when none did); `evicted` is the last trace forgotten to make room —
    /// `trace` itself when it was the weakest (then it is not stored).
    Inserted { trace: Trace, cosine: f64, evicted: Option<Trace> },
    /// The exact text was already stored: rehearsed instead (full activation).
    Reinforced(Trace),
    /// The text was empty after cleaning.
    Rejected,
    /// The daily write limit was reached; `text` is the cleaned text.
    RateLimited { text: String },
}

/// One injected trace (as it was before the exposure update).
#[derive(Clone, Debug, PartialEq)]
pub struct Recalled {
    pub trace: Trace,
    pub score: f64,
    pub cosine: f64,
    pub retrievability: f64,
}

/// The outcome of a recall: the pack to inject and what is in it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Recall {
    /// `[unix tz] text　《id:…》` lines, at most `budget_chars` (the first always fits).
    pub pack_text: String,
    pub recalled: Vec<Recalled>,
}

/// A dream verdict as applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DreamAction {
    Keep,
    Replace,
    /// The adjudicator (or the gist embedding) failed: the seed stays labile.
    Error,
}

impl DreamAction {
    /// `"keep"`, `"replace"` or `"error"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Keep => "keep",
            Self::Replace => "replace",
            Self::Error => "error",
        }
    }
}

/// One adjudicated cluster — also its undo record ([`Engram::undo`]).
#[derive(Clone, Debug, PartialEq)]
pub struct DreamReport {
    pub action: DreamAction,
    /// The cluster as the adjudicator saw it, seed first.
    pub before: Vec<DreamMember>,
    /// The traces a replace removed (the seed and the named candidates), as they were.
    pub absorbed: Vec<Trace>,
    /// The gists a replace inserted.
    pub after: Vec<Trace>,
    pub error: Option<String>,
}

/// Counts for dashboards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stats {
    pub records: usize,
    pub labile: usize,
}

/// One cluster's report, and whether it failed because the store refused the write.
struct Adjudicated {
    report: DreamReport,
    stored_error: bool,
}

/// One stability band's rows as read for eviction, ranked, and how far they got.
struct Window {
    rows: Vec<(f64, Trace)>,
    /// Rows asked for, whether the band held at least that many, and the time since the
    /// last row read was recalled.
    limit: usize,
    more: bool,
    last_dt: f64,
}

/// What one recall cue left behind for [`Engram::cite`].
#[derive(Clone, Copy, Debug)]
struct Exposure {
    cosine: f64,
    retrievability: f64,
}

/// A long-term memory (ENGRAM v2.2) over a [`Storage`]. Its own state is bounded: the
/// parameters, the last day's write times and the last recall's exposures.
pub struct Engram {
    cfg: Config,
    store: Box<dyn Storage>,
    embedder: Box<dyn Embedder>,
    clock: Clock,
    ids: IdGen,
    timezone: String,
    write_times: Vec<f64>,
    exposures: HashMap<String, Exposure>,
    /// The id of the last seed a dream examined: the next dream continues after it and
    /// wraps around, so the labile queue is served as a cycle. A seed that cannot be
    /// settled (the verdict failed, the cue could not be embedded) is met again only
    /// after the others, and never blocks the ones behind it — however many there are.
    cursor: String,
    /// Where the scrub in [`Engram::reindex`] resumes: the last row it checked.
    scrub: String,
}

impl Engram {
    /// An engine over `store` (any [`Storage`]; [`crate::Store`] is bundled) on the wall clock. Nothing is loaded.
    pub fn open(store: Box<dyn Storage>, embedder: Box<dyn Embedder>, cfg: Config, timezone: &str) -> Result<Self> {
        Self::with_clock(store, embedder, cfg, timezone, system_clock())
    }

    /// [`Engram::open`] with an injected clock.
    pub fn with_clock(store: Box<dyn Storage>, embedder: Box<dyn Embedder>, cfg: Config, timezone: &str, clock: Clock) -> Result<Self> {
        cfg.validate()?;
        Ok(Self { cfg, store: Box::new(Guarded(store)), embedder, clock, ids: ulid_ids(), timezone: timezone.to_owned(), write_times: Vec::new(), exposures: HashMap::new(), cursor: String::new(), scrub: String::new() })
    }

    // ------------------------------------------------------------------ clock

    /// Current Unix time, seconds.
    pub fn now(&self) -> f64 {
        let t = (self.clock)();
        if t.is_nan() { 0.0 } else { t.clamp(f64::MIN, f64::MAX) } // every time the engine reads comes from here: finite
    }

    /// Where the dream's queue resumes (the last seed examined; `""`: the start). It lives
    /// in the engine: a host that opens a fresh engine for every dream keeps it itself.
    pub fn dream_cursor(&self) -> &str {
        &self.cursor
    }

    /// Resumes the dream's queue after `id` (see [`Engram::dream_cursor`]).
    pub fn set_dream_cursor(&mut self, id: &str) {
        self.cursor = id.to_owned();
    }

    /// Replaces the time source (seed replays run on a virtual clock).
    pub fn set_clock(&mut self, clock: Clock) {
        self.clock = clock;
    }

    /// Replaces the id generator (monotonic ULIDs by default).
    pub fn set_ids(&mut self, ids: IdGen) {
        self.ids = ids;
    }

    /// Current local time, e.g. `'2026-06-12 09:30 +09:00'`.
    pub fn now_local(&self) -> String {
        self.local_at(self.now())
    }

    fn local_at(&self, now: f64) -> String {
        fmt_local(now as i64, &tz_field(&self.timezone, now))
    }

    // ------------------------------------------------- the forgetting curve (§3)

    /// `R = 2^(−max(0, now − last_recall) / stability)`.
    /// Stability is read as the engine reads every row ([`Self::clamp_stability`]), so a
    /// row straight from the store (zero, negative, NaN) cannot leave `[0, 1]`.
    pub fn retrievability(&self, t: &Trace, now: f64) -> f64 {
        (-(now - t.last_recall as f64).max(0.0) / self.clamp_stability(t.stability)).exp2()
    }

    /// `S·R` — proportional to the trace's total remaining retrievability.
    pub fn strength(&self, t: &Trace, now: f64) -> f64 {
        self.clamp_stability(t.stability) * self.retrievability(t, now)
    }

    /// A stored vector the engine can search with: the model's dimension, finite, and of
    /// unit length up to what a quantising store loses (VECTOR_DRIFT). The store ranked and
    /// thresholded by the very same numbers, so the engine neither rescales nor recomputes.
    fn sound(&self, v: &[f32]) -> bool {
        let n2 = dot(v, v);
        v.len() == self.embedder.dimension() && n2.is_finite() && (n2 - 1.0).abs() <= VECTOR_DRIFT
    }

    /// NaN would survive min/max and never fade: an immortal trace.
    fn clamp_stability(&self, s: f64) -> f64 {
        (if s.is_nan() { 1.0 } else { s }).max(1.0).min(self.cfg.max_stability)
    }

    /// Cue activation: the cosine rescaled above the model's baseline.
    fn activation(&self, cos: f64) -> f64 {
        ((cos.min(1.0) - self.cfg.cosine_floor) / (1.0 - self.cfg.cosine_floor)).max(0.0)
    }

    /// The retrieval update as a state write: stability grows with the spacing achieved
    /// (`1 − R`) and the cue activation `a`; `last_recall` moves to now.
    fn retrieved<'a>(&self, t: &'a Trace, now: f64, a: f64) -> Write<'a> {
        Write::State { id: &t.id, last_recall: now as i64, stability: self.grown(t, now, a), consolidated: t.consolidated }
    }

    /// The stability after a retrieval at activation `a`.
    fn grown(&self, t: &Trace, now: f64, a: f64) -> f64 {
        self.clamp_stability(t.stability * (1.0 + self.cfg.spacing_gain * a * (1.0 - self.retrievability(t, now))))
    }

    // ---------------------------------------------------------------- reading

    /// A row as the engine may use it. The store is outside the engine's control (an
    /// older version or a hand edit may have written it), so every row read is held to
    /// the write-time rules: id valid, text, cue and timezone cleaned, stability
    /// clamped, a vector with a non-finite component unindexed. `None`: unusable.
    fn checked(&self, t: Trace) -> Option<Trace> {
        let text = clean_text(&t.text, self.cfg.text_max);
        if !valid_id(&t.id) || text.is_empty() {
            return None;
        }
        // A vector the engine did not write (finite, unit length) cannot be trusted as an index.
        let (model_id, vector) = if t.vector.is_empty() || self.sound(&t.vector) {
            (t.model_id, t.vector)
        } else {
            (String::new(), Vec::new()) // unindexed: the scrub in reindex() embeds it again
        };
        Some(Trace {
            text,
            cue: clean_text(&t.cue, self.cfg.text_max),
            tz: clean_tz(&t.tz, TZ_MAX),
            stability: self.clamp_stability(t.stability),
            model_id,
            vector,
            ..t
        })
    }

    /// Up to `limit` usable rows of `view`. The store lists rows the engine cannot use
    /// (empty text, bad id, an unsound vector), so when some are dropped the read widens
    /// until `limit` usable rows are found or the view ends: no "first N" is ever
    /// answered from junk alone. Every read of an ordered view goes through here.
    fn find(&self, view: View<'_>, limit: usize) -> Result<Vec<Trace>> {
        // Widening is bounded: a view headed by more unusable rows than JUNK_READ + 8·limit
        // is read no further (eviction removes such rows first, so this is a damaged store).
        let reach = limit.saturating_mul(8).saturating_add(JUNK_READ);
        let mut ask = limit;
        loop {
            let rows = self.store.find(view, ask)?;
            let exhausted = rows.len() < ask;
            let usable: Vec<Trace> = rows.into_iter().filter_map(|t| self.checked(t)).collect();
            if usable.len() >= limit || exhausted || ask >= reach {
                return Ok(usable.into_iter().take(limit).collect());
            }
            ask = ask.saturating_mul(2).min(reach);
        }
    }

    /// The `k` best usable hits, and the cosine of the k-th when the store may hold more.
    /// The store ranks rows it cannot judge, so when some are dropped the search widens
    /// until `k` usable hits are found or the store runs out: junk nearer than the real
    /// answer never hides it. The cosine is recomputed from the vector that passed
    /// [`Self::sound`] — with the same `dot` the store uses, so an honest store gives the
    /// same number — and held to `min_cos`; it is never above 1 and a non-finite one is no match.
    fn near(&self, q: &[f32], min_cos: f64, k: usize, stated_by: Option<i64>) -> Result<(Vec<Hit>, Option<f64>)> {
        // No search can return more rows than the store holds, so that bounds the widening (and
        // a store that answers every ask in full, however large, is not asked for ever more).
        let reach = self.store.count(View::All)?.min(NEAR_REACH_MAX).max(k);
        let mut ask = k;
        let usable = loop {
            let hits = self.store.near(q, self.embedder.model_id(), min_cos, ask, stated_by)?;
            let exhausted = hits.len() < ask;
            let usable: Vec<Hit> = hits
                .into_iter()
                .filter(|h| h.trace.vector.len() == q.len())
                .filter_map(|h| {
                    let trace = self.checked(h.trace)?;
                    if trace.vector.len() != q.len() {
                        return None;
                    }
                    let cos = dot(&trace.vector, q);
                    (cos.is_finite() && cos >= min_cos).then(|| Hit { trace, cos: cos.min(1.0) })
                })
                .collect();
            if usable.len() >= k || exhausted || ask >= reach {
                break usable;
            }
            ask = ask.saturating_mul(2).min(reach);
        };
        let kth = (usable.len() >= k).then(|| usable[k.max(1) - 1].cos);
        Ok((usable.into_iter().take(k).collect(), kth))
    }

    /// Applies one operation's writes as one atomic commit. Everything that removes a trace
    /// goes through here, so a removed trace cannot keep a citation window open: a later
    /// trace under the same id (a generator derived from text and time repeats ids) would
    /// inherit its boost.
    fn commit(&mut self, writes: &[Write<'_>]) -> Result<()> {
        if writes.is_empty() {
            return Ok(());
        }
        self.store.write(writes)?;
        for w in writes {
            if let Write::Remove(id) = w {
                self.exposures.remove(*id);
            }
        }
        Ok(())
    }

    /// The trace with `id`.
    pub fn memory(&self, id: &str) -> Result<Option<Trace>> {
        Ok(self.find(View::Id(id), 1)?.pop())
    }

    /// The active parameters.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// The store (read-only: every write goes through the engine).
    pub fn store(&self) -> &dyn Storage {
        self.store.as_ref()
    }

    /// Counts of all traces and of the labile ones the dream can take (indexed under the current model).
    pub fn stats(&self) -> Result<Stats> {
        let model_id = self.embedder.model_id();
        Ok(Stats { records: self.store.count(View::All)?, labile: self.store.count(View::Labile { model_id, after: "" })? })
    }

    /// Erases everything.
    pub fn reset(&mut self) -> Result<()> {
        self.store.clear()?;
        self.write_times.clear();
        self.exposures.clear();
        self.cursor.clear();
        self.scrub.clear();
        Ok(())
    }

    /// A fresh, valid ([`valid_id`]) id no trace (nor `taken`) has: a gist that repeats
    /// its seed's text in the same millisecond must not inherit — and then lose — the
    /// seed's id under a generator derived from text and time. A generator stuck on
    /// invalid or used ids is an error.
    fn new_id(&mut self, millis: i64, text: &str, taken: &[String]) -> Result<String> {
        for _ in 0..64 {
            let id = (self.ids)(millis, text);
            if valid_id(&id) && !taken.contains(&id) && self.store.find(View::Id(&id), 1)?.is_empty() {
                return Ok(id);
            }
        }
        Err("the id generator keeps returning ids that are invalid or in use".into())
    }

    /// A freshly embedded vector, normalised — or `None` when the embedder contradicted
    /// its own dimension or returned a zero or non-finite vector (handled like an
    /// offline embedder: indexed on a later attempt).
    fn vector(&self, v: &[f32]) -> Option<Vec<f32>> {
        let n = normalized(v);
        let n2 = dot(&n, &n);
        (v.len() == self.embedder.dimension() && n2.is_finite() && (n2 - 1.0).abs() <= 1e-3).then_some(n)
    }

    /// Re-embeds up to `limit` traces not indexed under the current model, newest first,
    /// and scrubs: one page of rows per call, resuming where the last call stopped and
    /// wrapping around, is checked for a vector the engine cannot use (a wrong dimension,
    /// a rescaled, zero or non-finite vector under the current model, which the store
    /// still lists as indexed) and those are embedded again. An offline embedder leaves
    /// them for a later call. Returns how many were indexed.
    pub fn reindex(&mut self, limit: usize) -> Result<usize> {
        if limit == 0 {
            return Ok(0);
        }
        let model_id = self.embedder.model_id().to_owned();
        let mut stale = self.find(View::Stale { model_id: &model_id }, limit)?;
        let page = self.store.find(View::After(&self.scrub), REINDEX_BATCH.min(limit))?;
        self.scrub = page.last().map_or_else(String::new, |t| t.id.clone());
        for t in page {
            let unsound = t.model_id == model_id && (t.vector.is_empty() || !self.sound(&t.vector));
            if let Some(t) = self.checked(t).filter(|t| unsound && !stale.iter().any(|s| s.id == t.id)) {
                stale.push(Trace { model_id: String::new(), vector: Vec::new(), ..t });
            }
        }
        let mut done = 0;
        for batch in stale.chunks(REINDEX_BATCH) {
            let texts: Vec<&str> = batch.iter().map(|t| t.text.as_str()).collect();
            // An embedder that fails, or answers with a different count, is offline for now.
            let Some(raw) = self.embedder.embed_documents(&texts).ok().filter(|v| v.len() == texts.len()) else { break };
            let vectors: Vec<(&Trace, Vec<f32>)> = batch.iter().zip(&raw).filter_map(|(t, v)| Some((t, self.vector(v)?))).collect();
            let writes: Vec<Write<'_>> = vectors.iter().map(|(t, v)| Write::Index { id: &t.id, model_id: &model_id, vector: v }).collect();
            self.commit(&writes)?;
            done += writes.len();
        }
        Ok(done)
    }

    // ------------------------------------------------------- wake phase (§4)

    /// Stores one self-contained proposition. Exact duplicates are rehearsed; anything
    /// else is inserted, never overwritten. A write whose `cue` reactivates an older
    /// trace is born labile for the dream to adjudicate. Over capacity, the weakest trace
    /// past the grace period is forgotten. `salience` (0–10) scales S0.
    pub fn remember(&mut self, text: &str, salience: f64, cue: &str) -> Result<Remembered> {
        let now = self.now();
        let text = clean_text(text, self.cfg.text_max);
        let cue = clean_text(cue, self.cfg.text_max);
        if text.is_empty() {
            return Ok(Remembered::Rejected);
        }
        // The first usable trace with this text (an unusable older one must not hide it).
        if let Some(t) = self.find(View::Text(&text), TEXT_PROBE)?.into_iter().next() {
            let w = self.retrieved(&t, now, 1.0);
            self.commit(&[w])?;
            // Built here, not read back: the write has landed, so no later failure may report an error.
            return Ok(Remembered::Reinforced(Trace { last_recall: now as i64, stability: self.grown(&t, now, 1.0), ..t }));
        }
        let day_ago = now - 86_400.0;
        // A clock stepped back leaves recent writes stamped slightly ahead: they still count.
        // Writes further ahead than a day (a clock that ran far ahead) do not.
        self.write_times.retain(|&t| t >= day_ago && t <= now + 86_400.0);
        if self.write_times.len() >= self.cfg.writes_per_day {
            return Ok(Remembered::RateLimited { text });
        }
        let v = self.embedder.embed_documents(&[&text]).ok().and_then(|vs| vs.into_iter().next()).and_then(|v| self.vector(&v));
        let model_id = self.embedder.model_id().to_owned();
        let (nearest, cued) = match &v {
            Some(v) => {
                let (q, cued) = self.cue_vectors(&[(cue.as_str(), v.as_slice())]).remove(0);
                (self.near(&q, self.cfg.theta_related, 1, None)?.0.pop(), cued)
            }
            None => (None, false),
        };
        // A cue that could not be embedded, or an unindexed trace, might be what this
        // write reactivates: then it is not known to reactivate nothing, and it stays
        // labile until a dream can tell.
        let consolidated = cued && nearest.is_none() && self.find(View::Stale { model_id: &model_id }, 1)?.is_empty();
        let sal = if salience.is_nan() { 1.0 } else { salience.clamp(0.0, 10.0) };
        let trace = Trace {
            id: self.new_id((now * 1000.0) as i64, &text, &[])?,
            text,
            created_at: now as i64,
            tz: clean_tz(&tz_field(&self.timezone, now), TZ_MAX),
            last_recall: now as i64,
            stability: self.clamp_stability(self.cfg.initial_stability * sal),
            consolidated,
            cue,
            model_id: if v.is_some() { model_id } else { String::new() },
            vector: v.unwrap_or_default(),
        };
        let victims = self.victims(&trace, now)?;
        let mut writes = vec![Write::Insert(&trace)];
        writes.extend(victims.iter().map(|t| Write::Remove(&t.id)));
        self.commit(&writes)?;
        self.write_times.push(now); // counted once the write has landed: a failed one costs no slot
        Ok(Remembered::Inserted { cosine: nearest.map_or(0.0, |h| h.cos), evicted: victims.iter().find(|v| v.id == trace.id).or(victims.last()).cloned(), trace })
    }

    /// The traces to forget so that `new` fits, weakest first (lowest `log2 strength`)
    /// among the traces past the grace period. Traces inside it are protected unless
    /// they exceed a tenth of the capacity (a flood evicts its own members) or too few
    /// older traces exist. `new` itself may be the victim. The candidates are the heads
    /// of every stability band ([`Trace::band`]) — every band, not only those up to the
    /// configured maximum: a row stored above it (a lowered cap, another writer) must
    /// still die. One short index read per band. Rows are ranked as [`Self::checked`]
    /// reads them; a row it cannot use counts against the capacity and dies first.
    fn victims(&self, new: &Trace, now: f64) -> Result<Vec<Trace>> {
        let total = self.store.count(View::All)?.saturating_add(1);
        let excess = total.saturating_sub(self.cfg.capacity);
        if excess == 0 {
            return Ok(Vec::new());
        }
        let cutoff = (now - self.cfg.grace_period) as i64;
        let young = self.store.count(View::Young { created_after: cutoff })?.saturating_add(usize::from(new.created_at > cutoff));
        let rank = |t: &Trace| t.stability.log2() - (now - t.last_recall as f64).max(0.0) / t.stability;
        let ranked = |t: Trace| match self.checked(t.clone()) {
            Some(c) => (rank(&c), c),
            None => (f64::NEG_INFINITY, t),
        };
        // A band's index order (`last_recall`) is time-invariant, but the rank is not: within a
        // band S differs by up to √2, so the head is only nearly the weakest. A band's
        // window grows until no row beyond it can rank at or below the `excess`-th weakest
        // found (rank grows with S and falls with elapsed time, and both are bounded by
        // the band and the last row read) — or until those are all dead (strength ≤ 2⁻⁶⁴ s, so
        // their order is immaterial) — or until BAND_CAP rows.
        let cap = BAND_CAP.max(excess.saturating_add(BAND_DEPTH));
        let read = |band: u8, created_by: i64, limit: usize| -> Result<Window> {
            let raw = self.store.find(View::Band { band, created_by }, limit)?;
            let last_dt = raw.last().map_or(0.0, |t| (now - t.last_recall as f64).max(0.0));
            Ok(Window { more: raw.len() == limit, limit, last_dt, rows: raw.into_iter().map(ranked).collect() })
        };
        let heads = |created_by: i64| -> Result<Vec<(f64, Trace)>> {
            let mut windows = (0..=u8::MAX).map(|b| read(b, created_by, excess.saturating_add(BAND_DEPTH))).collect::<Result<Vec<_>>>()?;
            loop {
                let mut ranks: Vec<f64> = windows.iter().flat_map(|w| w.rows.iter().map(|r| r.0)).collect();
                if ranks.len() < excess {
                    break;
                }
                let weakest = *ranks.select_nth_unstable_by(excess - 1, f64::total_cmp).1;
                if weakest <= DEAD {
                    break;
                }
                let mut grew = false;
                for (b, w) in windows.iter_mut().enumerate() {
                    let s_lo = (b as f64 / 2.0).exp2().min(self.cfg.max_stability) * (1.0 - 1e-9);
                    if w.more && w.limit < cap && s_lo.log2() - w.last_dt / s_lo <= weakest {
                        *w = read(b as u8, created_by, (w.limit * 4).min(cap))?;
                        grew = true;
                    }
                }
                if !grew {
                    break;
                }
            }
            Ok(windows.into_iter().flat_map(|w| w.rows).collect())
        };
        let mut pool = if young <= self.cfg.capacity / 10 { heads(cutoff)? } else { Vec::new() };
        if pool.len() < excess {
            pool = heads(i64::MAX)?;
            pool.push((rank(new), new.clone()));
        }
        pool.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.id.cmp(&b.1.id)));
        pool.truncate(excess);
        Ok(pool.into_iter().map(|(_, t)| t).collect())
    }

    /// Retrieves up to `inject_n` traces relevant to `query` and packs them for prompt
    /// injection. Injection is exposure, not use: each injected trace gets the retrieval
    /// update at half its cue activation; [`Engram::cite`] completes it.
    pub fn recall(&mut self, query: &str) -> Result<Recall> {
        let now = self.now();
        self.exposures.clear(); // a recall always ends the previous citation window
        let recall = self.select(query, now)?;
        let writes: Vec<Write<'_>> = recall.recalled.iter().map(|c| self.retrieved(&c.trace, now, 0.5 * self.activation(c.cosine))).collect();
        self.commit(&writes)?;
        for c in &recall.recalled {
            self.exposures.insert(c.trace.id.clone(), Exposure { cosine: c.cosine, retrievability: c.retrievability });
        }
        Ok(recall)
    }

    /// What [`Engram::recall`] would inject, without strengthening anything (for
    /// measurement and previews).
    pub fn peek(&mut self, query: &str) -> Result<Recall> {
        let now = self.now();
        self.select(query, now)
    }

    /// The recall pool, exactly: each cue asks the store for its nearest traces above
    /// the cosine no qualifying trace can be below, and the search widens until no trace
    /// it has not returned could still clear the score floor.
    fn select(&mut self, query: &str, now: f64) -> Result<Recall> {
        let parts = cues(query, self.cfg.max_cues);
        if parts.is_empty() || self.store.count(View::All)? == 0 {
            return Ok(Recall::default()); // nothing to score: do not pay for an embedding
        }
        let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
        let qs: Option<Vec<Vec<f32>>> = match self.embedder.embed_queries(&refs) {
            Ok(raw) if raw.len() == parts.len() => raw.iter().map(|v| self.vector(v)).collect(),
            _ => None, // offline, or it contradicted itself: nothing to cue with
        };
        let Some(qs) = qs else { return Ok(Recall::default()) };
        if qs.iter().any(|q| q.len() != qs[0].len()) {
            return Ok(Recall::default()); // an embedder whose dimension moved mid-call: nothing to compare
        }
        let total = self.store.count(View::All)?;
        // score = a(cos)·(α + (1−α)·R) ≤ a(cos): below this cosine no trace reaches min_score.
        let min_cos = self.cfg.cosine_floor + self.cfg.min_score * (1.0 - self.cfg.cosine_floor);
        let mut k = self.cfg.inject_n.saturating_mul(4);
        let pool = loop {
            let mut seen: BTreeMap<String, Trace> = BTreeMap::new(); // id order = insertion order
            let mut unseen_cos = f64::NEG_INFINITY; // the most a trace not returned could have
            for q in &qs {
                let (hits, kth) = self.near(q, min_cos, k, None)?;
                if let Some(c) = kth {
                    unseen_cos = unseen_cos.max(c);
                }
                seen.extend(hits.into_iter().map(|h| (h.trace.id.clone(), h.trace)));
            }
            let scored: Vec<(Trace, f64, f64, f64)> = seen
                .into_values()
                .filter(|t| t.vector.len() == qs[0].len())
                .map(|t| {
                    let cos = qs.iter().map(|q| dot(&t.vector, q).min(1.0)).fold(f64::NEG_INFINITY, f64::max); // vectors are finite (checked)
                    let rv = self.retrievability(&t, now);
                    let score = self.activation(cos) * (self.cfg.alpha + (1.0 - self.cfg.alpha) * rv);
                    (t, score, cos, rv)
                })
                .collect();
            let top = scored.iter().map(|c| c.1).fold(0.0, f64::max);
            let floor = self.cfg.min_score.max(self.cfg.relative_score * top);
            if unseen_cos == f64::NEG_INFINITY || self.activation(unseen_cos) < floor || k >= total {
                break scored.into_iter().filter(|c| c.1 >= floor).collect::<Vec<_>>();
            }
            k = k.saturating_mul(4);
        };
        let mut out = Recall::default();
        let mut used = 0;
        for j in self.mmr(&pool) {
            let (t, score, cosine, retrievability) = &pool[j];
            let line = format!("[{} {}] {}　《id:{}》\n", t.created_at, t.tz, t.text, t.id);
            let cost = line.chars().count();
            if !out.pack_text.is_empty() && used + cost > self.cfg.budget_chars {
                continue;
            }
            out.pack_text.push_str(&line);
            used += cost;
            out.recalled.push(Recalled { trace: t.clone(), score: *score, cosine: *cosine, retrievability: *retrievability });
        }
        Ok(out)
    }

    /// MMR over `pool` (trace, score, ..): `next = argmax[score − λ·max cos(m, selected)]`,
    /// ties in pool order.
    fn mmr(&self, pool: &[(Trace, f64, f64, f64)]) -> Vec<usize> {
        let mut max_sim = vec![0.0f64; pool.len()];
        let mut alive = vec![true; pool.len()];
        let mut selected = Vec::new();
        while selected.len() < self.cfg.inject_n {
            let best = (0..pool.len()).filter(|&i| alive[i]).fold(None, |b: Option<(usize, f64)>, i| {
                let val = pool[i].1 - self.cfg.mmr_lambda * max_sim[i];
                if b.is_none_or(|(_, v)| val > v) { Some((i, val)) } else { b }
            });
            let Some((b, _)) = best else { break };
            selected.push(b);
            alive[b] = false;
            for i in (0..pool.len()).filter(|&i| alive[i]) {
                max_sim[i] = max_sim[i].max(dot(&pool[i].0.vector, &pool[b].0.vector));
            }
        }
        selected
    }

    /// Marks the traces of the last recall cited in `reply` (as `《id:…》`) as actually
    /// used: their stability rises to what full activation would have given. Returns
    /// the ids affected.
    pub fn cite(&mut self, reply: &str) -> Result<Vec<String>> {
        let mut cited: Vec<(Trace, f64)> = Vec::new();
        for id in cited_ids(reply) {
            let Some(&c) = self.exposures.get(id) else { continue };
            let Some(t) = self.memory(id)? else { continue };
            if cited.iter().any(|(u, _)| u.id == t.id) {
                continue; // a citation counts once
            }
            let gain = self.cfg.spacing_gain;
            let partial = 1.0 + gain * 0.5 * self.activation(c.cosine) * (1.0 - c.retrievability);
            let full = 1.0 + gain * (1.0 - c.retrievability);
            let s = self.clamp_stability(t.stability * (full / partial));
            cited.push((t, s));
        }
        let writes: Vec<Write<'_>> =
            cited.iter().map(|(t, s)| Write::State { id: &t.id, last_recall: t.last_recall, stability: *s, consolidated: t.consolidated }).collect();
        self.commit(&writes)?;
        // Spent only once the write has landed: after a store error the citation can be retried.
        for (t, _) in &cited {
            self.exposures.remove(&t.id);
        }
        Ok(cited.into_iter().map(|(t, _)| t.id).collect())
    }

    /// Deletes the trace with `id` (never by meaning: that would take unrelated
    /// neighbours with it). Returns the deleted trace.
    pub fn forget(&mut self, id: &str) -> Result<Option<Trace>> {
        // By id alone, so a row the engine cannot use (invalid id, empty text) can still be removed.
        let Some(raw) = self.store.find(View::Id(id), 1)?.pop() else { return Ok(None) };
        self.commit(&[Write::Remove(&raw.id)])?;
        Ok(Some(self.checked(raw.clone()).unwrap_or(raw)))
    }

    // ------------------------------------------------- reactivation candidates

    /// Each (cue, own vector) pair's query vector: the cue's, or the trace's own vector
    /// when it has no cue — and whether that is the query the pair asked for. The
    /// distinct non-empty cues are embedded in one call; when it fails every cued pair
    /// falls back to its own vector, and a missing or wrong-dimension result falls back
    /// for its own pairs only. A fallback search cannot prove "reactivates nothing".
    fn cue_vectors(&mut self, pairs: &[(&str, &[f32])]) -> Vec<(Vec<f32>, bool)> {
        let mut slot: HashMap<&str, usize> = HashMap::new();
        let mut distinct: Vec<&str> = Vec::new();
        for &(cue, _) in pairs {
            if !cue.is_empty() && !slot.contains_key(cue) {
                slot.insert(cue, distinct.len());
                distinct.push(cue);
            }
        }
        let embedded: Vec<Option<Vec<f32>>> = if distinct.is_empty() {
            Vec::new()
        } else {
            match self.embedder.embed_queries(&distinct) {
                Ok(raw) if raw.len() == distinct.len() => raw.iter().map(|v| self.vector(v)).collect(),
                _ => Vec::new(), // offline, or a count that cannot be paired with the cues safely
            }
        };
        pairs
            .iter()
            .map(|&(cue, own)| match slot.get(cue).and_then(|&s| embedded.get(s).cloned().flatten()) {
                Some(q) => (q, true),
                None => (own.to_vec(), cue.is_empty() && !own.is_empty()),
            })
            .collect()
    }

    /// The older traces the cue `q` of `seed` reactivates, strongest first: cos ≥
    /// θ_related and not stated after the seed (facts of the same second — one turn's
    /// saves — are simultaneous), at most `dream_max_members − 1`.
    fn candidates(&self, q: &[f32], seed: &Trace) -> Result<Vec<Trace>> {
        let (hits, _) = self.near(q, self.cfg.theta_related, self.cfg.dream_max_members, Some(seed.created_at))?;
        Ok(hits.into_iter().map(|h| h.trace).filter(|t| t.id != seed.id).take(self.cfg.dream_max_members - 1).collect())
    }

    // ------------------------------------------------------ sleep phase (§5)

    /// The labile traces a dream of `budget` takes: at most 8·budget, first in first out
    /// as a cycle — the ids after the cursor, then (wrapping) those up to it. Pages are
    /// read until enough *usable* rows are found, so rows the engine can never use sit
    /// in the queue without hiding the seeds behind them. FIFO needs no ranking, and the
    /// cycle keeps a seed the LLM cannot judge from starving the queue.
    fn seeds(&self, budget: usize) -> Result<Vec<Trace>> {
        let model_id = self.embedder.model_id();
        let want = SEEDS_PER_BUDGET.saturating_mul(budget);
        let mut out: Vec<Trace> = Vec::new();
        for wrapped in [false, true] {
            if wrapped && self.cursor.is_empty() {
                break; // the first pass already started at the beginning
            }
            let mut after = if wrapped { String::new() } else { self.cursor.clone() };
            'pages: while out.len() < want {
                // Pages never shrink below SEED_PAGE: a queue of unusable rows is read in
                // a few finds, not one per row.
                let asked = (want - out.len()).max(SEED_PAGE);
                let page = self.store.find(View::Labile { model_id, after: &after }, asked)?;
                let (full, last) = (page.len() == asked, page.last().map(|t| t.id.clone()));
                for t in page {
                    if wrapped && t.id > self.cursor {
                        break 'pages;
                    }
                    out.extend(self.checked(t));
                    if out.len() >= want {
                        break 'pages;
                    }
                }
                match last {
                    Some(l) if full && l > after => after = l,
                    _ => break,
                }
            }
        }
        out.truncate(want);
        Ok(out)
    }

    /// The clusters the next [`Engram::dream`] would hand to the LLM (seed first),
    /// without calling it. Clusters may overlap.
    pub fn clusters(&mut self, budget: Option<usize>) -> Result<Vec<Vec<Trace>>> {
        let seeds = self.seeds(budget.unwrap_or(self.cfg.dream_budget))?;
        let qs = self.cue_vectors(&seeds.iter().map(|t| (t.cue.as_str(), t.vector.as_slice())).collect::<Vec<_>>());
        let mut out = Vec::new();
        for (seed, (q, _)) in seeds.into_iter().zip(&qs) {
            let near = self.candidates(q, &seed)?;
            if !near.is_empty() {
                out.push(std::iter::once(seed).chain(near).collect());
            }
        }
        Ok(out)
    }

    /// Offline consolidation — the only place traces are rewritten: re-index a batch of
    /// stale traces, then up to `budget` LLM verdicts (keep / replace, one atomic write
    /// each). Every report is its own undo record ([`Engram::undo`]), so once a verdict is
    /// applied the dream no longer returns `Err`: a later failure (the store, the LLM, the
    /// embedder) is the last report, an [`DreamAction::Error`] that changed nothing, and
    /// ends the dream. Only a failure before any verdict (re-indexing) is an `Err`.
    pub fn dream(&mut self, adjudicator: &mut dyn Adjudicator, budget: Option<usize>) -> Result<Vec<DreamReport>> {
        let now = self.now();
        let mut left = budget.unwrap_or(self.cfg.dream_budget);
        self.reindex(REINDEX_BATCH.saturating_mul(left.max(1)))?;
        let seeds = self.seeds(left)?;
        // A seed's cue and vector never change during a dream, so every cue is embedded
        // up front; candidates are still searched per seed, as each verdict rewrites
        // what the next seed searches.
        let qs = self.cue_vectors(&seeds.iter().map(|t| (t.cue.as_str(), t.vector.as_slice())).collect::<Vec<_>>());
        let mut reports = Vec::new();
        // Seeds that reactivate nothing are settled together in one write at the end: the
        // flag changes no later search, and a lost write only repeats a cheap check.
        let mut settled: Vec<Trace> = Vec::new();
        for (seed, (q, cued)) in seeds.iter().zip(&qs) {
            if left == 0 {
                break;
            }
            self.cursor.clone_from(&seed.id);
            let near = match self.memory(&seed.id).and_then(|s| Ok(match s.filter(|s| !s.consolidated) {
                Some(s) => Some((self.candidates(q, &s)?, s)),
                None => None,
            })) {
                Ok(Some(found)) => found,
                Ok(None) => continue,
                Err(e) => {
                    reports.push(self.failure(std::slice::from_ref(seed), now, &e));
                    break;
                }
            };
            let (near, seed) = near;
            if near.is_empty() {
                if *cued {
                    settled.push(seed);
                } // else a fallback search proves nothing: the seed stays labile, behind the rest
                continue;
            }
            left -= 1;
            let cluster: Vec<Trace> = std::iter::once(seed).chain(near).collect();
            let report = self.adjudicate(adjudicator, cluster, now); // an error leaves the seed labile
            let stored = report.stored_error;
            reports.push(report.report);
            if stored {
                return Ok(reports); // the store is failing: stop, keep what was applied
            }
        }
        // A lost settle write only repeats a cheap check at the next dream: never an error.
        let writes: Vec<Write<'_>> =
            settled.iter().map(|t| Write::State { id: &t.id, last_recall: t.last_recall, stability: t.stability, consolidated: true }).collect();
        if let Err(e) = self.commit(&writes) {
            reports.push(DreamReport { action: DreamAction::Error, before: Vec::new(), absorbed: Vec::new(), after: Vec::new(), error: Some(e.to_string()) });
        }
        Ok(reports)
    }

    /// One trace as a dream request member.
    fn member(&self, t: &Trace, now: f64) -> DreamMember {
        DreamMember {
            id: t.id.clone(),
            text: t.text.clone(),
            local_time: fmt_local(t.created_at, &t.tz),
            timezone: t.tz.clone(),
            retrievability: (self.retrievability(t, now) * 100.0).round() / 100.0,
        }
    }

    /// The report of a cluster that failed before anything was written.
    fn failure(&self, cluster: &[Trace], now: f64, e: &crate::Error) -> DreamReport {
        let before = cluster.iter().map(|t| self.member(t, now)).collect();
        DreamReport { action: DreamAction::Error, before, absorbed: Vec::new(), after: Vec::new(), error: Some(e.to_string()) }
    }

    /// Reverses a dream's verdicts, newest first: a later verdict may have absorbed an
    /// earlier gist, so only the reverse order restores the traces exactly.
    pub fn undo_all(&mut self, reports: &[DreamReport]) -> Result<()> {
        reports.iter().rev().try_for_each(|r| self.undo(r))
    }

    /// Reverses one dream verdict (several, newest first: [`Engram::undo_all`]): a replace removes its gists and restores what it
    /// absorbed (as it was then; later strengthening of the gists is lost), a keep makes
    /// the seed labile again. Restored traces count against the capacity at the next write.
    pub fn undo(&mut self, report: &DreamReport) -> Result<()> {
        match report.action {
            DreamAction::Replace => {
                let mut writes: Vec<Write<'_>> = report.after.iter().map(|g| Write::Remove(&g.id)).collect();
                writes.extend(report.absorbed.iter().map(Write::Insert));
                self.commit(&writes)
            }
            DreamAction::Keep => match report.before.first() {
                Some(seed) => match self.memory(&seed.id)? {
                    Some(t) => self.commit(&[Write::State { id: &t.id, last_recall: t.last_recall, stability: t.stability, consolidated: false }]),
                    None => Ok(()),
                },
                None => Ok(()),
            },
            DreamAction::Error => Ok(()),
        }
    }

    /// One cluster's verdict, applied in one atomic write. Never fails: a failure is an
    /// `Error` report that changed nothing (`stored_error` when the store refused the write).
    fn adjudicate(&mut self, adjudicator: &mut dyn Adjudicator, cluster: Vec<Trace>, now: f64) -> Adjudicated {
        let request = DreamRequest { current_time: self.local_at(now), members: cluster.iter().map(|t| self.member(t, now)).collect() };
        let report = |action, absorbed, after, error| DreamReport { action, before: request.members.clone(), absorbed, after, error };
        // The embedder call inside gists() is as fallible as the LLM: both leave the seed labile.
        let verdict = adjudicator.adjudicate(&request).and_then(|d| self.verdict(d, &cluster, now));
        let (absorbed, gists) = match verdict {
            Ok(v) => v,
            Err(e) => return Adjudicated { report: report(DreamAction::Error, Vec::new(), Vec::new(), Some(e.to_string())), stored_error: false },
        };
        let written = if gists.is_empty() {
            let s = &cluster[0];
            self.commit(&[Write::State { id: &s.id, last_recall: s.last_recall, stability: s.stability, consolidated: true }])
        } else {
            let mut writes: Vec<Write<'_>> = absorbed.iter().map(|t| Write::Remove(&t.id)).collect();
            writes.extend(gists.iter().map(Write::Insert));
            self.commit(&writes)
        };
        let applied = match written {
            Err(e) => report(DreamAction::Error, Vec::new(), Vec::new(), Some(e.to_string())),
            Ok(()) if gists.is_empty() => report(DreamAction::Keep, Vec::new(), Vec::new(), None),
            Ok(()) => report(DreamAction::Replace, absorbed, gists, None),
        };
        Adjudicated { stored_error: applied.error.is_some(), report: applied }
    }

    /// The absorbed traces (the seed and the candidates the verdict names) and their
    /// gists — no gists means keep. A replace whose every text cleans to nothing is an
    /// error, not a keep: silence never settles a seed.
    fn verdict(&mut self, d: DreamDecision, cluster: &[Trace], now: f64) -> Result<(Vec<Trace>, Vec<Trace>)> {
        let mut texts: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for t in &d.memories {
            let c = clean_text(t, self.cfg.text_max);
            if !c.is_empty() && seen.insert(c.clone()) {
                texts.push(c);
            }
        }
        if texts.is_empty() && !d.memories.is_empty() {
            return Err("the verdict's memories are empty after cleaning".into());
        }
        let picked: HashSet<&str> = d.absorbed_ids.iter().map(String::as_str).collect();
        let absorbed: Vec<Trace> =
            cluster.iter().enumerate().filter(|(i, t)| *i == 0 || picked.contains(t.id.as_str())).map(|(_, t)| t.clone()).collect();
        let gists = if texts.len() > absorbed.len() { Vec::new() } else { self.gists(&texts, &absorbed, now)? };
        Ok((absorbed, gists))
    }

    /// Replacement traces, or none when any text is unrelated to every absorbed member
    /// (confabulation guard). A gist inherits the strongest member's stability plus
    /// the *live* evidence of the others (`max S_i + Σ strength_i`, capped) and keeps
    /// the cluster's total strength (`R = Σ strength_i / S`, encoded into
    /// `last_recall`). It was stated when its newest member (the seed) was, so a later
    /// update still reaches it as an older candidate.
    fn gists(&mut self, texts: &[String], absorbed: &[Trace], now: f64) -> Result<Vec<Trace>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let raw = self.embedder.embed_documents(&refs)?;
        if raw.len() != texts.len() {
            return Err("embedder returned the wrong number of gist vectors".into());
        }
        let vectors = raw.iter().map(|v| self.vector(v).ok_or("embedder dimension mismatch")).collect::<std::result::Result<Vec<_>, _>>()?;
        for g in &vectors {
            let best = absorbed.iter().filter(|t| t.vector.len() == g.len()).map(|t| dot(&t.vector, g)).fold(f64::NEG_INFINITY, f64::max);
            if best < self.cfg.gist_min_cosine {
                return Ok(Vec::new());
            }
        }
        let strongest = absorbed.iter().reduce(|a, b| if a.stability >= b.stability { a } else { b }).expect("seed");
        let sum_sr: f64 = absorbed.iter().map(|t| self.strength(t, now)).sum();
        let stability = self.clamp_stability(strongest.stability + sum_sr - self.strength(strongest, now));
        let r = (sum_sr / stability).min(1.0);
        let elapsed = if r > 0.0 { (-r.log2()).min(64.0) } else { 64.0 }; // half-lives already elapsed
        let last_recall = now as i64 - (stability * elapsed).round() as i64;
        let newest = absorbed.iter().reduce(|a, b| if b.created_at > a.created_at { b } else { a }).expect("seed");
        let model_id = self.embedder.model_id().to_owned();
        let mut ids: Vec<String> = Vec::with_capacity(texts.len());
        for text in texts {
            let id = self.new_id((now * 1000.0) as i64, text, &ids)?;
            ids.push(id);
        }
        Ok(texts
            .iter()
            .zip(vectors)
            .zip(ids)
            .map(|((text, vector), id)| Trace {
                id,
                text: text.clone(),
                created_at: newest.created_at,
                tz: newest.tz.clone(),
                last_recall,
                stability,
                consolidated: true,
                cue: newest.cue.clone(),
                model_id: model_id.clone(),
                vector,
            })
            .collect())
    }
}

/// The ids quoted as `《id:…》` in `text`, in order.
fn cited_ids(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("《id:") {
        let after = &rest[start + "《id:".len()..];
        match after.find(['》', '《']) {
            Some(0) => rest = after,
            // A new opener before any closer: the earlier one was stray.
            Some(end) if after[end..].starts_with('《') => rest = &after[end..],
            Some(end) => {
                out.push(&after[..end]);
                rest = &after[end + '》'.len_utf8()..];
            }
            None => break,
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::cited_ids;

    #[test]
    fn citations_are_parsed_in_order() {
        assert_eq!(cited_ids("a《id:X1》 b 《id:》《id:Y2》《id:Z"), ["X1", "Y2"]);
        assert_eq!(cited_ids("《id:《id:X》"), ["X"]);
    }
}
