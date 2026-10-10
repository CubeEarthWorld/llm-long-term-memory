//! The ENGRAM v2.1 engine (SPEC §3–§5): traces with a forgetting curve, the wake-phase
//! verbs remember / recall / cite / forget, and the sleep-phase dream.

use crate::dream::{Adjudicator, DreamDecision, DreamMember, DreamRequest};
use crate::store::{Store, Write};
use crate::text::{clean_text, cues};
use crate::time::{fmt_local, system_clock, tz_field, Clock};
use crate::vector::{normalized, scores};
use crate::id::trace_id;
use crate::{Config, Embedder, Result, Trace};
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, RandomState};

/// Seeds examined per LLM call: a dream costs O(budget · capacity · dim).
const SEEDS_PER_BUDGET: usize = 8;
/// Traces re-embedded per embedder call.
const REINDEX_BATCH: usize = 64;

/// What [`Engram::remember`] did.
#[derive(Clone, Debug, PartialEq)]
pub enum Remembered {
    /// A new trace. `cosine` is how strongly its cue reactivated the nearest older
    /// trace (0 when none did); `evicted` is the last trace forgotten to make room.
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

/// One adjudicated cluster.
#[derive(Clone, Debug, PartialEq)]
pub struct DreamReport {
    pub action: DreamAction,
    /// The cluster as the adjudicator saw it, seed first.
    pub before: Vec<DreamMember>,
    /// Ids removed by a replace (the seed and the named candidates).
    pub absorbed: Vec<String>,
    /// The gists a replace inserted.
    pub after: Vec<Trace>,
    pub error: Option<String>,
}

/// Counts for dashboards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stats {
    pub records: usize,
    pub labile: usize,
    pub unindexed: usize,
}

/// What one recall cue left behind for [`Engram::cite`].
#[derive(Clone, Copy, Debug)]
struct Exposure {
    cosine: f64,
    retrievability: f64,
}

/// A long-term memory (ENGRAM v2.1). Every trace lives in RAM in insertion order
/// (which breaks every tie); the [`Store`] persists each operation as one atomic
/// write, applied to RAM only after the store accepted it.
///
/// No operation makes more than one linear pass over the store: every trace carries an
/// insertion sequence number that never changes, so the id and text indexes point at
/// sequence numbers (found by binary search) and survive removals without rebuilding,
/// and the labile / unindexed counts are kept up to date instead of recounted.
pub struct Engram {
    cfg: Config,
    store: Box<dyn Store>,
    embedder: Box<dyn Embedder>,
    clock: Clock,
    timezone: String,
    traces: Vec<Trace>,
    /// Insertion sequence number of each trace (strictly increasing, parallel to `traces`).
    seqs: Vec<u64>,
    next_seq: u64,
    by_id: HashMap<String, u64>,
    /// Text hash → sequence number (exact duplicates are rehearsed, so texts are unique).
    by_text: HashMap<u64, u64>,
    /// Inserts whose text hash was already taken: text lookups then fall back to a scan.
    text_collisions: usize,
    hasher: RandomState,
    labile: usize,
    unindexed: usize,
    write_times: Vec<f64>,
    exposures: HashMap<String, Exposure>,
}

/// Puts and removes of one operation, applied store-first.
#[derive(Default)]
struct Change {
    put: Vec<Trace>,
    remove: Vec<String>,
}

impl Engram {
    /// Builds an engine on the wall clock and loads the store (see [`Engram::initialize`]).
    pub fn open(store: Box<dyn Store>, embedder: Box<dyn Embedder>, cfg: Config, timezone: &str) -> Result<Self> {
        Self::with_clock(store, embedder, cfg, timezone, system_clock())
    }

    /// [`Engram::open`] with an injected clock.
    pub fn with_clock(store: Box<dyn Store>, embedder: Box<dyn Embedder>, cfg: Config, timezone: &str, clock: Clock) -> Result<Self> {
        cfg.validate()?;
        let mut me = Self {
            cfg,
            store,
            embedder,
            clock,
            timezone: timezone.to_owned(),
            traces: Vec::new(),
            seqs: Vec::new(),
            next_seq: 0,
            by_id: HashMap::new(),
            by_text: HashMap::new(),
            text_collisions: 0,
            hasher: RandomState::new(),
            labile: 0,
            unindexed: 0,
            write_times: Vec::new(),
            exposures: HashMap::new(),
        };
        me.initialize()?;
        Ok(me)
    }

    // ------------------------------------------------------------------ clock

    /// Current Unix time, seconds.
    pub fn now(&self) -> f64 {
        (self.clock)()
    }

    /// Replaces the time source (seed replays run on a virtual clock).
    pub fn set_clock(&mut self, clock: Clock) {
        self.clock = clock;
    }

    /// Current local time, e.g. `'2026-06-12 09:30 +09:00'`.
    pub fn now_local(&self) -> String {
        self.local_at(self.now())
    }

    fn local_at(&self, now: f64) -> String {
        fmt_local(now as i64, &tz_field(&self.timezone, now))
    }

    // ------------------------------------------------- the forgetting curve (§3)

    fn elapsed(t: &Trace, now: f64) -> f64 {
        (now - t.last_recall as f64).max(0.0)
    }

    /// `R = 2^(−max(0, now − last_recall) / stability)`.
    pub fn retrievability(&self, t: &Trace, now: f64) -> f64 {
        2f64.powf(-Self::elapsed(t, now) / t.stability)
    }

    /// `S·R` — proportional to the trace's total remaining retrievability.
    pub fn strength(&self, t: &Trace, now: f64) -> f64 {
        t.stability * self.retrievability(t, now)
    }

    /// `log2(strength)`: the order of [`Engram::strength`] without the underflow that
    /// would tie every dormant trace at 0.
    fn rank(t: &Trace, now: f64) -> f64 {
        t.stability.ln() / std::f64::consts::LN_2 - Self::elapsed(t, now) / t.stability
    }

    /// NaN would survive min/max and never rank lowest: an immortal trace.
    fn clamp_stability(&self, s: f64) -> f64 {
        (if s.is_nan() { 1.0 } else { s }).max(1.0).min(self.cfg.max_stability)
    }

    /// Cue activation: the cosine rescaled above the model's baseline.
    fn activation(&self, cos: f64) -> f64 {
        ((cos - self.cfg.cosine_floor) / (1.0 - self.cfg.cosine_floor)).max(0.0)
    }

    /// The retrieval update: stability grows with the spacing achieved (`1 − R`) and
    /// the cue activation `a`; `last_recall` moves to now.
    fn retrieved(&self, t: &Trace, now: f64, a: f64) -> Trace {
        let s = t.stability * (1.0 + self.cfg.spacing_gain * a * (1.0 - self.retrievability(t, now)));
        Trace { stability: self.clamp_stability(s), last_recall: now as i64, ..t.clone() }
    }

    // -------------------------------------------------------------- lifecycle

    /// Loads every trace, clamps out-of-range values (SPEC §2) and re-embeds traces
    /// indexed under another model.
    pub fn initialize(&mut self) -> Result<()> {
        let now = self.now() as i64;
        self.clear_ram();
        let mut fixed = Change::default();
        for raw in self.store.load_all()? {
            // A row an older version or a hand edit wrote is skipped, not fatal.
            if raw.id.is_empty() || raw.text.is_empty() || self.by_id.contains_key(&raw.id) {
                continue;
            }
            let (stability, last_recall, created_at) = (self.clamp_stability(raw.stability), raw.last_recall.min(now), raw.created_at.min(now));
            let changed = stability != raw.stability || last_recall != raw.last_recall || created_at != raw.created_at;
            let t = Trace { stability, last_recall, created_at, ..raw };
            if changed {
                fixed.put.push(t.clone());
            }
            self.push(t);
        }
        self.apply(fixed)?;
        self.reindex()
    }

    /// Erases everything.
    pub fn reset(&mut self) -> Result<()> {
        self.store.clear()?;
        self.clear_ram();
        self.write_times.clear();
        self.exposures.clear();
        Ok(())
    }

    /// Every trace, in insertion order.
    pub fn memories(&self) -> &[Trace] {
        &self.traces
    }

    /// The trace with `id`.
    pub fn memory(&self, id: &str) -> Option<&Trace> {
        self.index_of(id).map(|i| &self.traces[i])
    }

    /// The active parameters.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// The store (read-only: every write goes through the engine).
    pub fn store(&self) -> &dyn Store {
        self.store.as_ref()
    }

    /// The embedder.
    pub fn embedder(&self) -> &dyn Embedder {
        self.embedder.as_ref()
    }

    /// Counts of all, labile and unindexed traces.
    pub fn stats(&self) -> Stats {
        Stats { records: self.traces.len(), labile: self.labile, unindexed: self.unindexed }
    }

    /// Megabytes held by vectors.
    pub fn vector_mb(&self) -> f64 {
        self.traces.iter().map(|t| t.vector.len() * 4).sum::<usize>() as f64 / (1024.0 * 1024.0)
    }

    fn clear_ram(&mut self) {
        self.traces.clear();
        self.seqs.clear();
        self.by_id.clear();
        self.by_text.clear();
        (self.text_collisions, self.labile, self.unindexed) = (0, 0, 0);
    }

    fn text_key(&self, text: &str) -> u64 {
        self.hasher.hash_one(text)
    }

    /// Position of the trace with sequence number `seq`.
    fn slot(&self, seq: u64) -> Option<usize> {
        self.seqs.binary_search(&seq).ok()
    }

    fn index_of(&self, id: &str) -> Option<usize> {
        self.by_id.get(id).and_then(|&s| self.slot(s))
    }

    fn index_of_text(&self, text: &str) -> Option<usize> {
        let hit = self.by_text.get(&self.text_key(text)).and_then(|&s| self.slot(s)).filter(|&i| self.traces[i].text == text);
        if hit.is_some() || self.text_collisions == 0 {
            return hit;
        }
        self.traces.iter().position(|t| t.text == text)
    }

    /// A trace's share of the (labile, unindexed) counts.
    fn shares(&self, t: &Trace) -> (usize, usize) {
        (usize::from(!t.consolidated), usize::from(!self.is_indexed(t)))
    }

    fn count(&mut self, (labile, unindexed): (usize, usize), add: bool) {
        if add {
            self.labile += labile;
            self.unindexed += unindexed;
        } else {
            self.labile -= labile;
            self.unindexed -= unindexed;
        }
    }

    /// Appends a trace under a fresh sequence number.
    fn push(&mut self, t: Trace) {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.count(self.shares(&t), true);
        self.by_id.insert(t.id.clone(), seq);
        let key = self.text_key(&t.text);
        if let Some(old) = self.by_text.insert(key, seq)
            && self.slot(old).is_some()
        {
            self.by_text.insert(key, old); // keep the first; lookups of this one scan
            self.text_collisions += 1;
        }
        self.seqs.push(seq);
        self.traces.push(t);
    }

    /// Writes `ch` to the store as one commit, then to RAM: puts replace in place
    /// (O(log N) each) or append; removes keep the order of the rest (one compaction).
    fn apply(&mut self, ch: Change) -> Result<()> {
        if ch.put.is_empty() && ch.remove.is_empty() {
            return Ok(());
        }
        let writes: Vec<Write<'_>> =
            ch.put.iter().map(Write::Put).chain(ch.remove.iter().map(|id| Write::Remove(id))).collect();
        self.store.write(&writes)?;
        for t in ch.put {
            match self.index_of(&t.id) {
                Some(i) => {
                    self.count(self.shares(&t), true);
                    let old = std::mem::replace(&mut self.traces[i], t);
                    self.count(self.shares(&old), false);
                }
                None => self.push(t),
            }
        }
        if !ch.remove.is_empty() {
            let mut gone: HashSet<u64> = HashSet::with_capacity(ch.remove.len());
            for id in &ch.remove {
                let Some(i) = self.index_of(id) else { continue };
                let seq = self.seqs[i];
                self.by_id.remove(id);
                let key = self.text_key(&self.traces[i].text);
                if self.by_text.get(&key) == Some(&seq) {
                    self.by_text.remove(&key);
                }
                self.count(self.shares(&self.traces[i]), false);
                gone.insert(seq);
            }
            let seqs = &self.seqs;
            let mut i = 0;
            self.traces.retain(|_| {
                i += 1;
                !gone.contains(&seqs[i - 1])
            });
            self.seqs.retain(|s| !gone.contains(s));
        }
        Ok(())
    }

    /// One model id means one dimension (SPEC §7): a vector of any other length is
    /// stale and re-embedded, so the index never mixes dimensions.
    fn is_indexed(&self, t: &Trace) -> bool {
        t.model_id == self.embedder.model_id() && t.vector.len() == self.embedder.dimension()
    }

    /// A freshly embedded vector, normalised — or `None` when the embedder contradicted
    /// its own dimension (handled like an offline embedder: indexed on a later attempt).
    fn vector(&self, v: &[f32]) -> Option<Vec<f32>> {
        (v.len() == self.embedder.dimension()).then(|| normalized(v))
    }

    /// Re-embeds traces whose vector belongs to another model, in batches; an offline
    /// embedder leaves the rest stale (invisible to search) until the next attempt.
    fn reindex(&mut self) -> Result<()> {
        let stale: Vec<usize> = (0..self.traces.len()).filter(|&i| !self.is_indexed(&self.traces[i])).collect();
        for batch in stale.chunks(REINDEX_BATCH) {
            let texts: Vec<&str> = batch.iter().map(|&i| self.traces[i].text.as_str()).collect();
            let Ok(vectors) = self.embedder.embed_documents(&texts) else { return Ok(()) };
            let model_id = self.embedder.model_id().to_owned();
            let put = batch
                .iter()
                .zip(&vectors)
                .filter_map(|(&i, v)| {
                    let vector = self.vector(v)?;
                    Some(Trace { model_id: model_id.clone(), vector, ..self.traces[i].clone() })
                })
                .collect();
            self.apply(Change { put, remove: Vec::new() })?;
        }
        Ok(())
    }

    /// Indices of the indexed traces, in insertion order.
    fn indexed(&self) -> Vec<usize> {
        (0..self.traces.len()).filter(|&i| self.is_indexed(&self.traces[i])).collect()
    }

    /// Row-major cosines of `rows` (trace indices) against `queries`.
    fn scan(&self, rows: &[usize], queries: &[Vec<f32>]) -> Vec<f64> {
        let r: Vec<&[f32]> = rows.iter().map(|&i| self.traces[i].vector.as_slice()).collect();
        let q: Vec<&[f32]> = queries.iter().map(Vec::as_slice).collect();
        scores(&r, &q)
    }

    // ------------------------------------------------------- wake phase (§4)

    /// Stores one self-contained proposition. Exact duplicates are rehearsed; anything
    /// else is inserted, never overwritten. A write whose `cue` reactivates an older
    /// trace is born labile for the dream to adjudicate. Over capacity, the weakest
    /// trace past the grace period is forgotten. `salience` (0–10) scales S0.
    pub fn remember(&mut self, text: &str, salience: f64, cue: &str) -> Result<Remembered> {
        let now = self.now();
        let text = clean_text(text, self.cfg.text_max);
        let cue = clean_text(cue, self.cfg.text_max);
        if text.is_empty() {
            return Ok(Remembered::Rejected);
        }
        if let Some(i) = self.index_of_text(&text) {
            let r = self.retrieved(&self.traces[i], now, 1.0);
            self.apply(Change { put: vec![r.clone()], remove: Vec::new() })?;
            return Ok(Remembered::Reinforced(r));
        }
        let day_ago = now - 86_400.0;
        self.write_times.retain(|&t| t >= day_ago);
        if self.write_times.len() >= self.cfg.writes_per_day {
            return Ok(Remembered::RateLimited { text });
        }
        self.write_times.push(now);
        let v = self.embedder.embed_documents(&[&text]).ok().and_then(|vs| vs.into_iter().next()).and_then(|v| self.vector(&v));
        let candidates = match &v {
            Some(v) => {
                let q = self.cue_vectors(&[(cue.as_str(), v.as_slice())]).remove(0);
                self.candidates(&q, None)
            }
            None => Vec::new(),
        };
        let sal = if salience.is_nan() { 1.0 } else { salience.clamp(0.0, 10.0) };
        let consolidated = v.is_some() && candidates.is_empty() && self.unindexed == 0;
        let trace = Trace {
            id: trace_id((now * 1000.0) as i64, &text),
            text,
            created_at: now as i64,
            tz: tz_field(&self.timezone, now),
            last_recall: now as i64,
            stability: self.clamp_stability(self.cfg.initial_stability * sal),
            consolidated,
            cue,
            model_id: if v.is_some() { self.embedder.model_id().to_owned() } else { String::new() },
            vector: v.unwrap_or_default(),
        };
        let victims = self.victims(now, Some(&trace));
        let evicted = victims.last().map(|&i| if i == self.traces.len() { trace.clone() } else { self.traces[i].clone() });
        let remove = victims.iter().map(|&i| if i == self.traces.len() { trace.id.clone() } else { self.traces[i].id.clone() }).collect();
        self.apply(Change { put: vec![trace.clone()], remove })?;
        Ok(Remembered::Inserted { trace, cosine: candidates.first().map_or(0.0, |c| c.0), evicted })
    }

    /// The traces to forget while over capacity, in eviction order, as indices into
    /// `traces` (`traces.len()` stands for `extra`, a trace about to be appended).
    /// Traces inside the grace period are protected unless they exceed a tenth of the
    /// capacity (a flood evicts its own members) or nothing older exists. Ranks cannot
    /// change while evicting, so each age group is ordered once (weakest first, ties
    /// in insertion order).
    fn victims(&self, now: f64, extra: Option<&Trace>) -> Vec<usize> {
        let all: Vec<&Trace> = self.traces.iter().chain(extra).collect();
        let excess = all.len().saturating_sub(self.cfg.capacity);
        if excess == 0 {
            return Vec::new();
        }
        let (mut young, mut old) = (Vec::<(f64, usize)>::new(), Vec::<(f64, usize)>::new());
        for (i, t) in all.iter().enumerate() {
            let age = now - t.created_at as f64;
            let group = if age >= 0.0 && age < self.cfg.grace_period { &mut young } else { &mut old };
            group.push((Self::rank(t, now), i));
        }
        // At most `excess` victims come from either group: select those (linear), then
        // order only them — never the whole store.
        let young_total = young.len();
        for group in [&mut young, &mut old] {
            first_k(group, excess, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        }
        let (mut y, mut o) = (0, 0);
        (0..excess)
            .map(|_| {
                if o == old.len() || (young_total - y > self.cfg.capacity / 10 && young[y].0 < old[o].0) {
                    y += 1;
                    young[y - 1].1
                } else {
                    o += 1;
                    old[o - 1].1
                }
            })
            .collect()
    }

    /// Retrieves up to `inject_n` traces relevant to `query` and packs them for prompt
    /// injection. Injection is exposure, not use: each injected trace gets the retrieval
    /// update at half its cue activation; [`Engram::cite`] completes it.
    pub fn recall(&mut self, query: &str) -> Result<Recall> {
        let now = self.now();
        self.exposures.clear(); // a recall always ends the previous citation window
        let recall = self.select(query, now);
        let put = recall.recalled.iter().map(|c| self.retrieved(&c.trace, now, 0.5 * self.activation(c.cosine))).collect();
        self.apply(Change { put, remove: Vec::new() })?;
        for c in &recall.recalled {
            self.exposures.insert(c.trace.id.clone(), Exposure { cosine: c.cosine, retrievability: c.retrievability });
        }
        Ok(recall)
    }

    /// What [`Engram::recall`] would inject, without strengthening anything (for
    /// measurement and previews).
    pub fn peek(&mut self, query: &str) -> Recall {
        let now = self.now();
        self.select(query, now)
    }

    fn select(&mut self, query: &str, now: f64) -> Recall {
        let parts = cues(query, self.cfg.max_cues);
        let rows = self.indexed();
        if parts.is_empty() || rows.is_empty() {
            return Recall::default(); // nothing to score: do not pay for an embedding
        }
        let refs: Vec<&str> = parts.iter().map(String::as_str).collect();
        let qs: Option<Vec<Vec<f32>>> = match self.embedder.embed_queries(&refs) {
            Ok(raw) if raw.len() == parts.len() => raw.iter().map(|v| self.vector(v)).collect(),
            _ => None, // offline, or it contradicted itself: nothing to cue with
        };
        let Some(qs) = qs else { return Recall::default() };
        let k = qs.len();
        let sims = self.scan(&rows, &qs);
        let mut scored = Vec::with_capacity(rows.len());
        let mut top = 0.0f64;
        for (r, &i) in rows.iter().enumerate() {
            let cos = sims[r * k..(r + 1) * k].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let rv = self.retrievability(&self.traces[i], now);
            let score = self.activation(cos) * (self.cfg.alpha + (1.0 - self.cfg.alpha) * rv);
            top = top.max(score);
            scored.push((i, score, cos, rv));
        }
        let floor = self.cfg.min_score.max(self.cfg.relative_score * top);
        let pool: Vec<_> = scored.into_iter().filter(|c| c.1 >= floor).collect(); // insertion order: MMR ties keep it
        let mut out = Recall::default();
        let mut used = 0;
        for j in self.mmr(&pool) {
            let (i, score, cosine, retrievability) = pool[j];
            let t = &self.traces[i];
            let line = format!("[{} {}] {}　《id:{}》\n", t.created_at, t.tz, t.text, t.id);
            let cost = line.chars().count();
            if !out.pack_text.is_empty() && used + cost > self.cfg.budget_chars {
                continue;
            }
            out.pack_text.push_str(&line);
            used += cost;
            out.recalled.push(Recalled { trace: t.clone(), score, cosine, retrievability });
        }
        out
    }

    /// MMR over `pool` (trace index, score, ..): `next = argmax[score − λ·max cos(m, selected)]`.
    fn mmr(&self, pool: &[(usize, f64, f64, f64)]) -> Vec<usize> {
        let mut max_sim = vec![0.0f64; pool.len()];
        let mut alive = vec![true; pool.len()];
        let mut selected = Vec::new();
        while selected.len() < self.cfg.inject_n {
            let (mut best, mut best_val) = (None, f64::NEG_INFINITY);
            for (i, p) in pool.iter().enumerate() {
                let val = p.1 - self.cfg.mmr_lambda * max_sim[i];
                if alive[i] && val > best_val {
                    (best, best_val) = (Some(i), val);
                }
            }
            let Some(b) = best else { break };
            selected.push(b);
            alive[b] = false;
            if selected.len() == self.cfg.inject_n {
                break;
            }
            let rest: Vec<usize> = (0..pool.len()).filter(|&i| alive[i]).collect();
            if rest.is_empty() {
                break;
            }
            let rows: Vec<usize> = rest.iter().map(|&i| pool[i].0).collect();
            let sims = self.scan(&rows, std::slice::from_ref(&self.traces[pool[b].0].vector));
            for (r, &i) in rest.iter().enumerate() {
                max_sim[i] = max_sim[i].max(sims[r]);
            }
        }
        selected
    }

    /// Marks the traces of the last recall cited in `reply` (as `《id:…》`) as actually
    /// used: their stability rises to what full activation would have given. Returns
    /// the ids affected.
    pub fn cite(&mut self, reply: &str) -> Result<Vec<String>> {
        let mut put: Vec<Trace> = Vec::new();
        for id in cited_ids(reply) {
            let Some(c) = self.exposures.remove(id) else { continue };
            let Some(t) = self.memory(id) else { continue };
            let gain = self.cfg.spacing_gain;
            let partial = 1.0 + gain * 0.5 * self.activation(c.cosine) * (1.0 - c.retrievability);
            let full = 1.0 + gain * (1.0 - c.retrievability);
            put.push(Trace { stability: self.clamp_stability(t.stability * full / partial), ..t.clone() });
        }
        let ids = put.iter().map(|t| t.id.clone()).collect();
        self.apply(Change { put, remove: Vec::new() })?;
        Ok(ids)
    }

    /// Deletes the trace with `id` (never by meaning: that would take unrelated
    /// neighbours with it). Returns the deleted trace.
    pub fn forget(&mut self, id: &str) -> Result<Option<Trace>> {
        let Some(t) = self.memory(id).cloned() else { return Ok(None) };
        self.apply(Change { put: Vec::new(), remove: vec![t.id.clone()] })?;
        Ok(Some(t))
    }

    // ------------------------------------------------- reactivation candidates

    /// Each (cue, own vector) pair's query vector: the cue's, or the trace's own vector
    /// when it has no cue. The distinct non-empty cues are embedded in one call; when it
    /// fails every pair falls back to its own vector, and a missing or wrong-dimension
    /// result falls back for its own pairs only.
    fn cue_vectors(&mut self, pairs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
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
                Ok(raw) => raw.iter().take(distinct.len()).map(|v| self.vector(v)).collect(),
                Err(_) => Vec::new(), // offline: every cue falls back to its trace's vector
            }
        };
        pairs
            .iter()
            .map(|&(cue, own)| slot.get(cue).and_then(|&s| embedded.get(s).cloned().flatten()).unwrap_or_else(|| own.to_vec()))
            .collect()
    }

    /// Traces the cue `q` reactivates, as (cos, trace index); see [`Engram::near`].
    fn candidates(&self, q: &[f32], seed: Option<&Trace>) -> Vec<(f64, usize)> {
        let rows = self.indexed();
        let sims = self.scan(&rows, &[q.to_vec()]);
        self.near(&rows, &sims, 1, 0, seed)
    }

    /// The traces column `col` of a scan reactivates, strongest first: cos ≥ θ_related
    /// and not written after the seed, so evidence only ever rewrites its own past.
    /// Ties break on insertion order (ULID tails are random). At most
    /// `dream_max_members − 1`.
    fn near(&self, rows: &[usize], sims: &[f64], stride: usize, col: usize, seed: Option<&Trace>) -> Vec<(f64, usize)> {
        let mut near: Vec<(f64, usize, usize)> = rows
            .iter()
            .enumerate()
            .filter(|&(_, &i)| seed.is_none_or(|s| self.traces[i].id != s.id && self.traces[i].created_at <= s.created_at))
            .map(|(r, &i)| (sims[r * stride + col], r, i))
            .filter(|c| c.0 >= self.cfg.theta_related)
            .collect();
        first_k(&mut near, self.cfg.dream_max_members - 1, |a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        near.into_iter().map(|(c, _, i)| (c, i)).collect()
    }

    // ------------------------------------------------------ sleep phase (§5)

    /// The labile traces a dream of `budget` scans: first in, first out, at most
    /// 8·budget. FIFO needs no ranking and never starves a seed.
    fn seeds(&self, budget: usize) -> Vec<usize> {
        self.indexed().into_iter().filter(|&i| !self.traces[i].consolidated).take(SEEDS_PER_BUDGET.saturating_mul(budget)).collect()
    }

    /// The clusters the next [`Engram::dream`] would hand to the LLM (seed first),
    /// without calling it. Clusters may overlap. Nothing changes while previewing, so
    /// every cue is embedded in one call and all seeds are scored in one scan.
    pub fn clusters(&mut self, budget: Option<usize>) -> Vec<Vec<Trace>> {
        let seeds = self.seeds(budget.unwrap_or(self.cfg.dream_budget));
        if seeds.is_empty() {
            return Vec::new();
        }
        let pairs: Vec<(String, Vec<f32>)> = seeds.iter().map(|&i| (self.traces[i].cue.clone(), self.traces[i].vector.clone())).collect();
        let qs = self.cue_vectors(&pairs.iter().map(|(c, v)| (c.as_str(), v.as_slice())).collect::<Vec<_>>());
        let rows = self.indexed();
        let sims = self.scan(&rows, &qs);
        seeds
            .iter()
            .enumerate()
            .filter_map(|(j, &s)| {
                let seed = &self.traces[s];
                let mut cluster = vec![seed.clone()];
                cluster.extend(self.near(&rows, &sims, qs.len(), j, Some(seed)).into_iter().map(|(_, i)| self.traces[i].clone()));
                (cluster.len() >= 2).then_some(cluster)
            })
            .collect()
    }

    /// Offline consolidation — the only place traces are rewritten: backup → re-index
    /// stale traces → up to `budget` LLM verdicts (keep / replace, one transaction each).
    pub fn dream(&mut self, adjudicator: &mut dyn Adjudicator, budget: Option<usize>) -> Result<Vec<DreamReport>> {
        let now = self.now();
        self.store.backup();
        self.reindex()?;
        let mut left = budget.unwrap_or(self.cfg.dream_budget);
        let seeds: Vec<(String, String, Vec<f32>)> =
            self.seeds(left).into_iter().map(|i| (self.traces[i].id.clone(), self.traces[i].cue.clone(), self.traces[i].vector.clone())).collect();
        // A seed's cue and vector never change during a dream, so every cue is embedded
        // up front; candidates are still searched per seed, as each verdict rewrites
        // what the next seed searches.
        let qs = self.cue_vectors(&seeds.iter().map(|(_, c, v)| (c.as_str(), v.as_slice())).collect::<Vec<_>>());
        let mut reports = Vec::new();
        for ((id, _, _), q) in seeds.iter().zip(&qs) {
            let Some(seed) = self.memory(id).filter(|s| !s.consolidated).cloned() else { continue };
            if left == 0 {
                break;
            }
            let mut cluster = vec![seed.clone()];
            cluster.extend(self.candidates(q, Some(&seed)).into_iter().map(|(_, i)| self.traces[i].clone()));
            if cluster.len() < 2 {
                self.apply(Change { put: vec![Trace { consolidated: true, ..seed }], remove: Vec::new() })?;
                continue;
            }
            left -= 1;
            reports.push(self.adjudicate(adjudicator, cluster, now)?); // an error leaves the seed labile
        }
        Ok(reports)
    }

    fn adjudicate(&mut self, adjudicator: &mut dyn Adjudicator, cluster: Vec<Trace>, now: f64) -> Result<DreamReport> {
        let request = DreamRequest {
            current_time: self.local_at(now),
            members: cluster
                .iter()
                .map(|t| DreamMember {
                    id: t.id.clone(),
                    text: t.text.clone(),
                    local_time: fmt_local(t.created_at, &t.tz),
                    timezone: t.tz.clone(),
                    retrievability: (self.retrievability(t, now) * 100.0).round() / 100.0,
                })
                .collect(),
        };
        let report = |action, absorbed, after, error| DreamReport { action, before: request.members.clone(), absorbed, after, error };
        // The embedder call inside gists() is as fallible as the LLM: both leave the seed labile.
        let verdict = adjudicator.adjudicate(&request).and_then(|d| self.verdict(d, &cluster, now));
        let (absorbed, gists) = match verdict {
            Ok(v) => v,
            Err(e) => return Ok(report(DreamAction::Error, Vec::new(), Vec::new(), Some(e.to_string()))),
        };
        if gists.is_empty() {
            let seed = Trace { consolidated: true, ..cluster[0].clone() };
            self.apply(Change { put: vec![seed], remove: Vec::new() })?;
            return Ok(report(DreamAction::Keep, Vec::new(), Vec::new(), None));
        }
        let ids: Vec<String> = absorbed.iter().map(|t| t.id.clone()).collect();
        self.apply(Change { put: gists.clone(), remove: ids.clone() })?;
        Ok(report(DreamAction::Replace, ids, gists, None))
    }

    /// The absorbed traces (the seed and the candidates the verdict names) and their
    /// gists — no gists means keep.
    fn verdict(&mut self, d: DreamDecision, cluster: &[Trace], now: f64) -> Result<(Vec<Trace>, Vec<Trace>)> {
        let mut texts: Vec<String> = Vec::new();
        for t in &d.memories {
            let c = clean_text(t, self.cfg.text_max);
            if !c.is_empty() && !texts.contains(&c) {
                texts.push(c);
            }
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
        let mv: Vec<&[f32]> = absorbed.iter().map(|t| t.vector.as_slice()).collect();
        let gv: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
        let sims = scores(&mv, &gv);
        for j in 0..vectors.len() {
            let best = (0..absorbed.len()).fold(f64::NEG_INFINITY, |b, i| b.max(sims[i * vectors.len() + j]));
            if best < self.cfg.gist_min_cosine {
                return Ok(Vec::new());
            }
        }
        let strongest = absorbed.iter().reduce(|a, b| if a.stability >= b.stability { a } else { b }).expect("seed");
        let sum_sr: f64 = absorbed.iter().map(|t| self.strength(t, now)).sum();
        let stability = self.clamp_stability(strongest.stability + sum_sr - self.strength(strongest, now));
        let r = (sum_sr / stability).min(1.0);
        let elapsed = if r > 0.0 { (-r.ln() / std::f64::consts::LN_2).min(64.0) } else { 64.0 }; // half-lives already elapsed
        let last_recall = now as i64 - (stability * elapsed).round() as i64;
        let newest = absorbed.iter().reduce(|a, b| if b.created_at > a.created_at { b } else { a }).expect("seed");
        let model_id = self.embedder.model_id().to_owned();
        Ok(texts
            .iter()
            .zip(vectors)
            .map(|(text, vector)| Trace {
                id: trace_id((now * 1000.0) as i64, text),
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

/// Keeps the first `k` elements of `v` under `order`, sorted: a linear selection, then a
/// sort of just those `k` (O(n + k log k) rather than sorting all n).
fn first_k<T>(v: &mut Vec<T>, k: usize, order: impl Fn(&T, &T) -> std::cmp::Ordering) {
    if k == 0 {
        v.clear();
        return;
    }
    if v.len() > k {
        v.select_nth_unstable_by(k - 1, &order);
        v.truncate(k);
    }
    v.sort_unstable_by(order);
}

/// The ids quoted as `《id:…》` in `text`, in order.
fn cited_ids(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("《id:") {
        let after = &rest[start + "《id:".len()..];
        match after.find('》') {
            Some(0) => rest = after,
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
    }
}
