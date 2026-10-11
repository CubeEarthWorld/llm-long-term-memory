//! Round-2 audit, lane D: whole-system, model-based, stateful testing.
//!
//! A seeded random operation-sequence harness drives one engine (bundled `Store` behind a
//! failure-injecting wrapper, a failure-injecting `Embedder`, a failure-injecting
//! `Adjudicator`, a virtual clock that jumps forward, steps backward and idles for years)
//! and, after EVERY step, reads the store directly and checks the SPEC section 7
//! invariants plus a shadow model of the rows. Operations are data (`Op`), so a failing
//! run is shrunk by delta debugging to a minimal replayable `(seed, kept op indices)`.
//!
//! * `fuzz_*`            broad sweeps; fail on any invariant kind not listed in `KNOWN`.
//! * `r2d_*`             one named test per confirmed defect (fail until the defect is fixed).
//! * `AUDIT2_SEEDS=n`    seeds per sweep group (default 40); `AUDIT2_STEPS=n` max steps.
//!
//! Run: `CARGO_TARGET_DIR=C:/t/r2d cargo test --test audit2_d -j 3`

mod support;

use engram::{
    cues, Adjudicator, Config, DreamAction, DreamDecision, DreamReport, DreamRequest, Embedder, Engram, ManualClock, Recall, Remembered,
    Storage, Store, Trace, View, Write,
};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, MutexGuard};
use support::{FakeEmbedder, DAY, START};

type Rows = BTreeMap<String, Trace>;
type ER<T> = engram::Result<T>;

/// Invariant kinds that have their own named `r2d_*` test; the sweeps record them
/// (soft) and carry on instead of failing.
const KNOWN: &[&str] = &["I-err-but-applied", "I7-seed-skipped-future-stamp", "I7-seed-unsettled-future-stamp", "I7-gist-created-at-clamped"];

const MODELS: [(&str, usize); 3] = [("m1/token", 32), ("m2/token", 24), ("m3/token", 40)];
const TZ: &str = "Asia/Tokyo;+09:00";

fn lk<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// ------------------------------------------------------------------------- rng

#[derive(Clone, Debug)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % (n.max(1) as u64)) as usize
    }
    fn chance(&mut self, permille: u32) -> bool {
        self.next() % 1000 < u64::from(permille)
    }
    fn f(&mut self) -> f64 {
        (self.next() >> 11) as f64 / 9_007_199_254_740_992.0
    }
    fn pick<T: Clone>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len())].clone()
    }
}

// ------------------------------------------------------------- fault controls

#[derive(Clone, Debug, Default)]
struct FaultCfg {
    emb_p: u32,
    emb_kinds: Vec<u8>,
    st_write: u32,
    st_find: u32,
    st_near: u32,
    st_clear: u32,
    id_p: u32,
}

struct Ctl {
    f: FaultCfg,
    model: usize,
    emb_rng: Rng,
    st_rng: Rng,
    id_rng: Rng,
    id_counter: u64,
    live: Vec<String>,
    suppress: bool,
}

type Shared = Arc<Mutex<Ctl>>;

struct FuzzEmbedder {
    ctl: Shared,
    embs: Vec<FakeEmbedder>,
}

impl FuzzEmbedder {
    fn run(&mut self, texts: &[&str]) -> ER<Vec<Vec<f32>>> {
        let (m, fault) = {
            let mut c = lk(&self.ctl);
            let p = c.f.emb_p;
            let hit = !c.suppress && p > 0 && c.emb_rng.chance(p);
            let kind = if hit && !c.f.emb_kinds.is_empty() {
                let n = c.f.emb_kinds.len();
                let i = c.emb_rng.below(n);
                Some(c.f.emb_kinds[i])
            } else {
                None
            };
            (c.model, kind)
        };
        let dim = MODELS[m].1;
        let mut out: Vec<Vec<f32>> = texts.iter().map(|t| self.embs[m].embed(t)).collect();
        match fault {
            None => {}
            Some(0) => return Err("embedder offline".into()),
            Some(1) => {
                out.pop();
            }
            Some(2) => out.push(vec![1.0; dim]),
            Some(3) => out.iter_mut().for_each(|v| v.truncate(dim - 1)),
            Some(4) => out.iter_mut().for_each(|v| v.iter_mut().for_each(|x| *x = 0.0)),
            Some(5) => out.iter_mut().for_each(|v| v[0] = f32::NAN),
            Some(6) => out.iter_mut().for_each(|v| v[1] = f32::INFINITY),
            Some(7) => out.iter_mut().for_each(|v| v.iter_mut().for_each(|x| *x *= 1e30)),
            Some(8) => out.iter_mut().for_each(|v| v.iter_mut().for_each(|x| *x *= 1e-30)),
            Some(9) => {
                if let Some(v) = out.last_mut() {
                    v.iter_mut().for_each(|x| *x = 0.0);
                }
            }
            Some(_) => {
                if let Some(v) = out.first_mut() {
                    v.push(0.5);
                }
            }
        }
        Ok(out)
    }
}

impl Embedder for FuzzEmbedder {
    fn model_id(&self) -> &str {
        MODELS[lk(&self.ctl).model].0
    }
    fn dimension(&self) -> usize {
        MODELS[lk(&self.ctl).model].1
    }
    fn embed_documents(&mut self, texts: &[&str]) -> ER<Vec<Vec<f32>>> {
        self.run(texts)
    }
    fn embed_queries(&mut self, texts: &[&str]) -> ER<Vec<Vec<f32>>> {
        self.run(texts)
    }
}

/// The bundled store behind a wrapper that fails reads/writes at random. A failed write
/// changes nothing (atomic), like a rolled-back transaction.
struct Flaky {
    inner: Arc<Mutex<Store>>,
    ctl: Shared,
    writes: Arc<Mutex<usize>>,
}

impl Flaky {
    fn fail(&self, which: u8) -> bool {
        let mut c = lk(&self.ctl);
        if c.suppress {
            return false;
        }
        let p = match which {
            0 => c.f.st_write,
            1 => c.f.st_find,
            2 => c.f.st_near,
            _ => c.f.st_clear,
        };
        p > 0 && c.st_rng.chance(p)
    }
}

impl Storage for Flaky {
    fn find(&self, view: View<'_>, limit: usize) -> ER<Vec<Trace>> {
        if self.fail(1) {
            return Err("injected find failure".into());
        }
        lk(&self.inner).find(view, limit)
    }
    fn count(&self, view: View<'_>) -> ER<usize> {
        if self.fail(1) {
            return Err("injected count failure".into());
        }
        lk(&self.inner).count(view)
    }
    fn near(&self, q: &[f32], model_id: &str, min_cos: f64, k: usize, stated_by: Option<i64>) -> ER<Vec<engram::Hit>> {
        if self.fail(2) {
            return Err("injected near failure".into());
        }
        lk(&self.inner).near(q, model_id, min_cos, k, stated_by)
    }
    fn write(&mut self, writes: &[Write<'_>]) -> ER<()> {
        if self.fail(0) {
            return Err("injected write failure".into());
        }
        *lk(&self.writes) += 1;
        lk(&self.inner).write(writes)
    }
    fn clear(&mut self) -> ER<()> {
        if self.fail(3) {
            return Err("injected clear failure".into());
        }
        lk(&self.inner).clear()
    }
}

// ------------------------------------------------------------------ adjudicator

#[derive(Clone, Debug)]
struct AdjPlan {
    seed: u64,
    p_err: u32,
    p_keep: u32,
}

struct Call {
    members: Vec<String>,
    decision: Option<DreamDecision>,
}

struct Adj {
    rng: Rng,
    plan: AdjPlan,
    log: Arc<Mutex<Vec<Call>>>,
    known_ids: Vec<String>,
}

impl Adj {
    fn decide(&mut self, req: &DreamRequest) -> Option<DreamDecision> {
        if self.rng.chance(self.plan.p_err) {
            return None;
        }
        if self.rng.chance(self.plan.p_keep) {
            return Some(DreamDecision::keep());
        }
        let seed = &req.members[0];
        let cand = &req.members[1..];
        let named: Vec<&engram::DreamMember> = cand.iter().filter(|_| self.rng.chance(600)).collect();
        let ids: Vec<String> = named.iter().map(|m| m.id.clone()).collect();
        let t0 = seed.text.clone();
        let joined = std::iter::once(&seed.text).chain(named.iter().map(|m| &m.text)).cloned().collect::<Vec<_>>().join(" / ");
        let d = |memories: Vec<String>, absorbed_ids: Vec<String>| DreamDecision { memories, absorbed_ids };
        Some(match self.rng.below(12) {
            0..=2 => d(vec![joined], ids),
            3 => d(std::iter::once(t0).chain(named.iter().map(|m| m.text.clone())).collect(), ids),
            4 => d(vec![format!("{t0} revised")], Vec::new()),
            5 => d(vec!["zzzz qqqq wwww".into()], ids),
            6 => d(vec![String::new(), "  ".into(), "《》\u{7}".into()], ids),
            7 => d(vec![t0.repeat(400 / t0.chars().count().max(1) + 2)], ids),
            8 => d(vec![t0.clone(), t0.clone(), t0], ids),
            9 => {
                let mut bad = vec![String::new(), "nope".into(), seed.id.clone(), format!("{}\n", seed.id)];
                if !self.known_ids.is_empty() {
                    let i = self.rng.below(self.known_ids.len());
                    bad.push(self.known_ids[i].clone());
                }
                bad.extend(ids);
                d(vec![format!("{t0} v2")], bad)
            }
            10 => d(vec![format!("《{t0}》\u{7}\u{202E}{t0}\n\t x")], ids),
            _ => d(Vec::new(), ids), // keep with ids
        })
    }
}

impl Adjudicator for Adj {
    fn adjudicate(&mut self, req: &DreamRequest) -> ER<DreamDecision> {
        let d = self.decide(req);
        lk(&self.log).push(Call { members: req.members.iter().map(|m| m.id.clone()).collect(), decision: d.clone() });
        d.ok_or_else(|| "llm down".into())
    }
}

// -------------------------------------------------------------------- operations

#[derive(Clone, Debug)]
enum Op {
    Remember { text: String, sal: f64, cue: String },
    Recall(String),
    Peek(String),
    PeekRecall(String),
    Cite(u64),
    Forget(u64),
    Dream { budget: Option<usize>, plan: AdjPlan, undo: bool },
    UndoStale(u64),
    Reindex(usize),
    Clusters(Option<usize>),
    Reset,
    Switch(usize),
    Clock(f64),
    Faults(FaultCfg),
}

#[derive(Clone)]
struct Scenario {
    seed: u64,
    origin: f64,
    cfg: Config,
    counter_ids: bool,
    ops: Vec<Op>,
}

const TOPICS: [[&str; 6]; 6] = [
    ["tokyo", "osaka", "kyoto", "nagoya", "sapporo", "fukuoka"],
    ["coffee", "tea", "milk", "juice", "water", "beer"],
    ["cat", "dog", "bird", "fish", "horse", "fox"],
    ["work", "office", "meeting", "report", "salary", "boss"],
    ["train", "bus", "bike", "car", "walk", "taxi"],
    ["piano", "guitar", "violin", "drum", "flute", "song"],
];
const FILL: [&str; 6] = ["user", "lives", "likes", "moved", "owns", "visited"];

fn topic_words(r: &mut Rng, topic: usize, n: usize) -> Vec<String> {
    let mut w: Vec<String> = (0..n).map(|_| TOPICS[topic][r.below(6)].to_string()).collect();
    for _ in 0..r.below(3) {
        w.push(FILL[r.below(6)].to_string());
    }
    w
}

fn gen_text(r: &mut Rng, pool: &[String]) -> String {
    if !pool.is_empty() && r.chance(150) {
        return pool[r.below(pool.len())].clone();
    }
    let topic = r.below(6);
    let n = 2 + r.below(4);
    let mut w = topic_words(r, topic, n);
    if r.chance(150) {
        let t2 = r.below(6);
        w.extend(topic_words(r, t2, 2));
    }
    let mut s = w.join(" ");
    match r.below(40) {
        0 => s = format!("《{s}》"),
        1 => s = format!("{s}\u{7}\u{202E} \n\t  {s}"),
        2 => s = s.repeat(300 / s.len().max(1) + 2),
        3 => s = "   ".into(),
        4 => s = String::new(),
        5 => s = "《》\u{0}\u{85}".into(),
        6 => s = format!("東京に住んでいる。{s}。大阪へ引っ越した。"),
        7 => s = format!("{s}。").repeat(30),
        8 => s = format!("\u{FEFF}{s}\u{3000}\u{2028}"),
        _ => {}
    }
    s
}

fn gen_cfg(r: &mut Rng) -> Config {
    let smax = r.pick(&[1.0, 30.0 * DAY, 30.0 * DAY, 3650.0 * DAY, 3650.0 * DAY, 36500.0 * DAY]);
    let init = r.pick(&[3600.0, DAY, 10.0 * DAY]).min(smax);
    let c = Config {
        capacity: 3 + r.below(38),
        theta_related: r.pick(&[0.2, 0.5, 0.6, 0.75, 0.9]),
        cosine_floor: r.pick(&[0.0, 0.0, 0.3]),
        text_max: r.pick(&[8, 40, 170, 170, 170]),
        min_score: r.pick(&[0.1, 0.1, 0.0, 0.5]),
        alpha: r.pick(&[0.35, 0.0, 1.0]),
        mmr_lambda: r.pick(&[0.3, 0.0, 2.0]),
        gist_min_cosine: r.pick(&[0.3, 0.5, 0.9, -1.0]),
        grace_period: r.pick(&[0.0, 3600.0, 3.0 * DAY]),
        writes_per_day: r.pick(&[5, 20, 100, 1000]),
        inject_n: 1 + r.below(6),
        budget_chars: r.pick(&[60, 150, 400, 1024]),
        max_cues: 1 + r.below(8),
        dream_budget: 1 + r.below(6),
        dream_max_members: 2 + r.below(7),
        initial_stability: init,
        max_stability: smax,
        spacing_gain: r.pick(&[1.0, 3.0, 10.0]),
        relative_score: r.pick(&[0.6, 0.3, 0.0, 1.0]),
        ..Config::default()
    };
    c.validate().expect("generated config");
    c
}

fn gen_scenario(seed: u64, max_steps: usize) -> Scenario {
    let mut r = Rng::new(seed);
    let cfg = gen_cfg(&mut r);
    let counter_ids = seed % 2 == 1;
    let steps = 200 + r.below(max_steps.saturating_sub(200).max(1));
    let mut ops = Vec::with_capacity(steps);
    let mut pool: Vec<String> = Vec::new();
    for _ in 0..steps {
        let roll = r.below(1000);
        let op = match roll {
            0..=339 => {
                let text = gen_text(&mut r, &pool);
                pool.push(text.clone());
                if pool.len() > 40 {
                    pool.remove(0);
                }
                let sal = match r.below(12) {
                    0 => f64::NAN,
                    1 => -3.0,
                    2 => 11.0,
                    3 => f64::INFINITY,
                    4 => 0.0,
                    5 => 5.0,
                    _ => 1.0,
                };
                let cue = match r.below(20) {
                    0..=8 => String::new(),
                    9..=16 => {
                        let t = r.below(6);
                        topic_words(&mut r, t, 2).join(" ")
                    }
                    17 => "《cue》\u{7}".into(),
                    18 => "x".repeat(400),
                    _ => gen_text(&mut r, &pool),
                };
                Op::Remember { text, sal, cue }
            }
            340..=479 => Op::Recall(gen_query(&mut r)),
            480..=519 => Op::Peek(gen_query(&mut r)),
            520..=549 => Op::PeekRecall(gen_query(&mut r)),
            550..=629 => Op::Cite(r.next()),
            630..=679 => Op::Forget(r.next()),
            680..=779 => Op::Dream {
                budget: r.pick(&[None, None, Some(0), Some(1), Some(2), Some(7)]),
                plan: AdjPlan { seed: r.next(), p_err: r.pick(&[0, 0, 100, 400, 1000]), p_keep: r.pick(&[0, 100, 300]) },
                undo: false,
            },
            780..=809 => Op::Dream {
                budget: r.pick(&[None, Some(1), Some(3)]),
                plan: AdjPlan { seed: r.next(), p_err: r.pick(&[0, 100]), p_keep: r.pick(&[0, 100]) },
                undo: true,
            },
            810..=819 => Op::UndoStale(r.next()),
            820..=839 => Op::Reindex(r.pick(&[0, 1, 3, 100])),
            840..=859 => Op::Clusters(r.pick(&[None, Some(0), Some(1), Some(4)])),
            860..=862 => Op::Reset,
            863..=870 => Op::Switch(r.below(3)),
            871..=939 => Op::Clock(gen_clock(&mut r)),
            _ => Op::Faults(gen_faults(&mut r)),
        };
        ops.push(op);
    }
    Scenario { seed, origin: START, cfg, counter_ids, ops }
}

fn gen_query(r: &mut Rng) -> String {
    let n = 1 + r.below(4);
    let mut parts = Vec::new();
    for _ in 0..n {
        let topic = r.below(6);
        let n = 1 + r.below(3);
        parts.push(topic_words(r, topic, n).join(" "));
    }
    let sep = r.pick(&[". ", "\n", "。", "? "]);
    let mut q = parts.join(sep);
    match r.below(30) {
        0 => q = String::new(),
        1 => q = "   \n  ".into(),
        2 => q = format!("{q}《id:x》"),
        _ => {}
    }
    q
}

fn gen_clock(r: &mut Rng) -> f64 {
    gen_clock_raw(r).round()
}

fn gen_clock_raw(r: &mut Rng) -> f64 {
    match r.below(1000) {
        0..=449 => 1.0 + r.f() * 600.0,
        450..=699 => 3600.0 + r.f() * DAY,
        700..=829 => DAY + r.f() * 60.0 * DAY,
        830..=879 => (1.0 + r.f() * 30.0) * 365.0 * DAY,
        880..=959 => -(1.0 + r.f() * 1e6),
        960..=989 => 0.0,
        990..=996 => -(r.f() * 40.0 * DAY),
        _ => 300.0 * 365.0 * DAY * (1.0 + r.f() * 5.0),
    }
}

fn gen_faults(r: &mut Rng) -> FaultCfg {
    let p = |r: &mut Rng, xs: &[u32]| r.pick(xs);
    let mut kinds: Vec<u8> = (0..=10u8).filter(|_| r.chance(400)).collect();
    if kinds.is_empty() {
        kinds.push(0);
    }
    FaultCfg {
        emb_p: p(r, &[0, 0, 0, 100, 300, 1000]),
        emb_kinds: kinds,
        st_write: p(r, &[0, 0, 0, 50, 200, 500]),
        st_find: p(r, &[0, 0, 0, 0, 30, 100]),
        st_near: p(r, &[0, 0, 0, 0, 30, 100]),
        st_clear: p(r, &[0, 0, 300]),
        id_p: p(r, &[0, 0, 0, 150, 1000]),
    }
}

// --------------------------------------------------------------- text replicas

fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1C}'..='\u{1F}').contains(&c) || c == '\u{FEFF}'
}
fn is_dropped(c: char) -> bool {
    matches!(c, '《' | '》' | '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}') || (c.is_control() && !is_space(c))
}
fn has_content(s: &str) -> bool {
    s.chars().any(|c| !is_dropped(c) && !is_space(c))
}
fn shorten(text: &str, max: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max {
        return text.to_owned();
    }
    let cut = &chars[..max];
    for sep in ['。', '．', '.', '、', ' '] {
        if let Some(idx) = cut.iter().rposition(|&c| c == sep)
            && idx as f64 > max as f64 * 0.5
        {
            return cut[..=idx].iter().collect();
        }
    }
    cut.iter().collect()
}
fn clean(text: &str, max: usize) -> String {
    let mut out = String::new();
    let mut space = false;
    for c in text.chars().filter(|&c| !is_dropped(c)) {
        if is_space(c) {
            space = true;
        } else {
            if space && !out.is_empty() {
                out.push(' ');
            }
            space = false;
            out.push(c);
        }
    }
    shorten(&out, max).trim().to_owned()
}
fn id_ok(id: &str) -> bool {
    !id.is_empty() && !id.chars().any(|c| is_dropped(c) || c.is_control() || (is_space(c) && c != ' '))
}
fn text_ok(s: &str, max: usize) -> bool {
    !s.is_empty() && s.chars().count() <= max && s == s.trim() && !s.chars().any(|c| is_dropped(c) || (is_space(c) && c != ' ')) && !s.contains("  ")
}
fn cue_ok(s: &str, max: usize) -> bool {
    s.chars().count() <= max && !s.chars().any(|c| is_dropped(c) || (is_space(c) && c != ' '))
}

/// The ids quoted as `《id:…》`: each opener takes the text up to the first closer when no
/// opener intervenes (stated differently from the engine's scanner).
fn parse_cites(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(p) = text[from..].find("《id:") {
        let start = from + p + "《id:".len();
        from = start;
        let rest = &text[start..];
        if let Some(e) = rest.find(['》', '《'])
            && rest[e..].starts_with('》')
            && e > 0
        {
            out.push(rest[..e].to_string());
        }
    }
    out
}

fn dot64(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum()
}

// ------------------------------------------------------------------ the runner

#[derive(Clone, Debug)]
struct Fail {
    step: usize,
    kind: String,
    msg: String,
}

type Res = std::result::Result<(), Fail>;

macro_rules! ck {
    ($s:expr, $c:expr, $k:expr, $($a:tt)+) => {
        if !$c {
            $s.flag($k, format!($($a)+))?;
        }
    };
}

struct Run {
    cfg: Config,
    eng: Engram,
    inner: Arc<Mutex<Store>>,
    ctl: Shared,
    clock: ManualClock,
    store_writes: Arc<Mutex<usize>>,
    oracle: Vec<FakeEmbedder>,
    rows: Rows,
    /// id -> (cosine, retrievability) of the last recall's injected traces
    exposures: BTreeMap<String, (f64, f64)>,
    overshoot: bool,
    ever: HashSet<String>,
    dead: HashSet<String>,
    maxclock: f64,
    max_id: String,
    origin: f64,
    lo: Vec<f64>,
    hi: Vec<f64>,
    history: Vec<Vec<DreamReport>>,
    ords: HashMap<String, usize>,
    log: Vec<String>,
    step: usize,
    soft: BTreeMap<String, String>,
    strict: bool,
}

struct Outcome {
    fail: Option<Fail>,
    log: Vec<String>,
    soft: BTreeMap<String, String>,
}

fn build_run(scn: &Scenario) -> Run {
    let seed = scn.seed;
    let ctl: Shared = Arc::new(Mutex::new(Ctl {
        f: FaultCfg::default(),
        model: 0,
        emb_rng: Rng::new(seed ^ 0xE1),
        st_rng: Rng::new(seed ^ 0x57),
        id_rng: Rng::new(seed ^ 0x1D),
        id_counter: 0,
        live: Vec::new(),
        suppress: false,
    }));
    let inner = Arc::new(Mutex::new(Store::in_memory().expect("store")));
    let writes = Arc::new(Mutex::new(0));
    let flaky = Flaky { inner: inner.clone(), ctl: ctl.clone(), writes: writes.clone() };
    let embs = MODELS.iter().map(|&(id, d)| FakeEmbedder::with(d, id)).collect();
    let clock = ManualClock::new(scn.origin);
    let mut eng = Engram::with_clock(Box::new(flaky), Box::new(FuzzEmbedder { ctl: ctl.clone(), embs }), scn.cfg.clone(), "Asia/Tokyo", clock.clock()).expect("engine");
    if scn.counter_ids {
        let c = ctl.clone();
        let store = inner.clone();
        eng.set_ids(Box::new(move |_, _| {
            let mut c = lk(&c);
            let p = c.f.id_p;
            if p > 0 && c.id_rng.chance(p) {
                return match c.id_rng.below(5) {
                    0 => String::new(),
                    1 => "bad\nid".into(),
                    2 => "a《b》".into(),
                    3 => "tab\tid".into(),
                    _ => {
                        let live: Vec<String> = lk(&store).find(View::All, usize::MAX).map(|v| v.into_iter().map(|t| t.id).collect()).unwrap_or_default();
                        if live.is_empty() { String::new() } else { live[c.id_rng.below(live.len())].clone() }
                    }
                };
            }
            c.id_counter += 1;
            format!("G{:012}", c.id_counter)
        }));
    }
    Run {
        cfg: scn.cfg.clone(),
        eng,
        inner,
        ctl,
        clock,
        store_writes: writes,
        oracle: MODELS.iter().map(|&(id, d)| FakeEmbedder::with(d, id)).collect(),
        rows: Rows::new(),
        exposures: BTreeMap::new(),
        overshoot: false,
        ever: HashSet::new(),
        dead: HashSet::new(),
        maxclock: scn.origin,
        origin: scn.origin,
        max_id: String::new(),
        lo: Vec::new(),
        hi: Vec::new(),
        history: Vec::new(),
        ords: HashMap::new(),
        log: Vec::new(),
        step: 0,
        soft: BTreeMap::new(),
        strict: false,
    }
}

fn panic_msg(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "panic".into())
}

impl Run {
    fn fail(&self, kind: &str, msg: String) -> Fail {
        Fail { step: self.step, kind: kind.into(), msg }
    }

    /// A violated invariant: fatal, unless its kind is known (recorded, run continues).
    fn flag(&mut self, kind: &str, msg: String) -> Res {
        if !self.strict && KNOWN.contains(&kind) {
            self.soft.entry(kind.into()).or_insert_with(|| format!("step {}: {msg}", self.step));
            Ok(())
        } else {
            Err(self.fail(kind, msg))
        }
    }

    fn call<T>(&mut self, f: impl FnOnce(&mut Engram) -> T) -> std::result::Result<T, Fail> {
        match catch_unwind(AssertUnwindSafe(|| f(&mut self.eng))) {
            Ok(v) => Ok(v),
            Err(p) => Err(self.fail("panic", panic_msg(p))),
        }
    }

    fn read_rows(&self) -> Rows {
        lk(&self.inner).find(View::All, usize::MAX).expect("direct read").into_iter().map(|t| (t.id.clone(), t)).collect()
    }

    fn now(&self) -> f64 {
        self.clock.get()
    }

    fn model(&self) -> (&'static str, usize) {
        MODELS[lk(&self.ctl).model]
    }

    fn faultless(&self) -> (bool, bool) {
        let c = lk(&self.ctl);
        (c.suppress || c.f.emb_p == 0, c.suppress || (c.f.st_write == 0 && c.f.st_find == 0 && c.f.st_near == 0))
    }

    fn set_suppress(&self, on: bool) {
        lk(&self.ctl).suppress = on;
    }

    fn clamp_s(&self, s: f64) -> f64 {
        (if s.is_nan() { 1.0 } else { s }).max(1.0).min(self.cfg.max_stability)
    }

    /// A stored row as the engine reads it.
    fn chk(&self, t: &Trace, now: f64) -> Trace {
        let mut c = t.clone();
        c.stability = self.clamp_s(t.stability);
        let _ = now; // stored times are not rewritten on read
        c
    }

    fn retr(&self, t: &Trace, now: f64) -> f64 {
        let t = self.chk(t, now);
        (-(now - t.last_recall as f64).max(0.0) / t.stability).exp2()
    }

    fn activation(&self, cos: f64) -> f64 {
        ((cos.min(1.0) - self.cfg.cosine_floor) / (1.0 - self.cfg.cosine_floor)).max(0.0)
    }

    fn ord(&mut self, id: &str) -> usize {
        let n = self.ords.len();
        *self.ords.entry(id.to_string()).or_insert(n)
    }

    fn note(&mut self, s: String) {
        self.log.push(format!("{}:{s}", self.step));
    }

    // ------------------------------------------------------------ state checks

    fn check_rows(&mut self, rows: &Rows) -> Res {
        let (cur_model, cur_dim) = self.model();
        let _ = cur_model;
        let _ = cur_dim;
        ck!(self, self.overshoot || rows.len() <= self.cfg.capacity, "I3-count-over-capacity", "{} rows > capacity {}", rows.len(), self.cfg.capacity);
        let mut model_dim: HashMap<&str, usize> = HashMap::new();
        let now = self.now();
        let max = self.cfg.text_max;
        for t in rows.values() {
            ck!(self, id_ok(&t.id), "I5-id-invalid", "id {:?}", t.id);
            ck!(self, text_ok(&t.text, max), "I5-text-invalid", "text {:?}", t.text);
            ck!(self, cue_ok(&t.cue, max), "I5-cue-invalid", "cue {:?}", t.cue);
            ck!(self, t.tz == TZ, "I5-tz", "tz {:?}", t.tz);
            ck!(self, t.stability.is_finite() && t.stability >= 1.0 && t.stability <= self.cfg.max_stability, "I1-stability-range", "{} stability {}", t.id, t.stability);
            ck!(self, (t.created_at as f64) <= self.maxclock && (t.last_recall as f64) <= self.maxclock, "I2-future-stamp", "{} created {} last_recall {} > max clock {}", t.id, t.created_at, t.last_recall, self.maxclock);
            let r = self.eng.retrievability(t, now);
            ck!(self, (0.0..=1.0).contains(&r), "I1-R-range", "R {r} for {}", t.id);
            if t.vector.is_empty() {
                ck!(self, t.model_id.is_empty(), "I2-model-without-vector", "{} model {:?} no vector", t.id, t.model_id);
            } else {
                ck!(self, !t.model_id.is_empty(), "I2-vector-without-model", "{} vector without model", t.id);
                ck!(self, t.vector.iter().all(|x| x.is_finite()), "I2-vector-nonfinite", "{}", t.id);
                let n2 = dot64(&t.vector, &t.vector);
                ck!(self, (n2 - 1.0).abs() <= 1e-3, "I2-vector-not-unit", "{} |v|^2 = {n2}", t.id);
                let d = *model_dim.entry(t.model_id.as_str()).or_insert(t.vector.len());
                ck!(self, d == t.vector.len(), "I2-model-mixed-dims", "model {} has dims {d} and {}", t.model_id, t.vector.len());
                let want = MODELS.iter().find(|m| m.0 == t.model_id).map(|m| m.1);
                ck!(self, want == Some(t.vector.len()), "I2-dim-mismatch", "{} model {} dim {}", t.id, t.model_id, t.vector.len());
            }
        }
        Ok(())
    }

    /// After every step: invariants on the rows, then the shadow model moves on.
    fn finish(&mut self, next: Rows, restored: &HashSet<String>) -> Res {
        self.maxclock = self.maxclock.max(self.now());
        self.check_rows(&next)?;
        let mut new_ids: Vec<String> = Vec::new();
        for id in next.keys() {
            if !self.rows.contains_key(id) {
                if !restored.contains(id) {
                    new_ids.push(id.clone());
                }
                let again = self.ever.contains(id);
                ck!(self, !again || restored.contains(id), "I4-id-resurrected", "id {id} reappeared (dead={})", self.dead.contains(id));
                self.ever.insert(id.clone());
                self.dead.remove(id);
            }
        }
        for id in &new_ids {
            ck!(self, *id > self.max_id, "I2-id-order", "new id {id} does not sort after {:?}", self.max_id);
        }
        if let Some(m) = new_ids.iter().max() {
            self.max_id = self.max_id.clone().max(m.clone());
        }
        for id in self.rows.keys() {
            if !next.contains_key(id) {
                self.dead.insert(id.clone());
            }
        }
        self.exposures.retain(|id, _| next.contains_key(id));
        // the engine's own counters agree with the store (faults off for the audit read)
        self.set_suppress(true);
        let stats = self.eng.stats();
        self.set_suppress(false);
        let (cur, _) = self.model();
        match stats {
            Ok(s) => {
                let labile = next.values().filter(|t| !t.consolidated && t.model_id == cur).count();
                ck!(self, s.records == next.len() && s.labile == labile, "I-stats", "stats {s:?} vs rows {} labile {labile}", next.len());
            }
            Err(e) => ck!(self, false, "I-stats-err", "stats failed with faults off: {e}"),
        }
        if next.len() <= self.cfg.capacity {
            self.overshoot = false;
        }
        // digest for the determinism comparison (ids by order of first sight)
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |x: u64| h = (h ^ x).wrapping_mul(0x100_0000_01B3);
        let rows: Vec<Trace> = next.values().cloned().collect();
        for t in &rows {
            mix(self.ord(&t.id) as u64);
            for b in t.text.bytes() {
                mix(u64::from(b));
            }
            mix(t.stability.to_bits());
            mix((t.last_recall - self.origin as i64) as u64);
            mix((t.created_at - self.origin as i64) as u64);
            mix(u64::from(t.consolidated));
            mix(t.vector.len() as u64);
            for b in t.model_id.bytes() {
                mix(u64::from(b));
            }
            mix(t.vector.first().map_or(0, |x| u64::from(x.to_bits())));
        }
        let live: Vec<String> = next.keys().cloned().collect();
        lk(&self.ctl).live = live;
        self.note(format!("digest {h:016x} n={}", rows.len()));
        self.rows = next;
        Ok(())
    }

    // -------------------------------------------------------------- operations

    fn unchanged(&mut self, next: &Rows, kind: &str, what: &str) -> Res {
        if *next != self.rows {
            let d = diff(&self.rows, next);
            self.flag(kind, format!("{what}: store changed (removed {:?} added {:?} changed {:?})", d.0, d.1, d.2))?;
        }
        Ok(())
    }

    fn only_state_changes(&mut self, prev: &Rows, next: &Rows, allowed: &BTreeSet<String>, what: &str) -> Res {
        let d = diff(prev, next);
        ck!(self, d.0.is_empty() && d.1.is_empty(), "I-ids-changed", "{what}: ids removed {:?} added {:?}", d.0, d.1);
        for id in &d.2 {
            ck!(self, allowed.contains(id), "I-unnamed-rewrite", "{what}: row {id} changed but was not named");
            let (a, b) = (&prev[id], &next[id]);
            ck!(self, a.text == b.text && a.vector == b.vector && a.model_id == b.model_id && a.cue == b.cue && a.created_at == b.created_at && a.tz == b.tz,
                "I-immutable-fields", "{what}: {id} text/vector/cue/created changed");
        }
        Ok(())
    }

    fn op_remember(&mut self, text: &str, sal: f64, cue: &str) -> Res {
        let now = self.now();
        let prev = self.rows.clone();
        let (emb_ok, st_ok) = self.faultless();
        let (cur, dim) = self.model();
        let w = self.cfg.writes_per_day;
        let in_range = |v: &[f64]| v.iter().filter(|&&t| t >= now - 86_400.0 && t <= now + 86_400.0).count();
        let lo_n = in_range(&self.lo);
        let cleaned = clean(text, self.cfg.text_max);
        let writes_before = *lk(&self.store_writes);
        let r = self.call(|e| e.remember(text, sal, cue))?;
        let next = self.read_rows();
        let _ = writes_before;
        match r {
            Err(e) => {
                self.note(format!("remember Err({e})"));
                // possibly pushed into the engine's rate window before failing
                self.lo.retain(|&t| t >= now - 86_400.0 && t <= now + 86_400.0);
                self.hi.push(now);
                if next != prev {
                    self.flag("I-err-but-applied", format!("remember returned Err({e}) but the store changed: {:?}", diff(&prev, &next)))?;
                }
            }
            Ok(Remembered::Rejected) => {
                self.note("remember Rejected".into());
                ck!(self, !has_content(text), "I5-rejected-with-content", "text {text:?} has content");
                self.unchanged(&next, "I-rejected-wrote", "Rejected")?;
            }
            Ok(Remembered::RateLimited { text: t }) => {
                self.note("remember RateLimited".into());
                self.lo.retain(|&x| x >= now - 86_400.0 && x <= now + 86_400.0);
                self.hi.retain(|&x| x >= now - 86_400.0 && x <= now + 86_400.0);
                ck!(self, t == cleaned, "I-ratelimit-text", "RateLimited text {t:?} vs cleaned {cleaned:?}");
                ck!(self, self.hi.len() >= w, "I-ratelimit-too-early", "RateLimited with at most {} writes in the window (limit {w})", self.hi.len());
                self.unchanged(&next, "I-ratelimit-wrote", "RateLimited")?;
            }
            Ok(Remembered::Reinforced(t)) => {
                self.note("remember Reinforced".into());
                let target = prev.values().find(|p| p.text == cleaned).cloned();
                let Some(target) = target else {
                    return self.flag("I-reinforced-nothing", format!("Reinforced {} but no row has text {cleaned:?}", t.id));
                };
                ck!(self, t.id == target.id, "I-reinforce-target", "reinforced {} but first row with the text is {}", t.id, target.id);
                let c = self.chk(&target, now);
                let want = self.clamp_s(c.stability * (1.0 + self.cfg.spacing_gain * (1.0 - self.retr(&target, now))));
                let mut allowed = BTreeSet::new();
                allowed.insert(target.id.clone());
                self.only_state_changes(&prev, &next, &allowed, "Reinforced")?;
                let n = &next[&target.id];
                ck!(self, n.last_recall == now as i64 && n.consolidated == target.consolidated, "I-reinforce-state", "last_recall {} (now {}), consolidated {} -> {}", n.last_recall, now as i64, target.consolidated, n.consolidated);
                ck!(self, (n.stability - want).abs() <= 1e-9 * want.max(1.0), "I-reinforce-stability", "stability {} want {want}", n.stability);
                ck!(self, n.stability >= c.stability - 1e-9, "I-reinforce-lowered", "stability {} -> {}", c.stability, n.stability);
            }
            Ok(Remembered::Inserted { trace, cosine, evicted }) => {
                self.note(format!("remember Inserted ev={}", evicted.as_ref().map_or("-".into(), |e| self.ords.get(&e.id).map_or("?".to_string(), |o| o.to_string()))));
                ck!(self, lo_n < w, "I-ratelimit-ignored", "inserted although {lo_n} writes already in the window (limit {w})");
                self.lo.retain(|&x| x >= now - 86_400.0 && x <= now + 86_400.0);
                self.hi.retain(|&x| x >= now - 86_400.0 && x <= now + 86_400.0);
                self.lo.push(now);
                self.hi.push(now);
                ck!(self, has_content(text) && !cleaned.is_empty(), "I5-inserted-empty", "inserted {text:?}");
                ck!(self, !prev.values().any(|p| p.text == cleaned), "I-duplicate-inserted", "exact text {cleaned:?} was stored already");
                ck!(self, trace.text == cleaned && trace.cue == clean(cue, self.cfg.text_max), "I5-inserted-text", "trace text/cue {:?}/{:?}", trace.text, trace.cue);
                let sal = if sal.is_nan() { 1.0 } else { sal.clamp(0.0, 10.0) };
                let want_s = self.clamp_s(self.cfg.initial_stability * sal);
                ck!(self, trace.stability == want_s && trace.created_at == now as i64 && trace.last_recall == now as i64, "I-new-trace-fields", "S {} want {want_s}, created {} last_recall {} now {}", trace.stability, trace.created_at, trace.last_recall, now as i64);
                let removed: BTreeSet<String> = prev.keys().filter(|id| !next.contains_key(*id)).cloned().collect();
                let new_stored = next.contains_key(&trace.id);
                let excess = (prev.len() + 1).saturating_sub(self.cfg.capacity);
                ck!(self, next.len() == (prev.len() + 1).min(self.cfg.capacity), "I3-count-after-insert", "{} rows after insert (prev {}, cap {})", next.len(), prev.len(), self.cfg.capacity);
                let mut victims = removed.clone();
                if !new_stored {
                    victims.insert(trace.id.clone());
                }
                ck!(self, victims.len() == excess, "I3-victim-count", "{} victims for excess {excess}", victims.len());
                if let Some(s) = next.get(&trace.id) {
                    ck!(self, *s == trace, "I-inserted-row", "stored row differs from the returned trace");
                }
                for (id, p) in &prev {
                    if let Some(n) = next.get(id) {
                        ck!(self, n == p, "I-insert-rewrote", "remember rewrote {id}");
                    }
                }
                match &evicted {
                    Some(e) => {
                        ck!(self, victims.contains(&e.id), "I3-evicted-report", "evicted {} is not among the removed {victims:?}", e.id);
                        ck!(self, new_stored || e.id == trace.id, "I3-evicted-newcomer", "newcomer not stored but evicted names {}", e.id);
                    }
                    None => ck!(self, victims.is_empty(), "I3-evicted-none", "victims {victims:?} but evicted None"),
                }
                // eviction against a brute-force ranking (SPEC section 3 / 4.5)
                if excess > 0 {
                    self.eviction_oracle(&prev, &trace, &victims, now)?;
                }
                // reactivation / labile flag against brute force (faults off)
                if emb_ok && st_ok {
                    ck!(self, trace.vector.len() == dim && trace.model_id == cur, "I2-faultless-unindexed", "no embedder fault but vector len {} model {:?}", trace.vector.len(), trace.model_id);
                    let q = if trace.cue.is_empty() { trace.vector.clone() } else { self.oracle_embed(&trace.cue) };
                    let mut best = f64::NEG_INFINITY;
                    for p in prev.values().filter(|p| p.model_id == cur && p.vector.len() == dim) {
                        best = best.max(dot64(&p.vector, &q));
                    }
                    let theta = self.cfg.theta_related;
                    if (best - theta).abs() > 1e-5 {
                        let stale = prev.values().filter(|p| p.model_id != cur).count();
                        let want_cons = best < theta && stale == 0;
                        ck!(self, trace.consolidated == want_cons, "I-consolidated-flag", "consolidated {} but best older cosine {best:.4} (theta {theta}), stale rows {stale}", trace.consolidated);
                        let want_cos = if best >= theta { best } else { 0.0 };
                        ck!(self, (cosine - want_cos).abs() < 1e-4, "I-reactivation-cosine", "cosine {cosine} vs brute force {want_cos}");
                    }
                }
            }
        }
        self.finish(next, &HashSet::new())
    }

    fn oracle_embed(&mut self, text: &str) -> Vec<f32> {
        let m = lk(&self.ctl).model;
        self.oracle[m].embed(text)
    }

    fn eviction_oracle(&mut self, prev: &Rows, new: &Trace, victims: &BTreeSet<String>, now: f64) -> Res {
        let cfg = &self.cfg;
        let excess = (prev.len() + 1).saturating_sub(cfg.capacity);
        let cutoff = (now - cfg.grace_period) as i64;
        let young = prev.values().filter(|t| t.created_at > cutoff).count() + usize::from(new.created_at > cutoff);
        let rank = |t: &Trace| {
            let c = self.chk(t, now);
            c.stability.log2() - (now - c.last_recall as f64).max(0.0) / c.stability
        };
        let mut pool: Vec<(f64, String)> = Vec::new();
        if young <= cfg.capacity / 10 {
            pool = prev.values().filter(|t| t.created_at <= cutoff).map(|t| (rank(t), t.id.clone())).collect();
        }
        if pool.len() < excess {
            pool = prev.values().map(|t| (rank(t), t.id.clone())).collect();
            pool.push((rank(new), new.id.clone()));
        }
        pool.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        pool.truncate(excess);
        let want: BTreeSet<String> = pool.iter().map(|p| p.1.clone()).collect();
        if want == *victims {
            return Ok(());
        }
        // a different set is fine when only ties / dead traces (strength <= 2^-64) differ
        let all: HashMap<String, f64> = prev.values().map(|t| (t.id.clone(), rank(t))).chain(std::iter::once((new.id.clone(), rank(new)))).collect();
        let mut got: Vec<f64> = victims.iter().map(|id| all.get(id).copied().unwrap_or(f64::NAN)).collect();
        let mut exp: Vec<f64> = pool.iter().map(|p| p.0).collect();
        got.sort_by(f64::total_cmp);
        exp.sort_by(f64::total_cmp);
        let same = got.iter().zip(&exp).all(|(g, e)| (g - e).abs() <= 1e-9 * e.abs().max(1.0) || (*g <= -64.0 && *e <= -64.0));
        ck!(self, same, "I4-eviction-not-weakest", "victims {victims:?} (ranks {got:?}) vs brute-force weakest {want:?} (ranks {exp:?}), young {young}, cap {}", self.cfg.capacity);
        Ok(())
    }

    /// Brute-force recall scores over every row indexed under the current model.
    fn oracle_scores(&mut self, query: &str, rows: &Rows, now: f64) -> Vec<(String, f64, f64)> {
        let (cur, dim) = self.model();
        let parts = cues(query, self.cfg.max_cues);
        if parts.is_empty() {
            return Vec::new();
        }
        let qs: Vec<Vec<f32>> = parts.iter().map(|p| self.oracle_embed(p)).collect();
        let mut out = Vec::new();
        for t in rows.values().filter(|t| t.model_id == cur && t.vector.len() == dim) {
            let cos = qs.iter().map(|q| dot64(&t.vector, q).min(1.0)).fold(f64::NEG_INFINITY, f64::max);
            let rv = self.retr(t, now);
            out.push((t.id.clone(), self.activation(cos) * (self.cfg.alpha + (1.0 - self.cfg.alpha) * rv), cos));
        }
        out
    }

    fn check_pack(&mut self, r: &Recall) -> Res {
        let cfg = self.cfg.clone();
        ck!(self, r.recalled.len() <= cfg.inject_n, "I-recall-too-many", "{} recalled > inject_n {}", r.recalled.len(), cfg.inject_n);
        let mut expected = String::new();
        for c in &r.recalled {
            expected.push_str(&format!("[{} {}] {}　《id:{}》\n", c.trace.created_at, c.trace.tz, c.trace.text, c.trace.id));
        }
        ck!(self, expected == r.pack_text, "I-pack-format", "pack_text does not equal its recalled traces");
        let used = r.pack_text.chars().count();
        ck!(self, used <= cfg.budget_chars || r.recalled.len() <= 1, "I-pack-budget", "{used} chars over budget {} with {} lines", cfg.budget_chars, r.recalled.len());
        let ids: BTreeSet<&String> = r.recalled.iter().map(|c| &c.trace.id).collect();
        ck!(self, ids.len() == r.recalled.len(), "I-recall-duplicates", "duplicate recalled ids");
        let top = r.recalled.iter().map(|c| c.score).fold(0.0, f64::max);
        for c in &r.recalled {
            ck!(self, c.score >= cfg.min_score - 1e-9 && c.score >= cfg.relative_score * top - 1e-9, "I-recall-floor", "score {} below floor (top {top})", c.score);
            ck!(self, (0.0..=1.0).contains(&c.retrievability) && c.cosine <= 1.0 + 1e-9, "I-recall-range", "R {} cos {}", c.retrievability, c.cosine);
        }
        Ok(())
    }

    fn check_recall_oracle(&mut self, q: &str, prev: &Rows, r: &Recall, now: f64) -> Res {
        let scores = self.oracle_scores(q, prev, now);
        let top = scores.iter().map(|s| s.1).fold(0.0, f64::max);
        let by_id: HashMap<&str, &(String, f64, f64)> = scores.iter().map(|s| (s.0.as_str(), s)).collect();
        let floor = self.cfg.min_score.max(self.cfg.relative_score * top);
        for c in &r.recalled {
            match by_id.get(c.trace.id.as_str()) {
                Some(s) => {
                    ck!(self, (s.1 - c.score).abs() < 1e-5 && s.1 >= floor - 1e-5, "I-recall-oracle-score", "{} score {} vs brute force {} (floor {floor})", c.trace.id, c.score, s.1);
                }
                None => ck!(self, false, "I-recall-oracle-member", "{} recalled but is not an indexed row of the current model", c.trace.id),
            }
        }
        let mut sorted: Vec<f64> = scores.iter().map(|s| s.1).collect();
        sorted.sort_by(|a, b| b.total_cmp(a));
        if top >= self.cfg.min_score + 1e-5 && sorted.get(1).map_or(true, |s2| top - s2 > 1e-5) {
            let best = scores.iter().find(|s| s.1 == top).unwrap().0.clone();
            ck!(self, r.recalled.first().map(|c| c.trace.id.clone()) == Some(best.clone()), "I-recall-oracle-top", "best row {best} (score {top}) not first in {:?}", r.recalled.iter().map(|c| &c.trace.id).collect::<Vec<_>>());
        }
        ORACLE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some((ids, text)) = self.oracle_recall_full(q, prev, now) {
            ORACLE_EXACT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let got: Vec<String> = r.recalled.iter().map(|c| c.trace.id.clone()).collect();
            ck!(self, got == ids, "I-recall-oracle-set", "recalled {got:?} but brute force (pool, MMR, packing) gives {ids:?}");
            ck!(self, r.pack_text == text, "I-recall-oracle-pack", "pack text differs from brute force");
        }
        if top < self.cfg.min_score - 1e-5 {
            ck!(self, r.recalled.is_empty(), "I-recall-oracle-silence", "recalled {} traces though nothing scores >= min_score", r.recalled.len());
        }
        Ok(())
    }

    fn op_recall(&mut self, q: &str, peek: bool) -> Res {
        let now = self.now();
        let prev = self.rows.clone();
        let (emb_ok, st_ok) = self.faultless();
        let r = if peek { self.call(|e| e.peek(q))? } else { self.call(|e| e.recall(q))? };
        let next = self.read_rows();
        let verb = if peek { "peek" } else { "recall" };
        match r {
            Err(e) => {
                self.note(format!("{verb} Err({e})"));
                self.unchanged(&next, "I-err-but-applied", verb)?;
                if !peek {
                    self.exposures.clear();
                }
            }
            Ok(r) => {
                let ids: Vec<usize> = r.recalled.iter().map(|c| self.ord(&c.trace.id)).collect();
                self.note(format!("{verb} Ok {ids:?} {}", r.pack_text.chars().count()));
                self.check_pack(&r)?;
                if emb_ok && st_ok {
                    self.check_recall_oracle(q, &prev, &r, now)?;
                }
                for c in &r.recalled {
                    match prev.get(&c.trace.id) {
                        Some(p) => ck!(self, self.chk(p, now) == c.trace, "I-recalled-trace", "{} differs from the stored row", c.trace.id),
                        None => ck!(self, false, "I-recalled-ghost", "{} is not in the store", c.trace.id),
                    }
                }
                if peek {
                    self.unchanged(&next, "I-peek-wrote", "peek")?;
                } else {
                    let allowed: BTreeSet<String> = r.recalled.iter().map(|c| c.trace.id.clone()).collect();
                    self.only_state_changes(&prev, &next, &allowed, "recall")?;
                    for c in &r.recalled {
                        let (p, n) = (self.chk(&prev[&c.trace.id], now), &next[&c.trace.id]);
                        let want = self.clamp_s(p.stability * (1.0 + self.cfg.spacing_gain * 0.5 * self.activation(c.cosine) * (1.0 - self.retr(&prev[&c.trace.id], now))));
                        ck!(self, n.last_recall == now as i64 && n.consolidated == p.consolidated, "I-recall-state", "{}: last_recall {} consolidated {}", c.trace.id, n.last_recall, n.consolidated);
                        ck!(self, (n.stability - want).abs() <= 1e-9 * want.max(1.0) && n.stability >= p.stability - 1e-9, "I-recall-stability", "{}: {} -> {} want {want}", c.trace.id, p.stability, n.stability);
                    }
                    self.exposures = r.recalled.iter().map(|c| (c.trace.id.clone(), (c.cosine, c.retrievability))).collect();
                    ck!(self, self.exposures.len() <= self.cfg.inject_n, "I-exposures-bounded", "{} exposures", self.exposures.len());
                }
            }
        }
        self.finish(next, &HashSet::new())
    }

    fn op_peek_recall(&mut self, q: &str) -> Res {
        // identical state, identical clock, faults off: peek must equal what recall injects
        self.set_suppress(true);
        let now = self.now();
        let peek = self.call(|e| e.peek(q));
        let mid = self.read_rows();
        let rec = self.call(|e| e.recall(q));
        let next = self.read_rows();
        self.set_suppress(false);
        let (peek, rec) = (peek?, rec?);
        match (peek, rec) {
            (Ok(p), Ok(r)) => {
                self.note(format!("peekrecall {}", r.recalled.len()));
                ck!(self, p == r, "I-peek-differs-from-recall", "peek {:?} vs recall {:?}", p.recalled.iter().map(|c| (&c.trace.id, c.score)).collect::<Vec<_>>(), r.recalled.iter().map(|c| (&c.trace.id, c.score)).collect::<Vec<_>>());
                ck!(self, mid == self.rows, "I-peek-wrote", "peek changed the store");
                let prev = self.rows.clone();
                self.check_pack(&r)?;
                self.check_recall_oracle(q, &prev, &r, now)?;
                self.exposures = r.recalled.iter().map(|c| (c.trace.id.clone(), (c.cosine, c.retrievability))).collect();
                // a second recall in the same instant: stability grows only by the (1-R) term
            }
            (a, b) => ck!(self, false, "I-faultless-op-failed", "peek {:?} recall {:?} with faults off", a.err().map(|e| e.to_string()), b.err().map(|e| e.to_string())),
        }
        self.finish(next, &HashSet::new())
    }

    fn op_cite(&mut self, seed: u64) -> Res {
        let mut r = Rng::new(seed);
        let prev = self.rows.clone();
        let exp_ids: Vec<String> = self.exposures.keys().cloned().collect();
        let all_ids: Vec<String> = prev.keys().cloned().collect();
        let mut reply = String::from("sure. ");
        for _ in 0..1 + r.below(4) {
            let id = match r.below(6) {
                0 | 1 | 2 if !exp_ids.is_empty() => r.pick(&exp_ids),
                3 if !all_ids.is_empty() => r.pick(&all_ids),
                4 => "nope".to_string(),
                _ => String::new(),
            };
            reply.push_str(&match r.below(8) {
                0 => format!("《id:{id}"),
                1 => format!("《id:《id:{id}》"),
                2 => format!("《id:{id}》《id:{id}》"),
                3 => format!("《 id:{id}》"),
                4 => format!("《id:{id} 》"),
                _ => format!("《id:{id}》"),
            });
            reply.push(' ');
        }
        let now = self.now();
        let mut expected: Vec<String> = Vec::new();
        for id in parse_cites(&reply) {
            if self.exposures.contains_key(&id) && prev.contains_key(&id) && !expected.contains(&id) {
                expected.push(id);
            }
        }
        let res = self.call(|e| e.cite(&reply))?;
        let next = self.read_rows();
        match res {
            Err(e) => {
                self.note(format!("cite Err({e})"));
                self.unchanged(&next, "I-err-but-applied", "cite")?;
            }
            Ok(got) => {
                let g: Vec<usize> = got.iter().map(|i| self.ord(i)).collect();
                self.note(format!("cite Ok {g:?}"));
                ck!(self, got == expected, "I-cite-result", "cite returned {got:?}, expected {expected:?} (reply {reply:?})");
                ck!(self, got.len() <= self.cfg.inject_n, "I-exposures-bounded", "cite returned {} ids", got.len());
                let allowed: BTreeSet<String> = got.iter().cloned().collect();
                self.only_state_changes(&prev, &next, &allowed, "cite")?;
                for id in &got {
                    let (p, n) = (self.chk(&prev[id], now), &next[id]);
                    let (cos, rv) = self.exposures[id];
                    let partial = 1.0 + self.cfg.spacing_gain * 0.5 * self.activation(cos) * (1.0 - rv);
                    let full = 1.0 + self.cfg.spacing_gain * (1.0 - rv);
                    let want = self.clamp_s(p.stability * (full / partial));
                    ck!(self, n.stability >= p.stability - 1e-9, "I-cite-lowered", "{id}: {} -> {}", p.stability, n.stability);
                    ck!(self, (n.stability - want).abs() <= 1e-6 * want.max(1.0), "I-cite-formula", "{id}: {} want {want}", n.stability);
                    ck!(self, n.last_recall == p.last_recall && n.consolidated == p.consolidated, "I-cite-state", "{id}: last_recall/consolidated changed");
                }
                for id in &got {
                    self.exposures.remove(id);
                }
                // a citation is spent: citing the same reply again changes nothing
                let again = self.call(|e| e.cite(&reply))?;
                if let Ok(v) = again {
                    let after = self.read_rows();
                    ck!(self, v.is_empty() && after == next, "I-cite-twice", "second cite of the same reply returned {v:?} / changed the store");
                }
            }
        }
        let next = self.read_rows();
        self.finish(next, &HashSet::new())
    }

    fn op_forget(&mut self, seed: u64) -> Res {
        let mut r = Rng::new(seed);
        let prev = self.rows.clone();
        let ids: Vec<String> = prev.keys().cloned().collect();
        let deads: Vec<String> = self.dead.iter().cloned().collect();
        let id = match r.below(8) {
            0 => String::new(),
            1 => "nope".into(),
            2 => "a\nb".into(),
            3 if !deads.is_empty() => deads[r.below(deads.len())].clone(),
            _ if !ids.is_empty() => ids[r.below(ids.len())].clone(),
            _ => "x".into(),
        };
        let res = self.call(|e| e.forget(&id))?;
        let next = self.read_rows();
        match res {
            Err(e) => {
                self.note(format!("forget Err({e})"));
                self.unchanged(&next, "I-err-but-applied", "forget")?;
            }
            Ok(Some(t)) => {
                self.note("forget Some".into());
                ck!(self, t.id == id && prev.contains_key(&id), "I-forget-result", "forget({id:?}) returned {} / existed {}", t.id, prev.contains_key(&id));
                let mut want = prev.clone();
                want.remove(&id);
                ck!(self, next == want, "I-forget-collateral", "forget({id:?}) removed more or changed other rows: {:?}", diff(&prev, &next));
            }
            Ok(None) => {
                self.note("forget None".into());
                ck!(self, !prev.contains_key(&id), "I-forget-missed", "forget({id:?}) said None but the row exists");
                self.unchanged(&next, "I-forget-none-wrote", "forget None")?;
            }
        }
        self.finish(next, &HashSet::new())
    }

    fn op_reindex(&mut self, limit: usize) -> Res {
        let prev = self.rows.clone();
        let (emb_ok, st_ok) = self.faultless();
        let (cur, _) = self.model();
        let stale = prev.values().filter(|t| t.model_id != cur).count();
        let res = self.call(|e| e.reindex(limit))?;
        let next = self.read_rows();
        let d = diff(&prev, &next);
        ck!(self, d.0.is_empty() && d.1.is_empty(), "I-reindex-ids", "reindex changed ids {:?} {:?}", d.0, d.1);
        let mut indexed = 0;
        for id in &d.2 {
            let (a, b) = (&prev[id], &next[id]);
            ck!(self, a.model_id != cur && b.model_id == cur && !b.vector.is_empty(), "I-reindex-unstale", "{id}: model {:?} -> {:?}", a.model_id, b.model_id);
            ck!(self, Trace { model_id: String::new(), vector: Vec::new(), ..a.clone() } == Trace { model_id: String::new(), vector: Vec::new(), ..b.clone() }, "I-reindex-rewrote", "{id}: more than the index changed");
            indexed += 1;
        }
        if emb_ok && st_ok {
            let mut by_model: BTreeMap<String, Vec<(bool, String)>> = BTreeMap::new();
            for (id, t) in &prev {
                if t.model_id != cur {
                    by_model.entry(t.model_id.clone()).or_default().push((d.2.contains(id), id.clone()));
                }
            }
            for (m, v) in &by_model {
                let oldest_done = v.iter().filter(|x| x.0).map(|x| &x.1).min();
                let newest_left = v.iter().filter(|x| !x.0).map(|x| &x.1).max();
                if let (Some(a), Some(b)) = (oldest_done, newest_left) {
                    ck!(self, a > b, "I-reindex-order", "model {m:?}: re-indexed {a} while the newer {b} was left stale");
                }
            }
        }
        match res {
            Err(e) => self.note(format!("reindex Err({e})")),
            Ok(n) => {
                self.note(format!("reindex Ok {n}"));
                ck!(self, n == indexed, "I-reindex-count", "returned {n}, rows indexed {indexed}");
                ck!(self, n <= limit, "I-reindex-limit", "{n} > limit {limit}");
                if emb_ok && st_ok {
                    ck!(self, n == stale.min(limit), "I-reindex-short", "{n} indexed, expected min({stale}, {limit})");
                }
            }
        }
        self.finish(next, &HashSet::new())
    }

    fn op_clusters(&mut self, budget: Option<usize>) -> Res {
        let prev = self.rows.clone();
        let (cur, _) = self.model();
        let res = self.call(|e| e.clusters(budget))?;
        let next = self.read_rows();
        self.unchanged(&next, "I-clusters-wrote", "clusters")?;
        match res {
            Err(e) => self.note(format!("clusters Err({e})")),
            Ok(cl) => {
                self.note(format!("clusters Ok {}", cl.len()));
                let b = budget.unwrap_or(self.cfg.dream_budget);
                ck!(self, cl.len() <= 8 * b, "I-clusters-count", "{} clusters for budget {b}", cl.len());
                let mut seeds = HashSet::new();
                for c in &cl {
                    ck!(self, c.len() >= 2 && c.len() <= self.cfg.dream_max_members, "I-cluster-size", "cluster of {}", c.len());
                    let s = &c[0];
                    ck!(self, seeds.insert(s.id.clone()), "I-cluster-seed-dup", "seed {} twice", s.id);
                    ck!(self, prev.get(&s.id).is_some_and(|p| !p.consolidated && p.model_id == cur), "I-cluster-seed-state", "seed {} is not a labile indexed row", s.id);
                    let mut ids = HashSet::new();
                    for m in &c[1..] {
                        ck!(self, ids.insert(m.id.clone()) && m.id != s.id, "I-cluster-dup-member", "member {} repeated", m.id);
                        ck!(self, prev.get(&m.id).is_some_and(|p| p.created_at <= s.created_at && p.model_id == cur), "I-cluster-member", "member {} is newer than the seed or unindexed", m.id);
                    }
                }
            }
        }
        self.finish(next, &HashSet::new())
    }

    fn op_reset(&mut self) -> Res {
        let res = self.call(|e| e.reset())?;
        let next = self.read_rows();
        match res {
            Err(e) => {
                self.note(format!("reset Err({e})"));
                self.unchanged(&next, "I-err-but-applied", "reset")?;
            }
            Ok(()) => {
                self.note("reset Ok".into());
                ck!(self, next.is_empty(), "I-reset", "{} rows after reset", next.len());
                self.lo.clear();
                self.hi.clear();
                self.exposures.clear();
                self.overshoot = false;
                // the engine's cursor/exposures/write window are gone: a fresh write must not be limited
            }
        }
        self.finish(next, &HashSet::new())
    }

    fn op_undo_stale(&mut self, seed: u64) -> Res {
        if self.history.is_empty() {
            self.note("undo-stale none".into());
            let next = self.read_rows();
            return self.finish(next, &HashSet::new());
        }
        let mut r = Rng::new(seed);
        let reports = self.history[r.below(self.history.len())].clone();
        let restored: HashSet<String> = reports.iter().flat_map(|x| x.absorbed.iter().map(|t| t.id.clone())).collect();
        let res = self.call(|e| e.undo_all(&reports))?;
        let next = self.read_rows();
        if next.len() > self.cfg.capacity {
            self.overshoot = true; // a failed undo may have applied some reports
        }
        match res {
            Err(e) => self.note(format!("undo-stale Err({e})")),
            Ok(()) => {
                self.note("undo-stale Ok".into());
            }
        }
        self.finish(next, &restored)
    }

    fn op_dream(&mut self, budget: Option<usize>, plan: &AdjPlan, undo: bool) -> Res {
        let now = self.now();
        let prev = self.rows.clone();
        let (cur, dim) = self.model();
        let log: Arc<Mutex<Vec<Call>>> = Arc::new(Mutex::new(Vec::new()));
        let mut adj = Adj { rng: Rng::new(plan.seed), plan: plan.clone(), log: log.clone(), known_ids: prev.keys().cloned().collect() };
        let (emb_ok, st_ok) = self.faultless();
        let eff_budget = budget.unwrap_or(self.cfg.dream_budget);
        let preview = if emb_ok && st_ok && eff_budget > 0 && prev.values().all(|t| t.model_id == cur) { Some(self.call(|e| e.clusters(Some(eff_budget)))?) } else { None };
        let res = self.call(|e| e.dream(&mut adj, budget))?;
        let next = self.read_rows();
        let calls = std::mem::take(&mut *lk(&log));
        let d = diff(&prev, &next);
        ck!(self, calls.len() <= eff_budget, "I7-llm-over-budget", "{} LLM calls, budget {eff_budget}", calls.len());
        if let Some(Ok(cl)) = &preview {
            let first: Option<Vec<String>> = cl.first().map(|c| c.iter().map(|t| t.id.clone()).collect());
            let asked: Option<Vec<String>> = calls.first().map(|c| c.members.clone());
            ck!(self, first == asked, "I7-preview-differs", "clusters() previewed {first:?} but the dream first asked {asked:?}");
        }
        let mut seen = HashSet::new();
        for c in &calls {
            ck!(self, seen.insert(c.members[0].clone()), "I7-llm-twice-per-seed", "seed {} adjudicated twice in one dream", c.members[0]);
            ck!(self, c.members.len() >= 2 && c.members.len() <= self.cfg.dream_max_members, "I7-cluster-size", "cluster of {}", c.members.len());
            let seed = prev.get(&c.members[0]);
            ck!(self, seed.is_some_and(|s| !s.consolidated), "I7-seed-not-labile", "seed {} was not a labile row", c.members[0]);
        }
        let reports = match res {
            Err(e) => {
                self.note(format!("dream Err({e})"));
                ck!(self, calls.is_empty(), "I7-dream-err-after-llm", "dream returned Err after {} LLM verdict calls", calls.len());
                ck!(self, d.0.is_empty() && d.1.is_empty(), "I7-dream-err-wrote", "dream Err but ids changed {:?} {:?}", d.0, d.1);
                for id in &d.2 {
                    let (a, b) = (&prev[id], &next[id]);
                    ck!(self, Trace { model_id: String::new(), vector: Vec::new(), ..a.clone() } == Trace { model_id: String::new(), vector: Vec::new(), ..b.clone() }, "I7-dream-err-wrote", "{id} changed beyond its index");
                }
                return self.finish(next, &HashSet::new());
            }
            Ok(r) => r,
        };
        let tags: Vec<String> = reports.iter().map(|r| r.action.as_str().to_string()).collect();
        self.note(format!("dream Ok {tags:?} calls={}", calls.len()));
        // reports <-> LLM calls
        let judged: Vec<&DreamReport> = reports.iter().filter(|r| r.before.len() >= 2).collect();
        ck!(self, judged.len() == calls.len(), "I7-report-per-verdict", "{} judged reports for {} LLM calls", judged.len(), calls.len());
        let mut known: HashMap<String, i64> = prev.values().map(|t| (t.id.clone(), t.created_at)).collect();
        for rep in &reports {
            known.extend(rep.after.iter().map(|g| (g.id.clone(), g.created_at)));
        }
        for c in &calls {
            let sc = known.get(&c.members[0]).copied();
            for m in &c.members[1..] {
                ck!(self, known.get(m).is_some_and(|&t| Some(t) <= sc), "I7-candidate-newer", "candidate {m} ({:?}) is newer than its seed {} ({sc:?})", known.get(m), c.members[0]);
            }
        }
        let mut sim: Rows = prev.clone();
        let mut removed_exp: BTreeSet<String> = BTreeSet::new();
        let mut added_exp: BTreeSet<String> = BTreeSet::new();
        let mut keep_seeds: BTreeSet<String> = BTreeSet::new();
        let mut replace_seeds: BTreeSet<String> = BTreeSet::new();
        for (rep, call) in judged.iter().zip(&calls) {
            let ids: Vec<String> = rep.before.iter().map(|m| m.id.clone()).collect();
            ck!(self, ids == call.members, "I7-report-members", "report members {ids:?} vs asked {:?}", call.members);
            let seed_id = &call.members[0];
            match (rep.action, &call.decision) {
                (DreamAction::Error, _) => ck!(self, rep.error.is_some() && rep.absorbed.is_empty() && rep.after.is_empty(), "I7-error-report", "Error report with absorbed/after or no message"),
                (_, None) => ck!(self, false, "I7-verdict-from-error", "the LLM failed but the report says {:?}", rep.action),
                (DreamAction::Keep, Some(dec)) => {
                    ck!(self, rep.absorbed.is_empty() && rep.after.is_empty() && rep.error.is_none(), "I7-keep-report", "keep report carries rows");
                    keep_seeds.insert(seed_id.clone());
                    if let Some(s) = sim.get_mut(seed_id) {
                        s.consolidated = true;
                    }
                    let texts = cleaned_texts(&dec.memories, self.cfg.text_max);
                    if !dec.memories.is_empty() && texts.is_empty() {
                        ck!(self, false, "I7-silence-settled-seed", "a verdict of only-empty texts was applied as keep");
                    }
                    if emb_ok && !texts.is_empty() {
                        // legitimate fallbacks: more texts than absorbed members, or a confabulated gist
                        let named: HashSet<&str> = dec.absorbed_ids.iter().map(String::as_str).collect();
                        let absorbed: Vec<&Trace> = call.members.iter().enumerate().filter(|(i, m)| *i == 0 || named.contains(m.as_str())).filter_map(|(_, m)| prev.get(m)).collect();
                        let guard = texts.iter().any(|t| {
                            let g = self.oracle_embed(t);
                            absorbed.iter().filter(|a| a.vector.len() == g.len()).map(|a| dot64(&a.vector, &g)).fold(f64::NEG_INFINITY, f64::max) < self.cfg.gist_min_cosine + 1e-4
                        });
                        ck!(self, texts.len() > absorbed.len() || guard, "I7-keep-instead-of-replace", "keep though {} valid gist(s) for {} absorbed pass the guard", texts.len(), absorbed.len());
                    }
                }
                (DreamAction::Replace, Some(dec)) => {
                    ck!(self, rep.error.is_none() && !rep.after.is_empty(), "I7-replace-report", "replace report without gists");
                    let texts = cleaned_texts(&dec.memories, self.cfg.text_max);
                    let named: HashSet<&str> = dec.absorbed_ids.iter().map(String::as_str).collect();
                    let want_abs: Vec<&String> = call.members.iter().enumerate().filter(|(i, m)| *i == 0 || named.contains(m.as_str())).map(|(_, m)| m).collect();
                    let got_abs: Vec<&String> = rep.absorbed.iter().map(|t| &t.id).collect();
                    ck!(self, got_abs == want_abs, "I7-absorbed-set", "absorbed {got_abs:?} but seed + named candidates are {want_abs:?}");
                    ck!(self, rep.after.len() <= rep.absorbed.len(), "I7-dream-grew", "{} gists for {} absorbed", rep.after.len(), rep.absorbed.len());
                    let after_texts: Vec<&String> = rep.after.iter().map(|t| &t.text).collect();
                    ck!(self, after_texts == texts.iter().collect::<Vec<_>>(), "I7-gist-texts", "gists {after_texts:?} vs cleaned verdict {texts:?}");
                    let seed = rep.absorbed.first();
                    let sum: f64 = rep.absorbed.iter().map(|t| self.eng.strength(t, now)).sum();
                    if !rep.absorbed.is_empty() {
                        let strongest = rep.absorbed.iter().fold(None::<&Trace>, |a, b| match a { Some(a) if a.stability >= b.stability => Some(a), _ => Some(b) }).unwrap();
                        let want_s = self.clamp_s(strongest.stability + sum - self.eng.strength(strongest, now));
                        let r = (sum / want_s).min(1.0);
                        let elapsed = if r > 0.0 { (-r.log2()).min(64.0) } else { 64.0 };
                        let want_lr = now as i64 - (want_s * elapsed).round() as i64;
                        for g in &rep.after {
                            ck!(self, (g.stability - want_s).abs() <= 1e-9 * want_s.max(1.0), "I7-gist-stability-formula", "gist stability {} vs max S + live evidence {want_s}", g.stability);
                            ck!(self, (g.last_recall - want_lr).abs() <= 1, "I7-gist-last-recall-formula", "gist last_recall {} vs {want_lr}", g.last_recall);
                        }
                    }
                    for g in &rep.after {
                        ck!(self, text_ok(&g.text, self.cfg.text_max) && g.consolidated && id_ok(&g.id), "I7-gist-valid", "gist {:?}", g.text);
                        if let Some(raw) = prev.get(seed_id) {
                            if g.created_at != raw.created_at {
                                self.flag("I7-gist-created-at-clamped", format!("gist {} stated at {} but its seed {} was stated at {} (now {})", g.id, g.created_at, raw.id, raw.created_at, now as i64))?;
                            }
                        }
                        ck!(self, g.vector.len() == dim && g.model_id == cur, "I7-gist-index", "gist unindexed");
                        if let Some(s) = seed {
                            ck!(self, g.created_at == s.created_at && g.tz == s.tz && g.cue == s.cue, "I7-gist-inherit", "gist did not inherit the seed's time/tz/cue");
                        }
                        let gs = self.eng.strength(g, now);
                        let slack = 2f64.powf(1.0 / g.stability.max(1.0)) * (1.0 + 1e-9);
                        ck!(self, gs <= sum * slack + 1e-9, "I7-gist-inflates", "gist strength {gs} > cluster strength {sum}");
                        ck!(self, g.stability >= rep.absorbed.iter().map(|t| t.stability).fold(0.0, f64::max).min(self.cfg.max_stability) * (1.0 - 1e-9), "I7-gist-weaker-than-member", "gist stability {}", g.stability);
                        ck!(self, rep.absorbed.iter().any(|a| a.vector.len() == g.vector.len() && dot64(&a.vector, &g.vector) >= self.cfg.gist_min_cosine - 1e-4), "I7-confabulation-guard", "gist {:?} is not close to any absorbed member", g.text);
                    }
                    for a in &rep.absorbed {
                        let p = sim.get(&a.id);
                        ck!(self, p.is_some_and(|p| {
                            let mut q = self.chk(p, now);
                            q.model_id = a.model_id.clone();
                            q.vector = a.vector.clone();
                            q == *a
                        }), "I7-absorbed-as-was", "absorbed {} is not the pre-dream row: report {:?} vs row {:?}", a.id, Trace { vector: vec![], ..a.clone() }, p.map(|p| Trace { vector: vec![], ..self.chk(p, now) }));
                        removed_exp.insert(a.id.clone());
                    }
                    for a in &rep.absorbed {
                        sim.remove(&a.id);
                    }
                    sim.extend(rep.after.iter().map(|g| (g.id.clone(), g.clone())));
                    replace_seeds.insert(seed_id.clone());
                    added_exp.extend(rep.after.iter().map(|t| t.id.clone()));
                }
            }
        }
        // the store moved by exactly what the reports say, and nothing else was rewritten
        let absorbed_all = removed_exp.clone();
        removed_exp.retain(|id| prev.contains_key(id));
        added_exp.retain(|id| !absorbed_all.contains(id));
        let removed: BTreeSet<String> = d.0.iter().cloned().collect();
        let added: BTreeSet<String> = d.1.iter().cloned().collect();
        ck!(self, removed == removed_exp, "I7-unreported-removal", "removed {removed:?}, reports account for {removed_exp:?}; extra in next {:?}; reports {:?}", removed_exp.difference(&removed).map(|i| next.get(i).map(|t| (t.consolidated, t.stability))).collect::<Vec<_>>(), reports.iter().map(|r| (r.action, r.before.iter().map(|m| m.id.clone()).collect::<Vec<_>>(), r.absorbed.iter().map(|t| t.id.clone()).collect::<Vec<_>>(), r.after.iter().map(|t| t.id.clone()).collect::<Vec<_>>())).collect::<Vec<_>>());
        ck!(self, added == added_exp, "I7-unreported-insert", "added {added:?}, reports account for {added_exp:?}");
        for rep in reports.iter().filter(|r| r.action == DreamAction::Replace) {
            for g in &rep.after {
                // a later replace in the same dream may absorb an earlier gist: then it is gone
                if let Some(n) = next.get(&g.id) {
                    ck!(self, n == g, "I7-gist-row", "stored gist differs from the report");
                }
            }
        }
        let mut settled: BTreeSet<String> = BTreeSet::new();
        for id in &d.2 {
            let (a, b) = (&prev[id], &next[id]);
            let flag_cleared = a.consolidated != b.consolidated;
            ck!(self, !(a.consolidated && !b.consolidated), "I7-relabilized", "{id}: consolidated true -> false in a dream");
            let strip = |t: &Trace| Trace { model_id: String::new(), vector: Vec::new(), consolidated: false, ..self.chk(t, now) };
            ck!(self, strip(a) == strip(b), "I7-unnamed-rewrite", "{id}: a dream rewrote a trace beyond index/flag (stability or last_recall changed)");
            if flag_cleared {
                ck!(self, b.model_id == cur && !b.vector.is_empty(), "I7-settled-unindexed", "{id}: labile flag cleared on an unindexed row");
                if !keep_seeds.contains(id) {
                    settled.insert(id.clone());
                }
            } else {
                ck!(self, a.model_id != cur || a.vector.is_empty() || a == b, "I7-rewrite", "{id}");
            }
        }
        for s in &keep_seeds {
            if let Some(n) = next.get(s) {
                ck!(self, n.consolidated, "I7-keep-not-consolidated", "kept seed {s} still labile");
            }
        }
        for s in &settled {
            // settled: its cue found nothing older (checked from the store)
            let _ = s;
        }
        if emb_ok && st_ok && !reports.iter().any(|r| r.action == DreamAction::Replace) {
            self.seed_oracle(&prev, &next, &calls, eff_budget)?;
        }
        ck!(self, next.len() <= prev.len(), "I3-dream-grew-store", "{} rows -> {}", prev.len(), next.len());
        ck!(self, self.overshoot || next.len() <= self.cfg.capacity, "I3-count-after-dream", "{} rows after dream", next.len());
        let _ = replace_seeds;
        self.history.push(reports.clone());
        if self.history.len() > 6 {
            self.history.remove(0);
        }
        let restored = HashSet::new();
        self.finish(next.clone(), &restored)?;
        if undo {
            // newest first: the store must come back exactly (index and settled flags aside)
            self.set_suppress(true);
            let r = self.call(|e| e.undo_all(&reports));
            self.set_suppress(false);
            let r = r?;
            let after = self.read_rows();
            match r {
                Err(e) => ck!(self, false, "I7-undo-failed", "undo_all failed with faults off: {e}"),
                Ok(()) => {
                    self.note("undo Ok".into());
                    let restored: HashSet<String> = reports.iter().flat_map(|x| x.absorbed.iter().map(|t| t.id.clone())).collect();
                    let proj = |t: &Trace, cons: bool| {
                        let c = self.chk(t, now);
                        (c.id, c.text, c.created_at, c.tz, c.last_recall, c.stability.to_bits(), c.cue, cons)
                    };
                    let want: BTreeMap<_, _> = prev.values().map(|t| (t.id.clone(), proj(t, t.consolidated || settled.contains(&t.id)))).collect();
                    let got: BTreeMap<_, _> = after.values().map(|t| (t.id.clone(), proj(t, t.consolidated))).collect();
                    if want != got {
                        let wk: BTreeSet<_> = want.keys().collect();
                        let gk: BTreeSet<_> = got.keys().collect();
                        let bad: Vec<String> = want.iter().filter(|(k, v)| got.get(*k) != Some(v)).map(|(k, v)| format!("{k}: want {v:?} got {:?}", got.get(k))).take(3).collect();
                        self.flag("I7-undo-not-exact", format!("undo_all did not restore the pre-dream rows: missing {:?} extra {:?}; {bad:?}", wk.difference(&gk).collect::<Vec<_>>(), gk.difference(&wk).collect::<Vec<_>>()))?;
                    }
                    if after.len() > self.cfg.capacity {
                        self.overshoot = true;
                    }
                    return self.finish(after, &restored);
                }
            }
            return self.finish(after, &HashSet::new());
        }
        Ok(())
    }

    fn apply(&mut self, op: &Op) -> Res {
        match op {
            Op::Remember { text, sal, cue } => self.op_remember(text, *sal, cue),
            Op::Recall(q) => self.op_recall(q, false),
            Op::Peek(q) => self.op_recall(q, true),
            Op::PeekRecall(q) => self.op_peek_recall(q),
            Op::Cite(s) => self.op_cite(*s),
            Op::Forget(s) => self.op_forget(*s),
            Op::Dream { budget, plan, undo } => self.op_dream(*budget, plan, *undo),
            Op::UndoStale(s) => self.op_undo_stale(*s),
            Op::Reindex(n) => self.op_reindex(*n),
            Op::Clusters(b) => self.op_clusters(*b),
            Op::Reset => self.op_reset(),
            Op::Switch(m) => {
                lk(&self.ctl).model = *m;
                self.note(format!("switch {m}"));
                let next = self.read_rows();
                self.finish(next, &HashSet::new())
            }
            Op::Clock(d) => {
                let t = (self.clock.get() + d).max(self.origin - 1.0e8);
                self.clock.set(t);
                self.note(format!("clock {:.0}", t - self.origin));
                let next = self.read_rows();
                self.finish(next, &HashSet::new())
            }
            Op::Faults(f) => {
                lk(&self.ctl).f = f.clone();
                self.note("faults".into());
                let next = self.read_rows();
                self.finish(next, &HashSet::new())
            }
        }
    }
}

fn cleaned_texts(memories: &[String], max: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for m in memories {
        let c = clean(m, max);
        if !c.is_empty() && !out.contains(&c) {
            out.push(c);
        }
    }
    out
}

fn diff(a: &Rows, b: &Rows) -> (Vec<String>, Vec<String>, Vec<String>) {
    let removed = a.keys().filter(|k| !b.contains_key(*k)).cloned().collect();
    let added = b.keys().filter(|k| !a.contains_key(*k)).cloned().collect();
    let changed = a.iter().filter(|(k, v)| b.get(*k).is_some_and(|w| w != *v)).map(|(k, _)| k.clone()).collect();
    (removed, added, changed)
}

fn run(scn: &Scenario, idx: &[usize]) -> Outcome {
    run_mode(scn, idx, false)
}

fn run_mode(scn: &Scenario, idx: &[usize], strict: bool) -> Outcome {
    let mut r = build_run(scn);
    r.strict = strict;
    let mut fail = None;
    for (step, &i) in idx.iter().enumerate() {
        r.step = step;
        if let Err(f) = r.apply(&scn.ops[i]) {
            fail = Some(f);
            break;
        }
    }
    Outcome { fail, log: r.log, soft: r.soft }
}

fn all_idx(scn: &Scenario) -> Vec<usize> {
    (0..scn.ops.len()).collect()
}

/// Delta debugging: the smallest kept-op list (indices into the seed's op list) that
/// still fails with the same kind.
fn shrink(scn: &Scenario, first: &Fail) -> Vec<usize> {
    shrink_mode(scn, first, false)
}

fn shrink_mode(scn: &Scenario, first: &Fail, strict: bool) -> Vec<usize> {
    let mut cur: Vec<usize> = (0..=first.step).collect();
    let mut budget = 500usize;
    let fails = |idx: &[usize], budget: &mut usize| -> bool {
        if *budget == 0 {
            return false;
        }
        *budget -= 1;
        run_mode(scn, idx, strict).fail.is_some_and(|f| f.kind == first.kind)
    };
    let mut n = 2usize;
    while cur.len() >= 2 && budget > 0 {
        let chunk = cur.len().div_ceil(n);
        let mut reduced = false;
        let mut i = 0;
        while i < cur.len() && budget > 0 {
            let mut cand = cur.clone();
            cand.drain(i..(i + chunk).min(cur.len()));
            if !cand.is_empty() && fails(&cand, &mut budget) {
                cur = cand;
                reduced = true;
            } else {
                i += chunk;
            }
        }
        if reduced {
            n = (n - 1).max(2);
        } else if n >= cur.len() {
            break;
        } else {
            n = (n * 2).min(cur.len());
        }
    }
    cur
}

static ORACLE_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static ORACLE_EXACT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn sweep(group: u64) {
    let seeds: u64 = std::env::var("AUDIT2_SEEDS").ok().and_then(|s| s.parse().ok()).unwrap_or(60);
    let steps: usize = std::env::var("AUDIT2_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(900);
    let mut report: Vec<String> = Vec::new();
    let mut tags: BTreeMap<String, usize> = BTreeMap::new();
    let mut soft: BTreeMap<String, String> = BTreeMap::new();
    for k in 0..seeds {
        let seed = group * 100_000 + k;
        let scn = gen_scenario(seed, steps);
        let idx = all_idx(&scn);
        let a = run(&scn, &idx);
        soft.extend(a.soft.clone());
        for l in &a.log {
            let tag = l.split_once(':').map_or("", |x| x.1);
            let tag: String = tag.split(|c: char| c == '(' || c == '[' || c == '{' || c.is_ascii_digit()).next().unwrap_or("").trim().to_string();
            for w in ["\"replace\"", "\"keep\"", "\"error\""] {
                *tags.entry(format!("verdict {w}")).or_insert(0) += l.matches(w).count();
            }
            if !tag.starts_with("digest") {
                *tags.entry(tag).or_insert(0usize) += 1;
            }
        }
        if let Some(f) = &a.fail {
            let min = shrink(&scn, f);
            let ops: Vec<String> = min.iter().map(|&i| format!("#{i} {:?}", scn.ops[i])).collect();
            report.push(format!("seed {seed}: [{}] at step {} (op #{}): {}\n  replay(seed={seed}, keep={min:?}) cfg {:?}\n  ops: {}", f.kind, f.step, idx[f.step], f.msg, scn.cfg, ops.join("\n       ")));
            continue;
        }
        // determinism: the same seed must give the same transcript
        let b = run(&scn, &idx);
        if a.log != b.log {
            let at = a.log.iter().zip(&b.log).position(|(x, y)| x != y).unwrap_or(a.log.len().min(b.log.len()));
            report.push(format!("seed {seed}: [determinism] transcripts differ at line {at}: {:?} vs {:?}", a.log.get(at), b.log.get(at)));
        }
    }
    eprintln!("group {group} outcome counts: {tags:?}");
    eprintln!("recall oracle: {} faultless recalls, {} compared exactly (rest within float noise of a tie)", ORACLE_CALLS.load(std::sync::atomic::Ordering::Relaxed), ORACLE_EXACT.load(std::sync::atomic::Ordering::Relaxed));
    if !soft.is_empty() {
        eprintln!("soft (known) kinds seen in group {group}: {soft:#?}");
    }
    assert!(report.is_empty(), "{} seeds failed:\n{}", report.len(), report.join("\n"));
}

#[test]
fn fuzz_a() {
    sweep(0);
}
#[test]
fn fuzz_b() {
    sweep(1);
}
#[test]
fn fuzz_c() {
    sweep(2);
}
#[test]
fn fuzz_d() {
    sweep(3);
}

/// Replay a shrunk failure: `(seed, kept op indices)` as printed by a sweep.
#[allow(dead_code)]
fn replay(seed: u64, keep: &[usize]) -> Option<Fail> {
    let scn = gen_scenario(seed, 900);
    run(&scn, keep).fail
}

/// `AUDIT2_REPLAY="seed:i,j,k"` (or `seed:all`) prints the failure of one replay.
#[test]
fn replay_from_env() {
    let Ok(spec) = std::env::var("AUDIT2_REPLAY") else { return };
    let (seed, keep) = spec.split_once(':').expect("seed:indices");
    let seed: u64 = seed.parse().unwrap();
    let scn = gen_scenario(seed, 900);
    let idx: Vec<usize> = if keep == "all" { all_idx(&scn) } else { keep.split(',').map(|x| x.parse().unwrap()).collect() };
    let out = run(&scn, &idx);
    for l in out.log.iter().rev().take(12).rev() {
        eprintln!("{l}");
    }
    eprintln!("RESULT: {:?}", out.fail);
}

impl Run {
    /// With nothing replaced the rows are static during a dream, so every labile seed
    /// can be judged from the end state: a seed whose cue reactivates nothing must be
    /// settled; one that reactivates something must have been judged (budget permitting).
    fn seed_oracle(&mut self, prev: &Rows, next: &Rows, calls: &[Call], budget: usize) -> Res {
        let (cur, dim) = self.model();
        let stale = prev.values().filter(|t| t.model_id != cur).count();
        let seeds: Vec<&Trace> = next.values().filter(|n| prev.get(&n.id).is_some_and(|p| !p.consolidated) && n.model_id == cur && n.vector.len() == dim).collect();
        if calls.len() >= budget || seeds.len() > 8 * budget || stale > 64 * budget.max(1) {
            return Ok(());
        }
        let theta = self.cfg.theta_related;
        let now = self.now() as i64;
        let future = next.values().any(|t| t.created_at > now);
        let sfx = if future { "-future-stamp" } else { "" };
        let judged: HashSet<&String> = calls.iter().map(|c| &c.members[0]).collect();
        for s in seeds {
            let q = if s.cue.is_empty() { s.vector.clone() } else { self.oracle_embed(&s.cue) };
            let mut amb = false;
            let mut has = false;
            for t in next.values().filter(|t| t.id != s.id && t.model_id == cur && t.vector.len() == dim && t.created_at <= s.created_at) {
                let c = dot64(&t.vector, &q);
                amb |= (c - theta).abs() < 1e-5;
                has |= c >= theta;
            }
            if amb {
                continue;
            }
            if !has {
                if !s.consolidated {
                    self.flag(&format!("I7-seed-unsettled{sfx}"), format!("seed {} reactivates nothing but is still labile after a faultless dream (budget {budget}, {} calls)", s.id, calls.len()))?;
                }
            } else if !judged.contains(&s.id) {
                self.flag(&format!("I7-seed-skipped{sfx}"), format!("seed {} has candidates, budget left ({} of {budget} calls) but it was not judged", s.id, calls.len()))?;
            }
        }
        Ok(())
    }
}

// ------------------------------------------------- time translation (metamorphic)

/// Shifting the whole virtual timeline by a constant must not change anything the engine
/// does (only differences of times matter): same seed, same transcript.
#[test]
fn fuzz_time_translation() {
    let seeds: u64 = std::env::var("AUDIT2_SEEDS").ok().and_then(|s| s.parse().ok()).unwrap_or(40) / 2;
    let mut bad = Vec::new();
    for k in 0..seeds {
        let seed = 700_000 + k;
        let a = gen_scenario(seed, 600);
        let mut b = a.clone();
        b.origin += 123_456_789.0;
        let (ra, rb) = (run(&a, &all_idx(&a)), run(&b, &all_idx(&b)));
        if ra.fail.as_ref().map(|f| &f.kind) != rb.fail.as_ref().map(|f| &f.kind) || ra.log != rb.log {
            let at = ra.log.iter().zip(&rb.log).position(|(x, y)| x != y).unwrap_or(0);
            bad.push(format!("seed {seed}: shifted run differs at line {at}: {:?} vs {:?} (fails {:?} / {:?})", ra.log.get(at), rb.log.get(at), ra.fail, rb.fail));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

// ------------------------------------------------------- named defect repros

mod repro {
    use super::support::{build_with, cfg, insert, rows, FakeEmbedder, SharedStore, DAY};
    use engram::{Config, DreamDecision, DreamRequest, Remembered, Storage, Trace, View, Write};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Fails the n-th `find` after being armed, or every write while `fail_writes` is set.
    #[derive(Clone, Default)]
    pub struct Probe {
        pub inner: SharedStore,
        pub finds: Arc<AtomicUsize>,
        pub fail_at: Arc<AtomicUsize>,
        pub fail_writes: Arc<std::sync::atomic::AtomicBool>,
    }

    impl Storage for Probe {
        fn find(&self, view: View<'_>, limit: usize) -> engram::Result<Vec<Trace>> {
            let n = self.finds.fetch_add(1, Ordering::SeqCst) + 1;
            if n == self.fail_at.load(Ordering::SeqCst) {
                return Err("injected find failure".into());
            }
            self.inner.find(view, limit)
        }
        fn count(&self, view: View<'_>) -> engram::Result<usize> {
            self.inner.count(view)
        }
        fn near(&self, q: &[f32], m: &str, c: f64, k: usize, s: Option<i64>) -> engram::Result<Vec<engram::Hit>> {
            self.inner.near(q, m, c, k, s)
        }
        fn write(&mut self, w: &[Write<'_>]) -> engram::Result<()> {
            if self.fail_writes.load(Ordering::SeqCst) {
                return Err("injected write failure".into());
            }
            self.inner.write(w)
        }
        fn clear(&mut self) -> engram::Result<()> {
            self.inner.clear()
        }
    }

    /// D1. `remember` of an existing text commits the rehearsal and THEN re-reads the row;
    /// when that read fails it returns `Err` although the write has landed. A host that
    /// retries on `Err` rehearses the trace twice; the error does not mean "nothing happened".
    #[test]
    fn r2d_remember_reinforce_returns_err_after_commit() {
        let probe = Probe::default();
        let (mut m, clock) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(probe.clone()));
        insert(&mut m, "tokyo osaka", 1.0, "");
        clock.advance(DAY);
        let before = rows(&m);
        probe.finds.store(0, Ordering::SeqCst);
        probe.fail_at.store(2, Ordering::SeqCst); // find(Text) ok, commit, find(Id) fails
        let res = m.remember("tokyo osaka", 1.0, "");
        probe.fail_at.store(0, Ordering::SeqCst);
        let after = rows(&m);
        assert!(res.is_ok() || before == after, "remember returned Err({:?}) but the store changed: stability {} -> {}", res.as_ref().err().map(|e| e.to_string()), before[0].stability, after[0].stability);
    }

    /// D2. The seed is read through `checked`, which clamps `created_at` to now; that
    /// clamped value is the `stated_by` bound of the candidate search, while the store
    /// compares it with the RAW `created_at` of the candidates. After the clock steps
    /// back, every candidate stamped in the "future" window is filtered out, even though
    /// it is not newer than the seed: the seed reactivates nothing, is settled without
    /// the LLM and the update is never adjudicated.
    #[test]
    fn r2d_backward_clock_hides_older_candidates_from_the_dream() {
        let theta = Config { theta_related: 0.5, ..cfg() };
        let (mut m, clock) = build_with(theta.clone(), Box::new(FakeEmbedder::new()), Box::new(SharedStore::default()));
        insert(&mut m, "kyoto fukuoka likes", 1.0, "");
        let seed = insert(&mut m, "kyoto fukuoka likes moved", 1.0, "kyoto fukuoka"); // born labile: its cue reactivates the first
        assert!(!m.memory(&seed).unwrap().unwrap().consolidated);
        let judged = |m: &mut engram::Engram| {
            let mut calls = 0;
            m.dream(&mut |_: &DreamRequest| { calls += 1; Ok(DreamDecision::keep()) }, None).unwrap();
            calls
        };
        // control: same state, clock untouched
        let (mut c, _) = build_with(theta, Box::new(FakeEmbedder::new()), Box::new(SharedStore::default()));
        insert(&mut c, "kyoto fukuoka likes", 1.0, "");
        insert(&mut c, "kyoto fukuoka likes moved", 1.0, "kyoto fukuoka");
        assert_eq!(judged(&mut c), 1, "control: the dream judges the update");
        clock.advance(-3600.0); // NTP step: one hour back
        assert_eq!(judged(&mut m), 1, "after a backward clock step the dream settled the seed without judging it");
    }

    /// D3. A write counts against `writes_per_day` before it is known to succeed: during
    /// an outage a retrying host burns the whole daily quota on writes that stored nothing,
    /// and is then rate limited for a day after the store is back.
    #[test]
    fn r2d_failed_writes_burn_the_daily_quota() {
        let probe = Probe::default();
        let c = Config { writes_per_day: 3, ..cfg() };
        let (mut m, _clock) = build_with(c, Box::new(FakeEmbedder::new()), Box::new(probe.clone()));
        probe.fail_writes.store(true, Ordering::SeqCst);
        for i in 0..3 {
            assert!(m.remember(&format!("fact number {i}"), 1.0, "").is_err());
        }
        probe.fail_writes.store(false, Ordering::SeqCst);
        let r = m.remember("a fact that should be stored", 1.0, "").unwrap();
        assert!(matches!(r, Remembered::Inserted { .. }), "store recovered, nothing was ever stored, yet remember returned {r:?}");
        assert_eq!(rows(&m).len(), 1);
    }
}

mod cycle {
    use super::support::{build, cfg, insert, always_fail};
    use engram::{Config, DreamRequest};

    /// The dream's cursor serves the labile queue as a cycle: with an LLM that always
    /// fails and a budget of one verdict per dream, consecutive dreams must visit
    /// different seeds (SPEC 5: an undecidable seed never starves the ones behind it).
    #[test]
    fn dream_cursor_cycles_through_undecidable_seeds() {
        let (mut m, clock) = build(Config { theta_related: 0.5, ..cfg() });
        insert(&mut m, "kyoto fukuoka likes", 1.0, "");
        let mut seeds = Vec::new();
        for i in 0..3 {
            clock.advance(10.0);
            seeds.push(insert(&mut m, &format!("kyoto fukuoka likes {i}"), 1.0, "kyoto fukuoka"));
        }
        let mut visited = Vec::new();
        for _ in 0..3 {
            let mut asked: Vec<String> = Vec::new();
            let mut adj = |r: &DreamRequest| {
                asked.push(r.members[0].id.clone());
                always_fail(r)
            };
            let reports = m.dream(&mut adj, Some(1)).unwrap();
            assert_eq!(reports.len(), 1);
            visited.extend(asked);
        }
        let mut uniq = visited.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(uniq.len(), 3, "three dreams of one verdict each visited {visited:?}");
    }
}

mod repro2 {
    use super::support::{always_keep, build, build_with, cfg, insert, merge_to_gist, mem, rows, FakeEmbedder, SharedStore, DAY};
    use engram::Config;

    /// D4. A gist should keep "when it was stated" (SPEC 5: 要旨は「いつ述べられたか」を保つ). It inherits
    /// the seed as `checked` read it, and `checked` clamps `created_at` to now: after a backward
    /// clock step the gist is stamped with the earlier clock reading, not the seed's own time.
    #[test]
    fn r2d_replace_gist_keeps_the_seeds_stated_time_after_backward_clock() {
        let (mut m, clock) = build(Config { theta_related: 0.5, ..cfg() });
        insert(&mut m, "kyoto fukuoka likes", 1.0, "");
        clock.advance(2.0 * DAY);
        let seed = insert(&mut m, "kyoto fukuoka likes moved", 1.0, "kyoto fukuoka");
        let stated = mem(&m, &seed).unwrap().created_at;
        clock.advance(-DAY); // the clock steps back one day, still after the older trace
        let reports = m.dream(&mut merge_to_gist, None).unwrap();
        assert_eq!(reports.len(), 1);
        let gist = &reports[0].after[0];
        assert_eq!(gist.created_at, stated, "the gist was stated at {} but its seed at {stated}", gist.created_at);
        let _ = (always_keep, build_with, rows, FakeEmbedder::new, SharedStore::default);
    }

    /// Documented behaviour (gap G5): `undo` is not versioned. Undoing the same reports a second
    /// time, after the host forgot one of the restored traces, brings the forgotten trace back.
    #[test]
    fn note_stale_undo_resurrects_forgotten_trace() {
        let (mut m, clock) = build(Config { theta_related: 0.5, ..cfg() });
        let a = insert(&mut m, "kyoto fukuoka likes", 1.0, "");
        clock.advance(10.0);
        insert(&mut m, "kyoto fukuoka likes moved", 1.0, "kyoto fukuoka");
        let reports = m.dream(&mut merge_to_gist, None).unwrap();
        assert_eq!(rows(&m).len(), 1);
        m.undo_all(&reports).unwrap();
        assert!(m.forget(&a).unwrap().is_some());
        assert!(mem(&m, &a).is_none());
        m.undo_all(&reports).unwrap(); // a stale record
        assert!(mem(&m, &a).is_some(), "a forgotten id came back through a stale undo record");
    }

    /// Documented behaviour (gap G6): the engine refuses an id only while it is LIVE. A generator
    /// that repeats an id (the doc of `IdGen` forbids it) gets the id of a removed trace back.
    #[test]
    fn note_idgen_may_reuse_a_removed_id() {
        let (mut m, _clock) = build(cfg());
        m.set_ids(Box::new(|_, _| "same-id".to_string()));
        let first = insert(&mut m, "alpha beta", 1.0, "");
        assert!(m.remember("gamma delta", 1.0, "").is_err(), "a live id is refused");
        m.forget(&first).unwrap();
        let second = insert(&mut m, "gamma delta", 1.0, "");
        assert_eq!(first, second);
    }
}

impl Run {
    /// A complete brute-force recall (candidate pool, MMR, packing) for comparison with the
    /// engine's answer. `None` when a decision is within float noise of a tie or a threshold.
    fn oracle_recall_full(&mut self, query: &str, rows: &Rows, now: f64) -> Option<(Vec<String>, String)> {
        let scores = self.oracle_scores(query, rows, now);
        let cfg = self.cfg.clone();
        let top = scores.iter().map(|s| s.1).fold(0.0, f64::max);
        let floor = cfg.min_score.max(cfg.relative_score * top);
        let eps = 1e-6;
        if scores.iter().any(|s| (s.1 - floor).abs() < eps || (s.1 - cfg.min_score).abs() < eps) {
            return None;
        }
        // id order = BTreeMap order
        let mut pool: Vec<(&Trace, f64)> = Vec::new();
        for (id, sc, _) in &scores {
            if *sc >= floor {
                pool.push((&rows[id], *sc));
            }
        }
        pool.sort_by(|a, b| a.0.id.cmp(&b.0.id));
        let mut max_sim = vec![0.0f64; pool.len()];
        let mut alive = vec![true; pool.len()];
        let mut order = Vec::new();
        while order.len() < cfg.inject_n {
            let mut best: Option<(usize, f64)> = None;
            let mut second = f64::NEG_INFINITY;
            for i in (0..pool.len()).filter(|&i| alive[i]) {
                let v = pool[i].1 - cfg.mmr_lambda * max_sim[i];
                match best {
                    Some((_, bv)) if v <= bv => second = second.max(v),
                    Some((_, bv)) => {
                        second = second.max(bv);
                        best = Some((i, v));
                    }
                    None => best = Some((i, v)),
                }
            }
            let Some((b, bv)) = best else { break };
            if bv - second < eps {
                return None;
            }
            order.push(b);
            alive[b] = false;
            for i in (0..pool.len()).filter(|&i| alive[i]) {
                max_sim[i] = max_sim[i].max(dot64(&pool[i].0.vector, &pool[b].0.vector));
            }
        }
        let mut ids = Vec::new();
        let mut text = String::new();
        let mut used = 0;
        for j in order {
            let t = self.chk(pool[j].0, now);
            let line = format!("[{} {}] {}　《id:{}》\n", t.created_at, t.tz, t.text, t.id);
            let cost = line.chars().count();
            if !text.is_empty() && used + cost > cfg.budget_chars {
                continue;
            }
            text.push_str(&line);
            used += cost;
            ids.push(t.id.clone());
        }
        Some((ids, text))
    }
}

/// Finds, for each known kind, the smallest replayable (seed, kept ops) the harness can shrink to.
/// `cargo test --test audit2_d find_minimal_known -- --ignored --nocapture`
#[test]
#[ignore]
fn find_minimal_known() {
    for kind in KNOWN {
        let mut best: Option<(u64, Vec<usize>)> = None;
        for seed in 0..60u64 {
            let scn = gen_scenario(seed, 900);
            if let Some(f) = run_mode(&scn, &all_idx(&scn), true).fail.filter(|f| f.kind == *kind) {
                let min = shrink_mode(&scn, &f, true);
                if best.as_ref().is_none_or(|b| min.len() < b.1.len()) {
                    best = Some((seed, min));
                }
            }
        }
        eprintln!("MINIMAL {kind}: {best:?}");
    }
}

/// Replays a shrunk failure in strict mode (known kinds are fatal) and demands it passes.
#[allow(dead_code)]
fn strict_replay_must_pass(seed: u64, keep: &[usize], kind: &str) {
    let scn = gen_scenario(seed, 900);
    let out = run_mode(&scn, keep, true);
    if let Some(f) = out.fail {
        panic!("replay(seed={seed}, keep={keep:?}) fails [{}] at step {}: {}", f.kind, f.step, f.msg);
    }
    let _ = kind;
}

// Fuzz-found, delta-debugged replays (the generator is frozen: do not edit `gen_*` without redoing these).

#[test]
fn r2d_replay_seed52_dream_skips_seed_after_backward_clock() {
    strict_replay_must_pass(52, &[177, 186, 198, 210], "I7-seed-skipped-future-stamp");
}

#[test]
fn r2d_replay_seed2_gist_stated_time_clamped_after_backward_clock() {
    strict_replay_must_pass(2, &[94, 103, 114, 117, 121], "I7-gist-created-at-clamped");
}

#[test]
fn r2d_replay_seed43_remember_err_but_applied() {
    strict_replay_must_pass(43, &[232, 305, 306, 307, 308, 314, 316, 317, 318, 319, 322, 323, 324, 325, 326, 327, 328, 329], "I-err-but-applied");
}
