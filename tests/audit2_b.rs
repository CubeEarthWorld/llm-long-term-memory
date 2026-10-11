//! Round-2 audit, lane B: the Storage contract (SPEC §2) and the bundled Store.
//!
//! `conformance::run` is a reusable suite: give it any `Storage` (plus a way to reopen it)
//! and it drives random operation sequences, comparing every view, count and near() result
//! with an oracle (`Model`, plain data). It is run against the bundled `Store` and against
//! `RefStorage`, a trivially-correct in-memory implementation written here (the sanity check
//! of the suite itself). The remaining tests are targeted reproductions of defects.

use redb::ReadableTable;
use engram::{Hit, Result, Storage, Store, Trace, View, Write};
use std::collections::BTreeMap;

mod support;
use support::temp_dir;

// ---------------------------------------------------------------- rng / data

pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    pub fn unit_f(&mut self) -> f32 {
        (self.next() >> 11) as f32 / (1u64 << 53) as f32 - 0.5
    }
    pub fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len() as u64) as usize]
    }
}

fn norm(v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
    if n == 0.0 { v } else { v.into_iter().map(|x| (f64::from(x) / n) as f32).collect() }
}

fn rand_unit(r: &mut Rng, dim: usize) -> Vec<f32> {
    norm((0..dim).map(|_| r.unit_f()).collect())
}

fn dot64(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum()
}

fn base(id: &str) -> Trace {
    Trace {
        id: id.into(),
        text: format!("t-{id}"),
        created_at: 1_000,
        tz: "UTC;+00:00".into(),
        last_recall: 2_000,
        stability: 4096.0,
        consolidated: true,
        cue: String::new(),
        model_id: String::new(),
        vector: Vec::new(),
    }
}

fn ids(v: &[Trace]) -> Vec<String> {
    v.iter().map(|t| t.id.clone()).collect()
}

// ---------------------------------------------------------------- oracle + reference

#[derive(Clone, Default)]
pub struct Model {
    pub rows: BTreeMap<String, Trace>,
}

impl Model {
    fn valid(t: &Trace) -> bool {
        t.stability.is_finite() && t.vector.iter().all(|x| x.is_finite())
    }

    /// Applies a batch atomically: `Err` leaves the model unchanged.
    pub fn apply(&mut self, ws: &[Write<'_>]) -> std::result::Result<(), String> {
        let mut m = self.clone();
        for w in ws {
            match *w {
                Write::Insert(t) => {
                    if !Self::valid(t) {
                        return Err("non-finite".into());
                    }
                    m.rows.insert(t.id.clone(), t.clone());
                }
                Write::State { id, last_recall, stability, consolidated } => {
                    if !stability.is_finite() {
                        return Err("non-finite".into());
                    }
                    if let Some(t) = m.rows.get_mut(id) {
                        (t.last_recall, t.stability, t.consolidated) = (last_recall, stability, consolidated);
                    }
                }
                Write::Index { id, model_id, vector } => {
                    if !vector.iter().all(|x| x.is_finite()) {
                        return Err("non-finite".into());
                    }
                    if let Some(t) = m.rows.get_mut(id) {
                        t.model_id = model_id.into();
                        t.vector = vector.to_vec();
                    }
                }
                Write::Remove(id) => {
                    m.rows.remove(id);
                }
            }
        }
        *self = m;
        Ok(())
    }

    pub fn view(&self, v: View<'_>) -> Vec<Trace> {
        let mut out: Vec<Trace> = self
            .rows
            .values()
            .filter(|t| match v {
                View::All => true,
                View::After(after) => t.id.as_str() > after,
                View::Id(id) => t.id == id,
                View::Text(x) => t.text == x,
                View::Labile { model_id, after } => !t.consolidated && t.model_id == model_id && t.id.as_str() > after,
                View::Stale { model_id } => t.model_id != model_id,
                View::Band { band, created_by } => t.band() == band && t.created_at <= created_by,
                View::Young { created_after } => t.created_at > created_after,
            })
            .cloned()
            .collect();
        match v {
            View::Band { .. } => out.sort_by(|a, b| (a.last_recall, &a.id).cmp(&(b.last_recall, &b.id))),
            View::Young { .. } => out.sort_by(|a, b| (a.created_at, &a.id).cmp(&(b.created_at, &b.id))),
            View::Stale { .. } => out.sort_by(|a, b| (&b.model_id, &b.id).cmp(&(&a.model_id, &a.id))),
            _ => {}
        }
        out
    }

    /// Every candidate of a search with its exact cosine, best first.
    pub fn ranked(&self, q: &[f32], model: &str, stated_by: Option<i64>) -> Vec<(f64, Trace)> {
        let mut c: Vec<(f64, Trace)> = self
            .rows
            .values()
            .filter(|t| t.model_id == model && !t.vector.is_empty() && t.vector.len() == q.len())
            .filter(|t| stated_by.is_none_or(|s| t.created_at <= s))
            .map(|t| (dot64(&t.vector, q) + 0.0, t.clone())) // + 0.0: -0.0 is the same cosine as 0.0
            .collect();
        c.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.id.cmp(&b.1.id)));
        c
    }
}

/// The trivially-correct reference implementation of `Storage`.
#[derive(Default)]
pub struct RefStorage(pub Model);

impl Storage for RefStorage {
    fn find(&self, view: View<'_>, limit: usize) -> Result<Vec<Trace>> {
        Ok(self.0.view(view).into_iter().take(limit).collect())
    }
    fn count(&self, view: View<'_>) -> Result<usize> {
        Ok(self.0.view(view).len())
    }
    fn near(&self, q: &[f32], model_id: &str, min_cos: f64, k: usize, stated_by: Option<i64>) -> Result<Vec<Hit>> {
        Ok(self.0.ranked(q, model_id, stated_by).into_iter().filter(|c| c.0 >= min_cos).take(k).map(|(cos, trace)| Hit { cos, trace }).collect())
    }
    fn write(&mut self, writes: &[Write<'_>]) -> Result<()> {
        self.0.apply(writes).map_err(|e| e.into())
    }
    fn clear(&mut self) -> Result<()> {
        self.0.rows.clear();
        Ok(())
    }
}

// ---------------------------------------------------------------- the reusable suite

pub mod conformance {
    use super::*;

    const EPS: f64 = 2e-5;

    /// Compares every view / count / near of `s` with the oracle. `what` labels failures.
    pub fn check_all<S: Storage>(s: &S, m: &Model, r: &mut Rng, what: &str) {
        let models = ["m1", "m2", "nope", ""];
        let limits = [0usize, 1, 3, usize::MAX];
        let check_view = |v: View<'_>, exact_order: bool| {
            let want = m.view(v);
            assert_eq!(s.count(v).unwrap(), want.len(), "{what}: count({v:?})");
            for &l in &limits {
                let got = s.find(v, l).unwrap();
                let exp: Vec<Trace> = want.iter().take(l).cloned().collect();
                if exact_order {
                    assert_eq!(got, exp, "{what}: find({v:?}, {l})");
                } else {
                    // Stale: only each other model's newest-first order is specified.
                    assert_eq!(got.len(), exp.len(), "{what}: find({v:?}, {l}) length");
                    let mut seen: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
                    for t in &got {
                        seen.entry(&t.model_id).or_default().push(&t.id);
                        assert_eq!(Some(t), m.rows.get(&t.id), "{what}: stale row content");
                    }
                    for (model, got_ids) in &seen {
                        let mut all: Vec<&str> = m.rows.values().filter(|t| t.model_id == *model).map(|t| t.id.as_str()).collect();
                        all.reverse(); // newest first
                        assert_eq!(*got_ids, all[..got_ids.len()], "{what}: stale {model:?} order/prefix");
                    }
                }
            }
        };
        check_view(View::All, true);
        let mut texts: Vec<String> = m.rows.values().map(|t| t.text.clone()).collect();
        texts.push("absent".into());
        for t in texts.iter().take(6) {
            check_view(View::Text(t), true);
        }
        for id in m.rows.keys().take(4).cloned().chain(["zzz-missing".to_string()]) {
            check_view(View::Id(&id), true);
        }
        let mut cuts: Vec<i64> = vec![i64::MIN, i64::MAX, 0, 1_000, 1_500, 5_000];
        cuts.extend(m.rows.values().take(5).flat_map(|t| [t.created_at, t.created_at - 1, t.created_at.saturating_add(1)]));
        let bands: Vec<u8> = {
            let mut b: Vec<u8> = m.rows.values().map(|t| t.band()).collect();
            b.sort();
            b.dedup();
            b.push(0);
            b.push(255);
            b
        };
        for &model in &models {
            let mut after = String::new();
            // Labile paging: the pages concatenated must equal the whole queue, with no dupes.
            check_view(View::Labile { model_id: model, after: "" }, true);
            let mut walked = Vec::new();
            loop {
                let page = s.find(View::Labile { model_id: model, after: &after }, 3).unwrap();
                if page.is_empty() {
                    break;
                }
                after = page.last().unwrap().id.clone();
                walked.extend(page);
            }
            assert_eq!(ids(&walked), ids(&m.view(View::Labile { model_id: model, after: "" })), "{what}: Labile paging {model:?}");
            for t in m.rows.values().take(3) {
                check_view(View::Labile { model_id: model, after: &t.id }, true);
            }
            check_view(View::Stale { model_id: model }, false);
        }
        for &b in &bands {
            for &c in &cuts {
                check_view(View::Band { band: b, created_by: c }, true);
            }
        }
        for &c in &cuts {
            check_view(View::Young { created_after: c }, true);
        }
        // near
        let dims = [4usize, 5, 6, 7];
        for _ in 0..24 {
            let model = *r.pick(&["m1", "m2", "nope"]);
            let dim = *r.pick(&dims);
            let q = match r.below(4) {
                0 => {
                    let cands: Vec<&Trace> = m.rows.values().filter(|t| t.vector.len() == dim).collect();
                    if cands.is_empty() { rand_unit(r, dim) } else { r.pick(&cands).vector.clone() }
                }
                1 => {
                    let cands: Vec<&Trace> = m.rows.values().filter(|t| t.vector.len() == dim).collect();
                    if cands.is_empty() { rand_unit(r, dim) } else { r.pick(&cands).vector.iter().map(|x| -x).collect() }
                }
                _ => rand_unit(r, dim),
            };
            let min_cos = *r.pick(&[-2.0, -0.2, 0.0, 0.5, 0.9, 0.999]);
            let k = *r.pick(&[0usize, 1, 3, 10, 1000]);
            let stated_by = if r.below(2) == 0 { None } else { Some(*r.pick(&[900i64, 1_000, 1_100, 1_500, i64::MAX, i64::MIN])) };
            check_near(s, m, &q, model, min_cos, k, stated_by, what);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn check_near<S: Storage>(s: &S, m: &Model, q: &[f32], model: &str, min_cos: f64, k: usize, stated_by: Option<i64>, what: &str) {
        let ranked = m.ranked(q, model, stated_by);
        let hits = s.near(q, model, min_cos, k, stated_by).unwrap();
        let tag = format!("{what}: near(model={model:?}, dim={}, min={min_cos}, k={k}, by={stated_by:?})", q.len());
        assert!(hits.len() <= k, "{tag}: too many");
        let mut seen = std::collections::HashSet::new();
        for h in &hits {
            assert!(seen.insert(h.trace.id.clone()), "{tag}: duplicate {}", h.trace.id);
            let (rc, rt) = ranked.iter().find(|c| c.1.id == h.trace.id).unwrap_or_else(|| panic!("{tag}: {} is not a candidate", h.trace.id));
            assert!((rc - h.cos).abs() <= EPS, "{tag}: cos {} vs {}", h.cos, rc);
            assert!(h.cos >= min_cos - EPS, "{tag}: below min");
            assert_eq!(&h.trace, rt, "{tag}: row content of {}", h.trace.id);
        }
        for w in hits.windows(2) {
            assert!(w[0].cos > w[1].cos || (w[0].cos == w[1].cos && w[0].trace.id < w[1].trace.id), "{tag}: order {} {} / {} {}", w[0].cos, w[0].trace.id, w[1].cos, w[1].trace.id);
        }
        let strict = ranked.iter().filter(|c| c.0 >= min_cos + EPS).count();
        let loose = ranked.iter().filter(|c| c.0 >= min_cos - EPS).count();
        assert!(hits.len() >= strict.min(k) && hits.len() <= loose.min(k), "{tag}: {} hits, expected {}..={}", hits.len(), strict.min(k), loose.min(k));
        if let Some(cut) = hits.last().filter(|_| hits.len() == k).map(|h| h.cos) {
            for c in ranked.iter().filter(|c| c.0 > cut + 2.0 * EPS && c.0 >= min_cos) {
                assert!(seen.contains(&c.1.id), "{tag}: missed {} (cos {} > cutoff {})", c.1.id, c.0, cut);
            }
        }
    }

    /// A random vector for model `model`, in every adversarial shape.
    pub fn vector(r: &mut Rng, model: &str, pool: &[Vec<f32>]) -> Vec<f32> {
        let dim = match model {
            "m1" => *r.pick(&[4usize, 4, 4, 6]), // mixed dimensions under one model
            _ => 5,
        };
        let same: Vec<&Vec<f32>> = pool.iter().filter(|v| v.len() == dim).collect();
        match r.below(8) {
            0 if !same.is_empty() => r.pick(&same).to_vec(), // duplicate
            1 if !same.is_empty() => r.pick(&same).iter().map(|x| -x).collect(), // antipodal
            2 => rand_unit(r, dim).into_iter().map(|x| x * 1e-20).collect(), // near-zero norm
            3 => vec![0.0; dim],
            _ => rand_unit(r, dim),
        }
    }

    fn random_trace(r: &mut Rng, id: &str, pool: &mut Vec<Vec<f32>>) -> Trace {
        let model = *r.pick(&["", "m1", "m1", "m2"]);
        let vector = if model.is_empty() { Vec::new() } else { vector(r, model, pool) };
        if !vector.is_empty() {
            pool.push(vector.clone());
        }
        Trace {
            id: id.into(),
            text: format!("text {}", r.below(12)),
            created_at: *r.pick(&[900i64, 1_000, 1_000, 1_100, 1_500, 5_000]),
            last_recall: *r.pick(&[10i64, 2_000, 2_000, 2_010, 3_000, -5]),
            stability: *r.pick(&[1.0f64, 1.5, 2.0, 700.0, 4096.0, 4096.0, 5800.0, 1e6, 1e9, 0.3]),
            consolidated: r.below(3) == 0,
            model_id: model.into(),
            vector,
            ..base(id)
        }
    }

    /// Random operation sequences with a reopen now and then. Returns the final storage.
    pub fn run<S: Storage>(mut s: S, reopen: &dyn Fn(S) -> S, seed: u64, batches: usize, what: &str) -> S {
        let mut r = Rng::new(seed);
        let mut m = Model::default();
        let mut pool: Vec<Vec<f32>> = Vec::new();
        let id_of = |n: u64| format!("id{n:03}");
        for step in 0..batches {
            let n_ops = 1 + r.below(6) as usize;
            let mut owned: Vec<Trace> = Vec::new();
            let mut plan: Vec<(u8, String, Trace)> = Vec::new();
            let poison = r.below(7) == 0;
            for _ in 0..n_ops {
                let id = id_of(r.below(40));
                match r.below(10) {
                    0..=3 => plan.push((0, id.clone(), random_trace(&mut r, &id, &mut pool))),
                    4..=5 => plan.push((1, id, base("x"))),
                    6..=7 => plan.push((2, id.clone(), random_trace(&mut r, &id, &mut pool))),
                    _ => plan.push((3, id, base("x"))),
                }
            }
            // `plan` borrowed into Writes
            owned.extend(plan.iter().map(|p| p.2.clone()));
            let mut bad = Trace { stability: f64::NAN, ..base(&id_of(r.below(40))) };
            if r.below(2) == 0 {
                bad = Trace { stability: 1.0, model_id: "m1".into(), vector: vec![f32::NAN; 4], ..bad };
            }
            let mut ws: Vec<Write<'_>> = Vec::new();
            for (i, (kind, id, _)) in plan.iter().enumerate() {
                ws.push(match kind {
                    0 => Write::Insert(&owned[i]),
                    1 => Write::State { id, last_recall: *r.pick(&[10i64, 2_000, 2_500, 3_000]), stability: *r.pick(&[1.0f64, 4096.0, 5800.0, 2e7]), consolidated: r.below(2) == 0 },
                    2 => {
                        let t = &owned[i];
                        Write::Index { id, model_id: &t.model_id, vector: &t.vector }
                    }
                    _ => Write::Remove(id),
                });
            }
            if poison {
                let at = r.below(ws.len() as u64 + 1) as usize;
                ws.insert(at, Write::Insert(&bad));
            }
            let want = m.apply(&ws);
            let got = s.write(&ws);
            assert_eq!(want.is_ok(), got.is_ok(), "{what} step {step}: write result {got:?} (poison={poison})");
            if poison {
                assert!(got.is_err(), "{what} step {step}: poisoned batch accepted");
            }
            if step % 4 == 3 {
                check_all(&s, &m, &mut r, &format!("{what} step {step}"));
            }
            if step % 11 == 10 {
                s = reopen(s);
                check_all(&s, &m, &mut r, &format!("{what} step {step} (reopened)"));
            }
            if step % 37 == 36 && r.below(2) == 0 {
                s.clear().unwrap();
                m.rows.clear();
            }
        }
        check_all(&s, &m, &mut r, &format!("{what} final"));
        s
    }
}

#[test]
fn reference_storage_passes_its_own_suite() {
    for seed in 1..=6 {
        conformance::run(RefStorage::default(), &|s| s, seed, 120, &format!("ref seed {seed}"));
    }
}

#[test]
fn store_passes_the_suite_in_memory() {
    for seed in 1..=8 {
        conformance::run(Store::in_memory().unwrap(), &|s| s, seed, 150, &format!("store seed {seed}"));
    }
}

/// `AUDIT2B_SEEDS=n` widens the search (default 3 extra seeds).
#[test]
fn store_passes_the_suite_on_more_seeds() {
    let n: u64 = std::env::var("AUDIT2B_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    for seed in 100..100 + n {
        conformance::run(Store::in_memory().unwrap(), &|s| s, seed, 200, &format!("store wide seed {seed}"));
    }
}

#[test]
fn store_passes_the_suite_across_reopens() {
    for seed in 11..=14 {
        let path = temp_dir("audit2b-reopen").join("s.redb");
        let p2 = path.clone();
        conformance::run(
            Store::open(&path).unwrap(),
            &move |s| {
                drop(s);
                Store::open(&p2).unwrap()
            },
            seed,
            150,
            &format!("store-file seed {seed}"),
        );
    }
}

/// A deliberately wrong storage proves the suite can fail: Band ignores the created_by filter.
#[test]
fn the_suite_detects_a_broken_band_filter() {
    struct Broken(RefStorage);
    impl Storage for Broken {
        fn find(&self, v: View<'_>, l: usize) -> Result<Vec<Trace>> {
            let v = if let View::Band { band, .. } = v { View::Band { band, created_by: i64::MAX } } else { v };
            self.0.find(v, l)
        }
        fn count(&self, v: View<'_>) -> Result<usize> {
            self.0.count(v)
        }
        fn near(&self, q: &[f32], m: &str, c: f64, k: usize, s: Option<i64>) -> Result<Vec<Hit>> {
            self.0.near(q, m, c, k, s)
        }
        fn write(&mut self, w: &[Write<'_>]) -> Result<()> {
            self.0.write(w)
        }
        fn clear(&mut self) -> Result<()> {
            self.0.clear()
        }
    }
    let r = std::panic::catch_unwind(|| conformance::run(Broken(RefStorage::default()), &|s| s, 3, 60, "broken"));
    assert!(r.is_err(), "the suite failed to notice a Band filter that is ignored");
}

// ---------------------------------------------------------------- IVF: near() against brute force

fn bulk_check(st: &mut Store, rows: Vec<Trace>, label: &str, seed: u64) {
    let mut m = Model::default();
    for chunk in rows.chunks(400) {
        let ws: Vec<Write<'_>> = chunk.iter().map(Write::Insert).collect();
        m.apply(&ws).unwrap();
        st.write(&ws).unwrap();
    }
    let mut r = Rng::new(seed);
    // churn: remove some, replace some (different vector / model), re-index some
    let keys: Vec<String> = m.rows.keys().cloned().collect();
    let mut replaced = Vec::new();
    for (i, id) in keys.iter().enumerate() {
        match i % 7 {
            0 => {
                let ws = [Write::Remove(id)];
                m.apply(&ws).unwrap();
                st.write(&ws).unwrap();
            }
            3 => {
                let old = &m.rows[id];
                let mut t = old.clone();
                t.vector = norm(old.vector.iter().map(|x| -x + 0.01 * r.unit_f()).collect());
                t.created_at += 7;
                replaced.push(t);
            }
            _ => {}
        }
    }
    for chunk in replaced.chunks(300) {
        let ws: Vec<Write<'_>> = chunk.iter().map(Write::Insert).collect();
        m.apply(&ws).unwrap();
        st.write(&ws).unwrap();
    }
    let pool: Vec<&Trace> = m.rows.values().collect();
    for j in 0..60 {
        let model = if j % 5 == 4 { "m2" } else { "m1" };
        let seedv = pool[r.below(pool.len() as u64) as usize];
        let dim = if j % 3 == 0 { seedv.vector.len() } else { 8 };
        let q = if j % 2 == 0 && seedv.vector.len() == dim {
            seedv.vector.clone()
        } else if j % 6 == 1 {
            rand_unit(&mut r, dim)
        } else {
            norm(seedv.vector.iter().map(|x| *x + 0.05 * r.unit_f()).chain(std::iter::repeat_n(0.1, dim.saturating_sub(seedv.vector.len()))).take(dim).collect())
        };
        let by = if j % 4 == 0 { Some(1_000 + (j as i64 % 13) * 100) } else { None };
        conformance::check_near(st, &m, &q, model, *r.pick(&[-2.0, 0.0, 0.7, 0.95]), *r.pick(&[1usize, 10, 64]), by, label);
    }
    // And the views still agree after all that splitting.
    conformance::check_all(st, &m, &mut r, label);
}

fn clustered(n: usize, dim: usize, centers: usize, spread: f32, seed: u64) -> Vec<Trace> {
    let mut r = Rng::new(seed);
    let cs: Vec<Vec<f32>> = (0..centers).map(|_| rand_unit(&mut r, dim)).collect();
    (0..n)
        .map(|i| {
            let c = &cs[(r.below(centers as u64)) as usize];
            Trace {
                model_id: "m1".into(),
                vector: norm(c.iter().map(|x| x + spread * r.unit_f()).collect()),
                created_at: 1_000 + (i as i64 % 13) * 100,
                stability: 2f64.powi((i % 9) as i32 + 10),
                last_recall: 2_000 + (i as i64 % 5) * 10,
                consolidated: i % 3 != 0,
                ..base(&format!("c{i:05}"))
            }
        })
        .collect()
}

#[test]
fn near_is_exact_on_clustered_data_below_8192() {
    let mut st = Store::in_memory().unwrap();
    bulk_check(&mut st, clustered(3_000, 8, 12, 0.08, 1), "clustered", 1);
}

#[test]
fn near_is_exact_on_duplicates_and_near_duplicates() {
    let mut r = Rng::new(2);
    let a = rand_unit(&mut r, 8);
    let b = rand_unit(&mut r, 8);
    let mut rows = clustered(600, 8, 5, 0.05, 3);
    for (i, t) in rows.iter_mut().enumerate() {
        match i % 4 {
            0 => t.vector = a.clone(),
            1 => t.vector = b.clone(),
            2 => t.vector = norm(a.iter().map(|x| x + 1e-4 * r.unit_f()).collect()),
            _ => {}
        }
    }
    rows.extend(clustered(1_200, 8, 1, 0.0, 4).into_iter().map(|t| Trace { id: format!("d{}", t.id), vector: a.clone(), ..t }));
    let mut st = Store::in_memory().unwrap();
    bulk_check(&mut st, rows, "duplicates", 2);
}

#[test]
fn near_is_exact_on_antipodal_and_near_zero_vectors() {
    let mut r = Rng::new(5);
    let mut rows = clustered(1_500, 8, 3, 0.02, 6);
    for (i, t) in rows.iter_mut().enumerate() {
        match i % 5 {
            0 => t.vector = t.vector.iter().map(|x| -x).collect(),
            1 => t.vector = t.vector.iter().map(|x| x * 1e-20).collect(),
            2 => t.vector = vec![0.0; 8],
            3 => t.vector = norm((0..8).map(|d| if d == 0 { 1.0 } else { 1e-3 * r.unit_f() }).collect()),
            _ => {}
        }
    }
    let mut st = Store::in_memory().unwrap();
    bulk_check(&mut st, rows, "antipodal/zero", 5);
}

#[test]
fn near_is_exact_with_two_models_and_mixed_dimensions() {
    let mut rows = clustered(1_400, 8, 6, 0.1, 7);
    for (i, t) in rows.iter_mut().enumerate() {
        if i % 3 == 0 {
            t.model_id = "m2".into();
        }
        if i % 5 == 0 {
            let mut r = Rng::new(i as u64 + 99);
            t.vector = rand_unit(&mut r, 6); // another dimension under the same model
        }
    }
    let mut st = Store::in_memory().unwrap();
    bulk_check(&mut st, rows, "two models / mixed dims", 7);
}

#[test]
fn near_is_exact_at_exactly_8192_vectors() {
    let mut st = Store::in_memory().unwrap();
    let rows = clustered(8_192, 6, 40, 0.15, 9);
    let mut m = Model::default();
    for chunk in rows.chunks(1_024) {
        let ws: Vec<Write<'_>> = chunk.iter().map(Write::Insert).collect();
        m.apply(&ws).unwrap();
        st.write(&ws).unwrap();
    }
    let mut r = Rng::new(10);
    for _ in 0..30 {
        let q = rand_unit(&mut r, 6);
        conformance::check_near(&st, &m, &q, "m1", -2.0, 20, None, "8192");
        conformance::check_near(&st, &m, &q, "m1", 0.9, 50, Some(1_500), "8192 by");
    }
}

// ---------------------------------------------------------------- targeted reproductions

#[test]
fn reinsert_changing_model_vector_band_and_text_leaves_no_trace_of_the_old_row() {
    let mut st = Store::in_memory().unwrap();
    let mut r = Rng::new(1);
    let old = Trace { model_id: "m1".into(), vector: rand_unit(&mut r, 4), consolidated: false, text: "old".into(), stability: 8.0, ..base("a") };
    st.write(&[Write::Insert(&old)]).unwrap();
    let new = Trace { model_id: "m2".into(), vector: rand_unit(&mut r, 5), consolidated: true, text: "new".into(), stability: 1e7, created_at: 77, last_recall: 5, ..base("a") };
    st.write(&[Write::Insert(&new)]).unwrap();
    assert_eq!(st.count(View::All).unwrap(), 1);
    assert_eq!(st.count(View::Text("old")).unwrap(), 0);
    assert_eq!(st.count(View::Labile { model_id: "m1", after: "" }).unwrap(), 0);
    assert_eq!(st.count(View::Band { band: old.band(), created_by: i64::MAX }).unwrap(), 0);
    assert_eq!(st.count(View::Young { created_after: 1_000 }).unwrap(), 0);
    assert!(st.near(&old.vector, "m1", -2.0, 5, None).unwrap().is_empty());
    assert_eq!(st.near(&new.vector, "m2", 0.99, 5, None).unwrap()[0].trace, new);
    assert_eq!(st.find(View::Stale { model_id: "m2" }, 5).unwrap().len(), 0);
    assert_eq!(st.find(View::Stale { model_id: "m1" }, 5).unwrap().len(), 1);
}

#[test]
fn a_rejected_insert_over_an_existing_id_changes_nothing() {
    let mut st = Store::in_memory().unwrap();
    let mut r = Rng::new(2);
    let old = Trace { model_id: "m1".into(), vector: rand_unit(&mut r, 4), consolidated: false, ..base("a") };
    let other = Trace { model_id: "m1".into(), vector: rand_unit(&mut r, 4), ..base("b") };
    st.write(&[Write::Insert(&old), Write::Insert(&other)]).unwrap();
    let bad = Trace { stability: f64::NAN, model_id: "m2".into(), vector: vec![0.5; 4], ..base("a") };
    let bad_vec = Trace { model_id: "m2".into(), vector: vec![f32::NAN; 4], ..base("a") };
    for b in [&bad, &bad_vec] {
        assert!(st.write(&[Write::Remove("b"), Write::Insert(b)]).is_err());
        assert!(st.write(&[Write::Insert(&base("zz")), Write::State { id: "a", last_recall: 1, stability: f64::INFINITY, consolidated: true }]).is_err());
        assert!(st.write(&[Write::Remove("b"), Write::Index { id: "a", model_id: "m2", vector: &[f32::NEG_INFINITY; 4] }]).is_err());
    }
    assert_eq!(st.find(View::All, 10).unwrap(), vec![old.clone(), other.clone()]);
    assert_eq!(st.count(View::All).unwrap(), 2);
    assert_eq!(st.near(&old.vector, "m1", 0.99, 5, None).unwrap().len(), if old.vector == other.vector { 2 } else { 1 });
}

/// SPEC §2: Young{created_after} is "stated after created_after". Nothing is after i64::MAX.
#[test]
fn young_after_i64_max_is_empty() {
    let mut st = Store::in_memory().unwrap();
    st.write(&[Write::Insert(&Trace { created_at: i64::MAX, ..base("a") }), Write::Insert(&Trace { created_at: i64::MAX - 1, ..base("b") })]).unwrap();
    assert_eq!(st.find(View::Young { created_after: i64::MAX - 1 }, 5).unwrap().len(), 1);
    assert_eq!(st.find(View::Young { created_after: i64::MAX }, 5).unwrap().len(), 0, "a trace stated AT i64::MAX is not after i64::MAX");
    assert_eq!(st.count(View::Young { created_after: i64::MAX }).unwrap(), 0);
}

/// Rows with created_at > created_by in the middle of a band neither end the scan nor hide the rows behind them.
#[test]
fn band_scan_skips_late_rows_in_the_middle() {
    let mut st = Store::in_memory().unwrap();
    let rows: Vec<Trace> = (0..30).map(|i| Trace { last_recall: i, created_at: if i % 3 == 1 { 9_000 } else { 100 }, ..base(&format!("r{i:02}")) }).collect();
    st.write(&rows.iter().map(Write::Insert).collect::<Vec<_>>()).unwrap();
    let b = rows[0].band();
    let got = st.find(View::Band { band: b, created_by: 500 }, 7).unwrap();
    assert_eq!(ids(&got), ["r00", "r02", "r03", "r05", "r06", "r08", "r09"]);
    assert_eq!(st.count(View::Band { band: b, created_by: 500 }).unwrap(), 20);
    // other bands bound the scan
    let other = Trace { stability: 1e9, ..base("zz") };
    st.write(&[Write::Insert(&other)]).unwrap();
    assert_eq!(st.count(View::Band { band: b, created_by: i64::MAX }).unwrap(), 30);
}

fn tamper_rows(path: &std::path::Path, f: impl FnOnce(&mut redb::Table<&str, &[u8]>)) {
    let db = redb::Database::open(path).unwrap();
    let wx = db.begin_write().unwrap();
    {
        let mut rows = wx.open_table(redb::TableDefinition::<&str, &[u8]>::new("rows")).unwrap();
        f(&mut rows);
    }
    wx.commit().unwrap();
}

fn damaged_row_store(name: &str) -> (std::path::PathBuf, Vec<Trace>) {
    let path = temp_dir(name).join("s.redb");
    let rows: Vec<Trace> = (0..5).map(|i| Trace { model_id: "m".into(), vector: vec![1.0, 0.0, 0.0, 0.0], consolidated: false, ..base(&format!("r{i}")) }).collect();
    {
        let mut st = Store::open(&path).unwrap();
        st.write(&rows.iter().map(Write::Insert).collect::<Vec<_>>()).unwrap();
    }
    tamper_rows(&path, |t| {
        t.insert("r2", b"{not json".as_slice()).unwrap();
    });
    (path, rows)
}

/// SPEC §2/§4 forget: any row can be deleted by id; the store must not keep an unreadable row forever.
#[test]
fn a_row_with_unreadable_json_can_be_removed() {
    let (path, _) = damaged_row_store("a2b-rm-damaged");
    let mut st = Store::open(&path).unwrap();
    let n = st.count(View::All).unwrap();
    let res = st.write(&[Write::Remove("r2")]);
    assert!(res.is_ok(), "Remove of a damaged row fails ({:?}); the row, its index entries and its count are stuck", res.err());
    assert_eq!(st.count(View::All).unwrap(), n - 1);
    assert_eq!(st.count(View::Labile { model_id: "m", after: "" }).unwrap(), 4);
}

/// SPEC §2: re-writing the id (a fresh Insert) must also be able to repair the damaged row.
#[test]
fn a_row_with_unreadable_json_can_be_overwritten() {
    let (path, rows) = damaged_row_store("a2b-ow-damaged");
    let mut st = Store::open(&path).unwrap();
    let res = st.write(&[Write::Insert(&rows[2])]);
    assert!(res.is_ok(), "Insert over a damaged row fails ({:?})", res.err());
    assert_eq!(st.count(View::All).unwrap(), 5);
    assert_eq!(st.count(View::Labile { model_id: "m", after: "" }).unwrap(), 5);
}

/// count(view) must equal find(view, MAX).len() (or find must fail): a damaged row is counted
/// by every view but silently dropped from every find.
#[test]
fn count_and_find_agree_when_a_row_is_damaged() {
    let (path, _) = damaged_row_store("a2b-count-damaged");
    let st = Store::open(&path).unwrap();
    for v in [View::All, View::Id("r2"), View::Labile { model_id: "m", after: "" }, View::Stale { model_id: "x" }, View::Text("t-r2")] {
        let c = st.count(v).unwrap();
        match st.find(v, usize::MAX) {
            Ok(rows) => assert_eq!(rows.len(), c, "count({v:?}) = {c} but find returned {} rows without an error", rows.len()),
            Err(_) => {}
        }
    }
}

/// SPEC §2: damaged entries are errors, never panics. A list entry with the wrong number of
/// floats (but a valid header) used to reach `dot`'s length assert during a split.
#[test]
fn a_wrong_dimension_list_entry_does_not_panic_the_split() {
    let path = temp_dir("a2b-split-dim").join("s.redb");
    let mut r = Rng::new(3);
    let rows: Vec<Trace> = (0..512).map(|i| Trace { model_id: "m".into(), vector: rand_unit(&mut r, 8), ..base(&format!("r{i:04}")) }).collect();
    Store::open(&path).unwrap().write(&rows.iter().map(Write::Insert).collect::<Vec<_>>()).unwrap();
    {
        let db = redb::Database::open(&path).unwrap();
        let wx = db.begin_write().unwrap();
        {
            let mut lists = wx.open_table(redb::TableDefinition::<(&str, u32, &str), &[u8]>::new("lists")).unwrap();
            let mut entry = 5i64.to_le_bytes().to_vec();
            entry.extend([0u8; 12]); // 3 floats instead of 8
            let (m, l, id) = {
                let it = lists.iter().unwrap();
                let first = it.into_iter().next().unwrap().unwrap();
                let k = first.0.value();
                (k.0.to_owned(), k.1, k.2.to_owned())
            };
            lists.insert((m.as_str(), l, id.as_str()), entry.as_slice()).unwrap();
        }
        wx.commit().unwrap();
    }
    let mut st = Store::open(&path).unwrap();
    let extra = Trace { model_id: "m".into(), vector: rand_unit(&mut r, 8), ..base("r9999") };
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| st.write(&[Write::Insert(&extra)])));
    assert!(res.is_ok(), "the 513th insert panicked in split() on a damaged list entry");
}

#[test]
fn the_store_is_shareable_between_reader_threads() {
    fn assert_traits<T: Send + Sync>() {}
    assert_traits::<Store>();
    let mut st = Store::in_memory().unwrap();
    let mut r = Rng::new(4);
    let rows: Vec<Trace> = (0..900).map(|i| Trace { model_id: "m".into(), vector: rand_unit(&mut r, 8), ..base(&format!("r{i:04}")) }).collect();
    st.write(&rows.iter().map(Write::Insert).collect::<Vec<_>>()).unwrap();
    let st = std::sync::Arc::new(st);
    let handles: Vec<_> = (0..6)
        .map(|k| {
            let (st, rows) = (st.clone(), rows.clone());
            std::thread::spawn(move || {
                for i in (k..900).step_by(37) {
                    let h = st.near(&rows[i].vector, "m", 0.999, 3, None).unwrap();
                    assert_eq!(h[0].trace.id, rows[i].id);
                    assert_eq!(st.count(View::All).unwrap(), 900);
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

/// count() is the META counter: replacing, removing a missing id and re-inserting keep it equal to the row count.
#[test]
fn count_tracks_the_rows_through_replace_remove_and_reopen() {
    let path = temp_dir("a2b-count").join("s.redb");
    {
        let mut st = Store::open(&path).unwrap();
        let t = base("a");
        st.write(&[Write::Insert(&t), Write::Insert(&t), Write::Insert(&base("b")), Write::Remove("nope"), Write::Remove("b"), Write::Remove("b")]).unwrap();
        assert_eq!(st.count(View::All).unwrap(), 1);
        assert!(st.write(&[Write::Insert(&base("c")), Write::State { id: "a", last_recall: 0, stability: f64::NAN, consolidated: true }]).is_err());
        assert_eq!(st.count(View::All).unwrap(), 1);
    }
    let st = Store::open(&path).unwrap();
    assert_eq!(st.count(View::All).unwrap(), 1);
    assert_eq!(st.find(View::All, 9).unwrap().len(), 1);
}

/// `find` takes `usize::MAX` for "all"; `near`'s k is the same type, so "all" must work (not a capacity-overflow panic).
#[test]
fn near_accepts_k_of_usize_max() {
    let mut st = Store::in_memory().unwrap();
    st.write(&[Write::Insert(&Trace { model_id: "m".into(), vector: vec![1.0, 0.0], ..base("a") })]).unwrap();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| st.near(&[1.0, 0.0], "m", 0.0, usize::MAX, None).map(|h| h.len())));
    match r {
        Ok(n) => assert_eq!(n.unwrap(), 1),
        Err(_) => panic!("near(k = usize::MAX) panics (Vec::with_capacity(k))"),
    }
}
