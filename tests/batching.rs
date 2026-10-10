//! Cue embeddings are batched: clusters() and dream() embed the distinct non-empty cues of
//! their whole seed list in one call, and clusters() returns exactly what one embedding and
//! one full scan per seed would.

mod support;

use support::rows;
use engram::{Engram, Store, Trace};
use std::sync::{Arc, Mutex};
use support::{build_with, cfg, merge_to_gist, CountingEmbedder, FakeEmbedder};

const WORDS: [&str; 6] = ["kyoto", "trip", "hotel", "train", "temple", "autumn"];

type Batches = Arc<Mutex<Vec<Vec<String>>>>;

fn related(offline_queries: bool, bad_cue: Option<&str>) -> (Engram, Batches) {
    let (embedder, batches) = CountingEmbedder::new(offline_queries, bad_cue);
    let (mut s, clock) = build_with(cfg(), Box::new(embedder), Box::new(Store::in_memory().unwrap()));
    for i in 0..40 {
        clock.advance(3600.0);
        let cue = if i % 4 == 0 { String::new() } else { format!("where {} {}", WORDS[i % 3], WORDS[(i + 3) % 6]) };
        s.remember(&format!("{} {} {} n{i}", WORDS[i % 6], WORDS[(i + 1) % 6], WORDS[(i + 2) % 6]), 1.0, &cue).unwrap();
    }
    batches.lock().unwrap().clear();
    (s, batches)
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).fold(0.0, |s, (&x, &y)| s + f64::from(x) * f64::from(y))
}

/// The cluster definition computed directly: one embedding and one full scan per seed.
fn reference(s: &Engram, cue_vector: impl Fn(&Trace) -> Vec<f32>, budget: usize) -> Vec<Vec<String>> {
    let c = s.config();
    let rows = rows(s);
    let mut out = Vec::new();
    for seed in rows.iter().filter(|m| !m.consolidated).take(8 * budget) {
        let q = cue_vector(seed);
        let mut near: Vec<(f64, usize)> = rows
            .iter()
            .enumerate()
            .filter(|(_, m)| m.id != seed.id && m.created_at <= seed.created_at)
            .map(|(i, m)| (dot(&m.vector, &q), i))
            .filter(|n| n.0 >= c.theta_related)
            .collect();
        near.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut cluster = vec![seed.text.clone()];
        cluster.extend(near.iter().take(c.dream_max_members - 1).map(|&(_, i)| rows[i].text.clone()));
        if cluster.len() >= 2 {
            out.push(cluster);
        }
    }
    out
}

fn texts(clusters: Vec<Vec<Trace>>) -> Vec<Vec<String>> {
    clusters.into_iter().map(|c| c.into_iter().map(|t| t.text).collect()).collect()
}

/// The engine's normalisation of a fake cue vector (renormalised in f64, like every vector).
fn cue_vec(fake: &Mutex<FakeEmbedder>, cue: &str) -> Vec<f32> {
    let v = fake.lock().unwrap().embed(cue);
    let n = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
    v.iter().map(|&x| (f64::from(x) / n) as f32).collect()
}

#[test]
fn clusters_embed_every_cue_in_one_call() {
    let (mut s, batches) = related(false, None);
    let mut cues: Vec<String> = Vec::new();
    for m in rows(&s).iter().filter(|m| !m.consolidated).take(80) {
        if !m.cue.is_empty() && !cues.contains(&m.cue) {
            cues.push(m.cue.clone());
        }
    }
    assert!(s.stats().unwrap().labile > 10);
    let got = texts(s.clusters(Some(10)).unwrap());
    assert_eq!(*batches.lock().unwrap(), vec![cues]);
    let fake = Mutex::new(FakeEmbedder::new());
    assert!(!got.is_empty());
    assert_eq!(got, reference(&s, |m| if m.cue.is_empty() { m.vector.clone() } else { cue_vec(&fake, &m.cue) }, 10));
}

#[test]
fn dream_embeds_every_cue_in_one_call() {
    let (mut s, batches) = related(false, None);
    assert!(!s.dream(&mut merge_to_gist, Some(10)).unwrap().is_empty());
    let b = batches.lock().unwrap();
    assert_eq!(b.len(), 1);
    let distinct: std::collections::HashSet<&String> = b[0].iter().collect();
    assert_eq!(distinct.len(), b[0].len());
}

#[test]
fn no_cues_no_query_call() {
    let (embedder, batches) = CountingEmbedder::new(false, None);
    let (mut s, _) = build_with(cfg(), Box::new(embedder), Box::new(Store::in_memory().unwrap()));
    for t in ["alpha beta gamma", "alpha beta delta", "alpha beta"] {
        s.remember(t, 1.0, "").unwrap();
    }
    batches.lock().unwrap().clear();
    assert!(!s.clusters(None).unwrap().is_empty());
    s.dream(&mut merge_to_gist, None).unwrap();
    assert!(batches.lock().unwrap().is_empty());
}

#[test]
fn offline_queries_fall_back_to_own_vectors() {
    let (mut s, batches) = related(true, None);
    let got = texts(s.clusters(Some(10)).unwrap());
    assert_eq!(batches.lock().unwrap().len(), 1);
    assert_eq!(got, reference(&s, |m| m.vector.clone(), 10));
}

#[test]
fn wrong_dimension_cue_falls_back_for_that_cue_only() {
    let bad = "where kyoto train";
    let (mut s, _) = related(false, Some(bad));
    assert!(rows(&s).iter().any(|m| m.cue == bad && !m.consolidated));
    let fake = Mutex::new(FakeEmbedder::new());
    let want = reference(&s, |m| if m.cue.is_empty() || m.cue == bad { m.vector.clone() } else { cue_vec(&fake, &m.cue) }, 10);
    assert_eq!(texts(s.clusters(Some(10)).unwrap()), want);
}
