//! Generality: does the engine depend on one embedding model or dimension? A
//! Matryoshka model cut to its first d components is "another model of another
//! dimension". For each d: the cosine distribution, recall quality and eviction
//! precision on the forgetting timeline, and the dream's cue capture — with the default
//! thresholds and with thresholds calibrated on that model (SPEC §6: unrelated p90 →
//! `cosine_floor`, p99 → `theta_related`).
//!
//! `ltm bench generality [--dims 768,512,256,128,64] [--model X.gguf --query-prefix … --document-prefix …]`

use super::forget::{run, Spec};
use super::supersede::cluster_capture;
use super::{bench_dir, build_cache, cache_path, load_corpus, load_updates, percentile, CachedEmbedder, Corpus, Flags, PyRandom, Truncated};
use crate::settings::AppConfig;
use engram::{cues, Config, Embedder, Result};
use serde_json::{json, Value};

/// Unrelated-pair cosines (the `cosine_floor` calibration point) and matching-pair cosines.
fn distribution(e: &mut dyn Embedder, corpus: &Corpus) -> Result<Value> {
    let pool: Vec<(&str, String)> = corpus.gold.iter().filter_map(|g| cues(&g.probes[0], 8).into_iter().next().map(|c| (g.fact.as_str(), c))).collect();
    let facts: Vec<&str> = pool.iter().map(|p| p.0).collect();
    let probes: Vec<&str> = pool.iter().map(|p| p.1.as_str()).collect();
    let norm = |vs: Vec<Vec<f32>>| vs.into_iter().map(|v| crate::normalized(&v)).collect::<Vec<_>>();
    let (d, q) = (norm(e.embed_documents(&facts)?), norm(e.embed_queries(&probes)?));
    let same: Vec<f64> = d.iter().zip(&q).map(|(a, b)| crate::dot(a, b)).collect();
    let mut perm: Vec<usize> = (0..facts.len()).collect();
    PyRandom::new(7).shuffle(&mut perm);
    let cross: Vec<f64> = perm.iter().enumerate().map(|(j, &i)| crate::dot(&d[if i == j { (i + 1) % d.len() } else { i }], &q[j])).collect();
    let pct = |v: &[f64], qs: &[f64]| qs.iter().map(|&p| crate::round_to(percentile(v, p), 3)).collect::<Vec<_>>();
    Ok(json!({
        "unrelated_p50_p90_p99": pct(&cross, &[50.0, 90.0, 99.0]),
        "matching_p10_p50_p90": pct(&same, &[10.0, 50.0, 90.0]),
        "separation": crate::round_to(percentile(&same, 50.0) - percentile(&cross, 90.0), 3),
    }))
}

/// Thresholds placed on this model's distribution (gist guard keeps EmbeddingGemma's
/// relative position between floor and θ).
fn calibrate(dist: &Value) -> Config {
    let p = |i: usize| dist["unrelated_p50_p90_p99"][i].as_f64().unwrap_or(0.0);
    let round2 = |x: f64| crate::round_to(x, 2);
    let floor = round2(p(1));
    let theta = round2(p(2)).max(floor + 0.01);
    Config { cosine_floor: floor, theta_related: theta, gist_min_cosine: round2(floor + (theta - floor) * 2.0 / 3.0), ..Config::default() }
}

pub fn main(args: &[String]) -> Result<()> {
    let f = Flags::parse(args);
    let corpus = load_corpus(&bench_dir().join("corpus"));
    let pairs = load_updates(&bench_dir().join("updates"));
    let mut app = AppConfig::default();
    if f.has("model") {
        app.glob.embedding_model = f.str("model", "");
        if f.has("query-prefix") {
            app.glob.embed_query_prefix = f.str("query-prefix", "");
        }
        if f.has("document-prefix") {
            app.glob.embed_document_prefix = f.str("document-prefix", "");
        }
    }
    let cache = cache_path(&app.glob.embedding_model);
    if !cache.exists() {
        build_cache(&corpus, &app, &cache)?; // embed the timeline's texts once per model
    }
    let base = CachedEmbedder::load(&cache)?;
    let dims: Vec<usize> = Some(f.list("dims")).filter(|d: &Vec<usize>| !d.is_empty()).unwrap_or_else(|| vec![base.dimension()]);
    let (span, capacity, seed) = (f.num("span-days", 365.0), f.num("capacity", 1000), f.num("seed", 1));
    let live = || -> Result<Box<dyn Embedder>> { Ok(crate::embedder(&app.glob)?.0) }; // updates and cues are not cached
    let cut = |e: Box<dyn Embedder>, d: usize| -> Box<dyn Embedder> { if d == e.dimension() { e } else { Box::new(Truncated::new(e, d)) } };
    println!("model={} dim={}", base.model_id(), base.dimension());
    println!("年表（既定閾値 0.4/0.55/0.5 と、そのモデルで較正した閾値）:");
    let mut rows = Vec::new();
    for &dim in &dims {
        let dist = distribution(cut(Box::new(base.clone()), dim).as_mut(), &corpus)?;
        let mut row = json!({"dimension": dim, "cosine": dist});
        for (label, cfg) in [("default", Config::default()), ("calibrated", calibrate(&dist))] {
            let spec = Spec { arm: "engram".into(), seed, span_days: span, capacity, rehearsed_frac: 0.5, salience: 1.0, dream: true };
            let r = run(&corpus, &spec, cfg.clone(), cut(Box::new(base.clone()), dim))?;
            let x = json!({
                "thresholds": [cfg.cosine_floor, cfg.theta_related, cfg.gist_min_cosine],
                "rehearsed": r["final"]["rehearsed"], "dormant_survival": r["final"]["dormant"]["survival"],
                "evict_precision": r["evict_precision"], "invariants": r["invariants"], "forget_api": r["forget_api"],
                "capture": cluster_capture(&pairs, cfg, cut(live()?, dim))?,
            });
            let bad: i64 = x["invariants"].as_object().map_or(0, |o| o.values().filter_map(Value::as_i64).sum());
            println!(
                "  d={dim:4} {label:10} floor/θ/gist={}  top1={:.3} hit5={:.3} surv={:.3} evict_prec={:.3} capture={:.0}% bad={bad}",
                x["thresholds"], x["rehearsed"]["top1"].as_f64().unwrap_or(0.0), x["rehearsed"]["hit@5"].as_f64().unwrap_or(0.0),
                x["rehearsed"]["survival"].as_f64().unwrap_or(0.0), x["evict_precision"].as_f64().unwrap_or(0.0),
                x["capture"]["rate"].as_f64().unwrap_or(0.0) * 100.0
            );
            row[label] = x;
        }
        println!("        unrelated p50/p90/p99={} match p10/p50/p90={} sep={}", row["cosine"]["unrelated_p50_p90_p99"], row["cosine"]["matching_p10_p50_p90"], row["cosine"]["separation"]);
        rows.push(row);
    }
    let stem = std::path::Path::new(base.model_id()).file_stem().map_or_else(String::new, |s| s.to_string_lossy().into_owned());
    let out = bench_dir().join(format!("generality_{stem}.json"));
    std::fs::write(&out, serde_json::to_string_pretty(&json!({"model_id": base.model_id(), "dimension": base.dimension(), "runs": rows}))?)?;
    println!("wrote {}", out.display());
    Ok(())
}
