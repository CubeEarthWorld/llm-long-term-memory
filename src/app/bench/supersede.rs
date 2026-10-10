//! Overwrite forgetting (SPEC §5 dream): when a fact is updated, does the old version
//! really go? Every `old` is written, 30 days pass, every `new` is written (each with
//! the cue its `save_memory` would carry), so the new side is labile and the dream
//! adjudicates. Only the verdict needs the real LLM.
//!
//! `ltm bench supersede --analyze`   cosine distribution and cluster capture, no LLM
//! `ltm bench supersede --run [--theta 0.65]`   live dream, stale rate (needs an API key)

use super::super::llm::client;
use super::super::settings::AppConfig;
use super::{bench_dir, load_updates, percentile, r4, Flags, Update, DAY};
use crate::{Config, DreamAction, Embedder, Engram, ManualClock, MemoryStore, Result};
use serde_json::{json, Value};
use std::collections::HashSet;

const GAP_DAYS: f64 = 30.0;

fn scenario(pairs: &[Update], cfg: Config, embedder: Box<dyn Embedder>) -> Result<(Engram, ManualClock)> {
    let clock = ManualClock::new(1_700_000_000.0);
    let mut m = Engram::with_clock(Box::new(MemoryStore::default()), embedder, cfg, "Asia/Tokyo", clock.clock())?;
    for p in pairs {
        clock.advance(600.0);
        m.remember(&p.old, 1.0, &p.cue_old)?;
    }
    clock.advance(GAP_DAYS * DAY);
    for p in pairs {
        clock.advance(600.0);
        m.remember(&p.new, 1.0, &p.cue_new)?;
    }
    Ok((m, clock))
}

/// How many pairs the dream's cue-driven clusters seat together (shared with `generality`).
pub fn cluster_capture(pairs: &[Update], cfg: Config, embedder: Box<dyn Embedder>) -> Result<Value> {
    let (mut m, _) = scenario(pairs, cfg, embedder)?;
    let clusters: Vec<HashSet<String>> = m.clusters(Some(pairs.len())).into_iter().map(|c| c.into_iter().map(|t| t.text).collect()).collect();
    let got = pairs.iter().filter(|p| clusters.iter().any(|c| c.contains(&p.old) && c.contains(&p.new))).count();
    Ok(json!({"captured": got, "pairs": pairs.len(), "rate": super::super::round_to(got as f64 / pairs.len() as f64, 3), "clusters": clusters.len()}))
}

/// The bottleneck of overwrite forgetting is capture, not the LLM: does a cue reach
/// the version it supersedes?
fn analyze(pairs: &[Update], mut cfg: AppConfig) -> Result<()> {
    let (mut e, _) = super::super::embedder(&cfg.glob)?;
    let olds: Vec<&str> = pairs.iter().map(|p| p.old.as_str()).collect();
    let news: Vec<&str> = pairs.iter().map(|p| p.new.as_str()).collect();
    let (a, b) = (e.embed_documents(&olds)?, e.embed_documents(&news)?);
    let cos: Vec<f64> = a.iter().zip(&b).map(|(x, y)| crate::vector::dot(&crate::vector::normalized(x), &crate::vector::normalized(y))).collect();
    let q: Vec<f64> = [10.0, 25.0, 50.0, 75.0, 90.0].iter().map(|&p| percentile(&cos, p)).collect();
    let (lo, hi) = (cos.iter().copied().fold(f64::INFINITY, f64::min), cos.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    println!("cos(old,new): min {lo:.3} p10 {:.3} p25 {:.3} median {:.3} p75 {:.3} p90 {:.3} max {hi:.3}", q[0], q[1], q[2], q[3], q[4]);
    for th in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85] {
        let hit = cos.iter().filter(|&&c| c >= th).count();
        println!("  theta_related={th:.2} → cos で到達 {hit:3}/{} ({:.0}%)", cos.len(), 100.0 * hit as f64 / cos.len() as f64);
    }
    println!("\n実際に dream が同じクラスタへ入れたペア数（cue で引き直すので cos(old,new) とは別物）:");
    for th in [0.45, 0.50, 0.55, 0.60, 0.65] {
        for members in [8, 16] {
            cfg.memory.theta_related = th;
            cfg.memory.dream_max_members = members;
            let c = cluster_capture(pairs, cfg.memory.clone(), super::super::embedder(&cfg.glob)?.0)?;
            println!("  theta={th:.2} max_members={members:2} → {:3}/{} ({:.0}%)  clusters={}", c["captured"], c["pairs"], c["rate"].as_f64().unwrap_or(0.0) * 100.0, c["clusters"]);
        }
    }
    Ok(())
}

/// Lets the real LLM dream (twice: failed seeds are retried) and measures how often a
/// probe still surfaces the superseded version.
fn measure(pairs: &[Update], cfg: AppConfig) -> Result<Value> {
    let mut llm = client(&cfg.glob);
    if !llm.status().starts_with("OK") {
        return Err(format!("LLM unavailable: {}", llm.status()).into());
    }
    println!("live dream via {}", llm.status());
    let (mut m, _) = scenario(pairs, cfg.memory.clone(), super::super::embedder(&cfg.glob)?.0)?;
    let before: HashSet<String> = m.memories().iter().map(|t| t.text.clone()).collect();
    let mut judge = |r: &crate::DreamRequest| llm.adjudicate(r);
    let reports = m.dream(&mut judge, Some(1_000_000))?;
    let retry = m.dream(&mut judge, Some(1_000_000))?;
    let after: HashSet<String> = m.memories().iter().map(|t| t.text.clone()).collect();
    let gists: HashSet<&String> = after.difference(&before).collect();
    let count = |rs: &[crate::DreamReport], a: DreamAction| rs.iter().filter(|r| r.action == a).count();
    let mut errors: Vec<String> = reports.iter().chain(&retry).filter_map(|r| r.error.clone()).collect::<HashSet<_>>().into_iter().collect();
    errors.sort();
    errors.truncate(5);
    let mut stats = json!({
        "pairs": pairs.len(), "theta_related": cfg.memory.theta_related, "clusters": reports.len(),
        "replace": count(&reports, DreamAction::Replace), "keep": count(&reports, DreamAction::Keep), "error": count(&reports, DreamAction::Error),
        "gists": gists.len(), "retry_clusters": retry.len(), "retry_replace": count(&retry, DreamAction::Replace),
        "retry_keep": count(&retry, DreamAction::Keep), "retry_error": count(&retry, DreamAction::Error), "errors": errors,
    });
    // The sets the dream actually adjudicated.
    let judged: Vec<HashSet<&str>> = reports.iter().chain(&retry).map(|r| r.before.iter().map(|b| b.text.as_str()).collect()).collect();
    let (mut groups, mut samples): ([Vec<[bool; 4]>; 2], Vec<Value>) = ([Vec::new(), Vec::new()], Vec::new());
    for p in pairs {
        let pack: Vec<String> = m.peek(&p.probe).recalled.into_iter().map(|c| c.trace.text).collect();
        let captured = judged.iter().any(|c| c.contains(p.old.as_str()) && c.contains(p.new.as_str()));
        let current = pack.contains(&p.new) || pack.iter().any(|t| gists.contains(t));
        groups[usize::from(captured)].push([after.contains(&p.old), pack.contains(&p.old), current, pack.is_empty()]);
        if samples.len() < 12 {
            samples.push(json!({"probe": p.probe, "captured": captured, "old_in_pack": pack.contains(&p.old), "top": pack.iter().take(2).collect::<Vec<_>>()}));
        }
    }
    stats["captured"] = json!(groups[1].len());
    stats["not_captured"] = json!(groups[0].len());
    let all: Vec<[bool; 4]> = groups[1].iter().chain(&groups[0]).copied().collect();
    for (label, rows) in [("", &all), ("captured_", &groups[1]), ("missed_", &groups[0])] {
        for (i, field) in ["old_survival", "stale_in_pack", "current_in_pack", "empty_pack"].iter().enumerate() {
            let rate = if rows.is_empty() { Value::Null } else { json!(r4(rows.iter().filter(|r| r[i]).count() as f64 / rows.len() as f64)) };
            stats[format!("{label}{field}")] = rate;
        }
    }
    stats["samples"] = json!(samples);
    Ok(stats)
}

pub fn main(args: &[String]) -> Result<()> {
    let f = Flags::parse(args);
    let mut cfg = AppConfig::default();
    let theta: f64 = f.num("theta", 0.0);
    if theta > 0.0 {
        cfg.memory.theta_related = theta;
    }
    let pairs = load_updates(&bench_dir().join("updates"));
    crate::deterministic_ids(1); // same ids → same dream prompts → cached verdicts replay
    if f.has("analyze") {
        return analyze(&pairs, cfg);
    }
    if !f.has("run") {
        return Err("--analyze か --run を指定してください".into());
    }
    let res = measure(&pairs, cfg.clone())?;
    let out = bench_dir().join(format!("supersede_result_theta{}.json", cfg.memory.theta_related));
    std::fs::write(&out, serde_json::to_string_pretty(&res)?)?;
    let mut summary = res.clone();
    summary.as_object_mut().map(|o| o.remove("samples"));
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
