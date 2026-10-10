//! The app layer with fakes: one turn end to end (recall → converse with tools → cite),
//! the per-turn caps and fallback, and the session's turn log and seed replay.
#![cfg(feature = "app")]

mod support;

use engram::app::llm::{Converse, Llm, ToolFn};
use engram::app::seed::SeedItem;
use engram::app::session::Session;
use engram::app::settings::AppConfig;
use engram::{DreamDecision, DreamRequest};
use serde_json::json;
use std::sync::{Arc, Mutex};
use support::{temp_dir, FakeEmbedder, DAY};

/// Saves the utterance (with a salience and a cue), deletes on `forget:<id> …`, cites
/// every memory it was shown, and returns canned extraction candidates.
struct Scripted {
    saves: usize,
    extract: Vec<String>,
    known: Arc<Mutex<Option<String>>>,
}

impl Scripted {
    fn new(saves: usize, extract: &[&str]) -> (Self, Arc<Mutex<Option<String>>>) {
        let known = Arc::new(Mutex::new(None));
        (Self { saves, extract: extract.iter().map(|s| s.to_string()).collect(), known: known.clone() }, known)
    }
}

impl Llm for Scripted {
    fn status(&self) -> String {
        "OK - scripted".into()
    }
    fn model(&self) -> String {
        "scripted".into()
    }
    fn converse(&mut self, pack: &str, user: &str, tool: &mut ToolFn<'_>, _: &str) -> Converse {
        for i in 0..self.saves {
            let text = if i == 0 { user.to_owned() } else { format!("{user} {i}") };
            tool("save_memory", &json!({"text": text, "salience": 2, "cue": format!("why {user}")}));
        }
        if let Some(id) = user.strip_prefix("forget:").and_then(|rest| rest.split_whitespace().next()) {
            tool("delete_memory", &json!({"id": id}));
        }
        let cited: Vec<String> = pack.lines().filter_map(|l| l.find('《').map(|i| l[i..].to_owned())).collect();
        Converse { text: format!("ok {}", cited.join(" ")), prompt: format!("[pack]{pack}[/pack]") }
    }
    fn extract(&mut self, _: &str, _: &str, _: &str, known: &str) -> Vec<String> {
        *self.known.lock().unwrap() = Some(known.to_owned());
        self.extract.clone()
    }
    fn adjudicate(&mut self, _: &DreamRequest) -> engram::Result<DreamDecision> {
        Ok(DreamDecision::keep())
    }
}

fn config() -> AppConfig {
    let mut c = AppConfig::default();
    c.memory.cosine_floor = 0.0;
    c
}

fn session(dir: &std::path::Path, cfg: AppConfig, llm: Scripted, wipe: bool) -> Session {
    Session::with_parts(cfg, Box::new(FakeEmbedder::new()), "OK — fake".into(), Box::new(llm), &dir.join("app.db"), wipe).unwrap()
}

#[test]
fn turn_saves_recalls_and_cites() {
    let dir = temp_dir("turn");
    let mut s = session(&dir, config(), Scripted::new(1, &[]).0, true);
    let clock = engram::ManualClock::new(support::START);
    s.memory.set_clock(clock.clock());
    s.run_turn("user likes matcha ice cream", "").unwrap();
    let m1 = s.metrics.back().unwrap().clone();
    assert_eq!(m1.written[0]["action"], "inserted");
    assert_eq!(m1.written[0]["stability"], json!((2.0 * DAY) as i64));
    clock.advance(DAY);
    s.run_turn("matcha ice cream again?", "").unwrap();
    let m2 = s.metrics.back().unwrap().clone();
    assert!(!m2.recalled.is_empty() && m2.pack_text.contains("user likes matcha ice cream"));
    assert!(!m2.cited.is_empty() && s.memory.memory(&m2.cited[0]).unwrap().stability > 2.0 * DAY);
    assert!(m2.records() == 2 && m2.total_ms() >= 0.0);
    let first = s.memory.memory(m1.written[0]["id"].as_str().unwrap()).unwrap();
    assert_eq!(first.cue, "why user likes matcha ice cream");
    assert_eq!(s.log[1]["system"]["records"], 2);
}

#[test]
fn per_turn_cap_and_fallback() {
    let mut cfg = config();
    cfg.glob.max_writes_per_turn = 2;
    let dir = temp_dir("cap");
    let mut s = session(&dir, cfg.clone(), Scripted::new(5, &[]).0, true);
    s.run_turn("hello there", "").unwrap();
    let w = &s.metrics.back().unwrap().written;
    assert_eq!(w.iter().filter(|e| e["action"] == "inserted").count(), 2);
    assert_eq!(w.iter().filter(|e| e["action"] == "rate_limited").count(), 3);
    let (llm, known) = Scripted::new(0, &["extracted fact one", "extracted fact two"]);
    let dir2 = temp_dir("fallback");
    let mut s2 = session(&dir2, cfg, llm, true);
    s2.memory.remember("user likes matcha ice cream", 1.0, "").unwrap(); // so the pack is not empty
    s2.run_turn("matcha ice cream again?", "").unwrap();
    let m = s2.metrics.back().unwrap();
    let texts: Vec<&str> = m.written.iter().map(|e| e["text"].as_str().unwrap()).collect();
    assert_eq!(texts, ["extracted fact one", "extracted fact two"]);
    assert_eq!(known.lock().unwrap().as_deref(), Some(m.pack_text.as_str())); // the fallback sees what was recalled
    assert_eq!(m.write_note, "新規 2");
}

#[test]
fn delete_tool() {
    let dir = temp_dir("delete");
    let mut s = session(&dir, config(), Scripted::new(0, &[]).0, true);
    let id = support::insert(&mut s.memory, "delete me later", 1.0, "");
    s.run_turn(&format!("forget:{id} delete me later"), "").unwrap(); // recalled, so it may be deleted
    assert_eq!(s.metrics.back().unwrap().written[0]["action"], "deleted");
    assert!(s.memory.memories().is_empty());
}

fn item(text: &str, note: &str, advance: &str) -> SeedItem {
    SeedItem { text: text.into(), note: note.into(), advance: advance.into() }
}

#[test]
fn replay_runs_on_a_virtual_clock_and_restores_it() {
    let dir = temp_dir("replay");
    let mut s = session(&dir, config(), Scripted::new(1, &[]).0, true);
    let mut seen = Vec::new();
    s.replay(&[item("user likes tea", "重要", "0"), item("user moved to osaka", "", "5y")], None, |_, t, _| seen.push(t), true).unwrap();
    assert!(seen == [1, 2] && s.turn == 2 && s.seeded);
    let wall = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
    assert!((s.memory.now() - wall).abs() < 60.0); // the clock is restored
    assert_eq!(s.log.iter().map(|e| e["note"].as_str().unwrap()).collect::<Vec<_>>(), ["重要", ""]);
    assert!(s.log[1]["timestamp"].as_f64().unwrap() - s.log[0]["timestamp"].as_f64().unwrap() >= 5.0 * 31_536_000.0);
    let t0 = s.start_time();
    s.replay(&[item("kept virtual", "", "1d")], None, |_, _, _| {}, false).unwrap();
    assert!(s.memory.now() - wall > 0.9 * DAY); // still virtual
    s.replay(&[item("pinned", "", "1d")], Some(1_000_000.0), |_, _, _| {}, false).unwrap();
    assert_eq!(s.memory.now(), 1_000_000.0 + 86_400.0); // an epoch pins the virtual clock
    drop(s);
    assert_eq!(session(&dir, config(), Scripted::new(1, &[]).0, false).start_time(), t0); // survives a restart
}

#[test]
fn turn_log_window_drops_the_oldest() {
    let dir = temp_dir("window");
    let mut cfg = config();
    cfg.glob.max_turn_log = 2;
    let mut s = session(&dir, cfg.clone(), Scripted::new(1, &[]).0, true);
    for i in 0..3 {
        s.run_turn(&format!("turn {i} fact about cats"), "").unwrap();
    }
    assert_eq!(s.log.iter().map(|e| e["turn"].as_u64().unwrap()).collect::<Vec<_>>(), [2, 3]); // RAM window
    drop(s);
    let reopened = session(&dir, cfg, Scripted::new(1, &[]).0, false);
    assert_eq!(reopened.log.iter().map(|e| e["turn"].as_u64().unwrap()).collect::<Vec<_>>(), [2, 3]); // DB window
}

#[test]
fn turn_log_survives_restart_and_reset_clears_it() {
    let dir = temp_dir("restart");
    let mut s = session(&dir, config(), Scripted::new(1, &[]).0, true);
    s.run_turn("user has a cat", "n1").unwrap();
    drop(s);
    let mut s2 = session(&dir, config(), Scripted::new(1, &[]).0, false);
    assert!(s2.turn == 1 && s2.log[0]["note"] == "n1" && s2.log[0]["system"]["records"] == 1);
    assert_eq!(s2.memory.memories().len(), 1);
    s2.reset().unwrap();
    assert!(s2.turn == 0 && s2.log.is_empty() && s2.start_time().is_none() && s2.memory.memories().is_empty());
    drop(s2);
    assert!(session(&dir, config(), Scripted::new(1, &[]).0, false).log.is_empty());
}

#[test]
fn app_config_round_trips_and_rejects_bad_values() {
    let c = AppConfig::from_json(&json!({"memory": {"capacity": "42", "alpha": 0.5}, "glob": {"tool_fallback": "false", "max_turn_log": 0}})).unwrap();
    assert!(c.memory.capacity == 42 && c.memory.alpha == 0.5 && !c.glob.tool_fallback && c.glob.max_turn_log == 1);
    assert_eq!(AppConfig::from_json(&c.to_json()).unwrap(), c);
    assert!(AppConfig::from_json(&json!({"memory": {"capacity": 0}})).is_err());
}

#[test]
fn dream_reports_are_logged_for_the_ui() {
    let dir = temp_dir("dreamlog");
    let mut s = session(&dir, config(), Scripted::new(0, &[]).0, true);
    for t in ["trip kyoto plan schedule note", "trip kyoto hotel schedule note"] {
        s.memory.remember(t, 1.0, "").unwrap();
    }
    let reports = s.dream(None).unwrap().to_vec();
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0]["action"], "keep");
    assert_eq!(reports[0]["before"][0]["text"], "trip kyoto hotel schedule note");
    assert_eq!(s.last_dream, reports);
    assert_eq!(s.cluster_preview().len(), 0); // settled
    assert_eq!(s.snapshot().len(), 2);
}

#[test]
fn the_model_cannot_delete_what_it_was_not_shown() {
    let dir = temp_dir("delete-uninjected");
    let mut c = config();
    c.memory.cosine_floor = 0.99; // nothing is similar enough to be injected
    let mut s = session(&dir, c, Scripted::new(0, &[]).0, true);
    let id = support::insert(&mut s.memory, "a memory the model never saw", 1.0, "");
    s.run_turn(&format!("forget:{id}"), "").unwrap();
    assert_eq!(s.metrics.back().unwrap().written[0]["action"], "not_injected");
    assert_eq!(s.memory.memories().len(), 1);
}

#[test]
fn a_session_starts_without_an_embedding_model() {
    let dir = temp_dir("offline");
    let mut s = Session::with_parts(config(), Box::new(engram::app::Offline), "ERROR — no model".into(), Box::new(Scripted::new(1, &[]).0), &dir.join("o.db"), true).unwrap();
    s.run_turn("user likes tea", "").unwrap();
    let m = s.memory.memories();
    assert_eq!(m.len(), 1);
    assert!(m[0].vector.is_empty()); // kept, unindexed until a model is back
}
