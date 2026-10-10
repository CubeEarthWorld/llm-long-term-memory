//! One conversational turn — recall → converse (+ save / delete tools) → cite — and its
//! metrics. Generation is confined to the converse step; the rest is engine arithmetic.

use super::llm::Llm;
use super::settings::Settings;
use super::{round_to, timed, EmbedTimer};
use crate::{Engram, Recall, Remembered, Stats};
use serde_json::{json, Value};

/// Actions that mean a memory was created or strengthened.
const SAVED: [&str; 2] = ["inserted", "reinforced"];

/// Everything recorded about one turn.
#[derive(Clone, Debug, Default)]
pub struct TurnMetrics {
    pub turn: u64,
    pub utterance: String,
    pub embed_ms: f64,
    pub retrieve_ms: f64,
    pub llm_ms: f64,
    pub write_ms: f64,
    pub counts: Option<Stats>,
    pub db_size_bytes: u64,
    pub vector_mb: f64,
    pub recalled: Vec<Value>,
    pub pack_text: String,
    pub prompt: String,
    pub response: String,
    pub cited: Vec<String>,
    pub written: Vec<Value>,
    pub write_note: String,
}

impl TurnMetrics {
    pub fn total_ms(&self) -> f64 {
        self.embed_ms + self.retrieve_ms + self.llm_ms + self.write_ms
    }

    pub fn records(&self) -> usize {
        self.counts.map_or(0, |c| c.records)
    }

    pub fn pack_chars(&self) -> usize {
        self.pack_text.chars().count()
    }

    /// The detail shown under a turn in the UI and stored in the turn log.
    pub fn detail(&self) -> Value {
        json!({
            "title": super::TITLE,
            "response": self.response,
            "write_note": self.write_note,
            "records": self.records(),
            "pack_chars": self.pack_chars(),
            "pack_n": self.recalled.len(),
            "pack_text": self.pack_text,
            "prompt": self.prompt,
            "written": self.written,
            "cited": self.cited,
            "times": {
                "total": round_to(self.total_ms(), 1),
                "llm": round_to(self.llm_ms, 1),
                "retrieve": round_to(self.retrieve_ms, 1),
                "write": round_to(self.write_ms, 1),
                "embed": round_to(self.embed_ms, 1),
            },
            "recalled": self.recalled,
        })
    }

    /// One row of the metrics table.
    pub fn row(&self) -> Value {
        let c = self.counts.unwrap_or(Stats { records: 0, labile: 0, unindexed: 0 });
        json!({
            "turn": self.turn,
            "system": super::SYSTEM_ID,
            "utterance": self.utterance,
            "embed_ms": round_to(self.embed_ms, 1),
            "retrieve_ms": round_to(self.retrieve_ms, 1),
            "llm_ms": round_to(self.llm_ms, 1),
            "write_ms": round_to(self.write_ms, 1),
            "total_ms": round_to(self.total_ms(), 1),
            "records": c.records,
            "labile": c.labile,
            "unindexed": c.unindexed,
            "pack_chars": self.pack_chars(),
            "pack_n": self.recalled.len(),
            "db_kb": round_to(self.db_size_bytes as f64 / 1024.0, 1),
            "vector_mb": round_to(self.vector_mb, 3),
        })
    }
}

/// The per-turn invariants over metrics rows, keyed by their display label.
pub fn invariants(rows: &[Value], budget_chars: usize, capacity: usize) -> Value {
    let all = |key: &str, max: usize| rows.iter().all(|r| r[key].as_u64().unwrap_or(0) <= max as u64);
    json!({
        format!("全 pack <= {budget_chars}字 (注入予算)"): all("pack_chars", budget_chars),
        format!("全 records <= {capacity}件 (capacity)"): all("records", capacity),
    })
}

/// The JSON event for a remember outcome.
pub fn remember_event(m: &Engram, r: &crate::Result<Remembered>) -> Value {
    let now = m.now();
    let trace = |action: &str, t: &crate::Trace| {
        json!({"action": action, "id": t.id, "text": t.text, "stability": t.stability.round() as i64,
               "R": round_to(m.retrievability(t, now), 3), "consolidated": t.consolidated})
    };
    match r {
        Ok(Remembered::Inserted { trace: t, cosine, evicted }) => {
            let mut e = trace("inserted", t);
            e["cos"] = json!(round_to(*cosine, 3));
            e["evicted"] = json!(evicted.as_ref().map(|v| v.text.clone()));
            e
        }
        Ok(Remembered::Reinforced(t)) => trace("reinforced", t),
        Ok(Remembered::Rejected) => json!({"action": "rejected", "error": "empty text", "text": ""}),
        Ok(Remembered::RateLimited { text }) => json!({"action": "rate_limited", "error": "daily write limit", "text": text}),
        Err(e) => json!({"action": "error", "error": e.to_string()}),
    }
}

/// The recalled items as the UI lists them.
pub fn recalled_items(r: &Recall) -> Vec<Value> {
    r.recalled
        .iter()
        .map(|c| {
            json!({"id": c.trace.id, "text": c.trace.text, "score": round_to(c.score, 3), "cos": round_to(c.cosine, 3),
                   "R": round_to(c.retrievability, 3), "stability": c.trace.stability.round() as i64,
                   "created_at": c.trace.created_at, "tz": c.trace.tz})
        })
        .collect()
}

/// One-line summary of the turn's memory mutations.
pub fn summarize(events: &[Value]) -> String {
    if events.is_empty() {
        return "保存なし".into();
    }
    let label = |a: &str| match a {
        "inserted" => "新規",
        "reinforced" => "強化",
        "deleted" => "削除",
        "rejected" => "却下",
        "rate_limited" => "レート上限",
        "not_found" => "対象なし",
        "not_injected" => "対象外",
        other => other,
    }
    .to_owned();
    let mut counts: Vec<(String, usize)> = Vec::new();
    for e in events {
        let a = e["action"].as_str().unwrap_or("?").to_owned();
        match counts.iter_mut().find(|(k, _)| *k == a) {
            Some((_, n)) => *n += 1,
            None => counts.push((a, 1)),
        }
    }
    counts.iter().map(|(k, n)| format!("{} {n}", label(k))).collect::<Vec<_>>().join(" / ")
}

/// Runs one turn against `m` with `llm`; `timer` separates embedding time from the rest.
pub fn run_turn(m: &mut Engram, llm: &mut dyn Llm, timer: &EmbedTimer, glob: &Settings, turn: u64, utterance: &str) -> TurnMetrics {
    let mut mt = TurnMetrics { turn, utterance: utterance.to_owned(), ..TurnMetrics::default() };

    // 1) RECALL — inject ≤ budget_chars of relevant traces.
    let (recall, logic_ms, embed_ms) = timed(timer, || m.recall(utterance));
    let recall = recall.unwrap_or_else(|e| {
        eprintln!("recall failed: {e}");
        Recall::default()
    });
    mt.retrieve_ms = logic_ms;
    mt.embed_ms += embed_ms;
    mt.pack_text = recall.pack_text.clone();
    mt.recalled = recalled_items(&recall);

    // 2) CONVERSE — generation + tools; then 3) CITE — used memories are strengthened.
    let max_writes = glob.max_writes_per_turn;
    let mut events: Vec<Value> = Vec::new();
    let mut write_ms = 0.0;
    let saved = |events: &[Value]| events.iter().filter(|e| e["action"].as_str().is_some_and(|a| SAVED.contains(&a))).count();
    let current_time = m.now_local();
    let (answer, logic_ms, embed_ms) = timed(timer, || {
        // Times the engine work of one mutation only (its embedding is counted by `timed`).
        let mut write = |m: &mut Engram, events: &mut Vec<Value>, f: &mut dyn FnMut(&mut Engram) -> Value| {
            let e0 = timer.peek_ms();
            let t0 = std::time::Instant::now();
            let ev = f(m);
            write_ms += t0.elapsed().as_secs_f64() * 1000.0 - (timer.peek_ms() - e0);
            events.push(ev.clone());
            ev
        };
        let mut tool = |name: &str, args: &Value| -> Value {
            let text = args["text"].as_str().unwrap_or("").to_owned(); // a non-string text is no proposition
            match name {
                "save_memory" => {
                    if saved(&events) >= max_writes {
                        events.push(json!({"action": "rate_limited", "error": "per-turn save limit", "text": text}));
                        return json!({"ok": false, "action": "rate_limited"});
                    }
                    // The model may send junk here.
                    let salience = args["salience"].as_f64().or_else(|| args["salience"].as_str().and_then(|s| s.trim().parse().ok())).unwrap_or(1.0);
                    let cue = args["cue"].as_str().unwrap_or("").to_owned();
                    let ev = write(m, &mut events, &mut |m| {
                        let r = m.remember(&text, salience, &cue);
                        remember_event(m, &r)
                    });
                    json!({"ok": ev["action"].as_str().is_some_and(|a| SAVED.contains(&a)), "action": ev["action"], "id": ev["id"]})
                }
                "delete_memory" => {
                    let id = args["id"].as_str().unwrap_or("").to_owned();
                    // The model may forget only what it was shown this turn (SPEC §4: id-only,
                    // and the prompt hands it nothing but the injected ids).
                    if !recall.recalled.iter().any(|c| c.trace.id == id) {
                        events.push(json!({"action": "not_injected", "id": id}));
                        return json!({"ok": false, "action": "not_injected"});
                    }
                    let ev = write(m, &mut events, &mut |m| match m.forget(&id) {
                        Ok(Some(t)) => json!({"action": "deleted", "id": id, "text": t.text}),
                        Ok(None) => json!({"action": "not_found", "id": id}),
                        Err(e) => json!({"action": "error", "id": id, "error": e.to_string()}),
                    });
                    json!({"ok": ev["action"] == "deleted", "action": ev["action"]})
                }
                other => json!({"error": format!("unknown tool {other}")}),
            }
        };
        let answer = llm.converse(&recall.pack_text, utterance, &mut tool, &current_time);
        let fallback: Vec<String> = if saved(&events) == 0 && glob.tool_fallback {
            llm.extract(utterance, &answer.text, &current_time, &recall.pack_text).into_iter().take(max_writes).collect()
        } else {
            Vec::new()
        };
        for text in &fallback {
            write(m, &mut events, &mut |m| {
                let r = m.remember(text, 1.0, "");
                remember_event(m, &r)
            });
        }
        answer
    });
    mt.llm_ms += (logic_ms - write_ms).max(0.0);
    mt.write_ms += write_ms;
    mt.embed_ms += embed_ms;
    mt.response = answer.text.clone();
    mt.prompt = answer.prompt;
    mt.write_note = summarize(&events);
    mt.written = events;
    mt.cited = m.cite(&answer.text).unwrap_or_default();
    mt.counts = Some(m.stats());
    mt.vector_mb = m.vector_mb();
    mt
}
