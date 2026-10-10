//! The runner session: assembles the runtime (embedder, LLM, SQLite store, engine,
//! metrics) and owns the turn log. The CLI drives the runner through it.

use crate::llm::{client, Llm};
use crate::seed::{parse_duration, SeedItem};
use crate::settings::AppConfig;
use crate::turn::{invariants, run_turn, TurnMetrics};
use crate::{round_to, EmbedTimer, Timed};
use crate::SqliteStore;
use engram::{fmt_local, system_clock, DreamAction, Embedder, Engram, ManualClock};
use rusqlite::Connection;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};

/// The runner's turn log, in its own table beside `memory` (same file, own connection).
struct TurnLog(Connection);

impl TurnLog {
    fn open(path: &Path) -> engram::Result<Self> {
        let conn = Connection::open(path)?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS turn_log (turn INTEGER PRIMARY KEY, utterance TEXT NOT NULL, \
             note TEXT NOT NULL DEFAULT '', timestamp REAL NOT NULL, system_json TEXT NOT NULL DEFAULT '{}');",
        )?;
        Ok(Self(conn))
    }

    fn save(&mut self, e: &Value, keep: usize) -> engram::Result<()> {
        let tx = self.0.transaction()?;
        tx.execute(
            "INSERT OR REPLACE INTO turn_log(turn,utterance,note,timestamp,system_json) VALUES(?,?,?,?,?)",
            rusqlite::params![e["turn"].as_i64(), e["utterance"].as_str(), e["note"].as_str(), e["timestamp"].as_f64(), e["system"].to_string()],
        )?;
        // Turns are consecutive, so the window is a key range (an index seek, not a sort).
        tx.execute("DELETE FROM turn_log WHERE turn <= ?", [e["turn"].as_i64().unwrap_or(0) - keep as i64])?;
        tx.commit()?;
        Ok(())
    }

    /// The newest `keep` turns, oldest first; unreadable rows are skipped.
    fn load(&self, keep: usize) -> engram::Result<Vec<Value>> {
        let mut stmt = self.0.prepare("SELECT turn, utterance, note, timestamp, system_json FROM turn_log ORDER BY turn DESC LIMIT ?")?;
        let rows = stmt.query_map([keep as i64], |r| {
            Ok(json!({
                "turn": r.get::<_, i64>(0)?,
                "utterance": r.get::<_, String>(1)?,
                "note": r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                "timestamp": r.get::<_, f64>(3)?,
                "system": serde_json::from_str::<Value>(&r.get::<_, String>(4).unwrap_or_default()).unwrap_or_else(|_| json!({})),
            }))
        })?;
        let mut out: Vec<Value> = rows.filter_map(|r| r.ok()).collect();
        out.reverse();
        Ok(out)
    }

    fn clear(&self) -> engram::Result<()> {
        self.0.execute("DELETE FROM turn_log", [])?;
        Ok(())
    }
}

/// One runner instance.
pub struct Session {
    pub cfg: AppConfig,
    pub memory: Engram,
    pub llm: Box<dyn Llm>,
    pub embedding_status: String,
    pub metrics: VecDeque<TurnMetrics>,
    pub log: Vec<Value>,
    pub turn: u64,
    timer: EmbedTimer,
    turn_log: TurnLog,
    db_path: PathBuf,
}

impl Session {
    /// Assembles the configured runtime over the app database.
    pub fn open(cfg: AppConfig, wipe: bool) -> engram::Result<Self> {
        // SPEC §2: a missing model is no reason not to start — memories are kept unindexed
        // and indexed once the model is back.
        let (embedder, status) = crate::embedder(&cfg.glob).unwrap_or_else(|e| (Box::new(crate::Offline), format!("ERROR — {e}")));
        let llm = client(&cfg.glob);
        Self::with_parts(cfg, embedder, status, llm, &crate::settings::db_path(), wipe)
    }

    /// Assembles a session from explicit parts. `wipe` starts from an empty database
    /// (file delete, then a clear in case the file was locked); otherwise the persisted
    /// turn log is restored.
    pub fn with_parts(cfg: AppConfig, embedder: Box<dyn Embedder>, embedding_status: String, llm: Box<dyn Llm>, db_path: &Path, wipe: bool) -> engram::Result<Self> {
        if wipe {
            for suffix in ["", "-wal", "-shm"] {
                let _ = std::fs::remove_file(format!("{}{suffix}", db_path.display()));
            }
        }
        let timer = EmbedTimer::default();
        let store = Box::new(SqliteStore::open(db_path)?);
        let embedder = Box::new(Timed { inner: embedder, timer: timer.clone() });
        let mut memory = Engram::open(store, embedder, cfg.memory.clone(), &cfg.glob.default_timezone)?;
        let turn_log = TurnLog::open(db_path)?;
        let log = if wipe {
            memory.reset()?;
            turn_log.clear()?;
            Vec::new()
        } else {
            turn_log.load(cfg.glob.max_turn_log)?
        };
        let turn = log.iter().filter_map(|e| e["turn"].as_u64()).max().unwrap_or(0);
        Ok(Self {
            cfg,
            memory,
            llm,
            embedding_status,
            metrics: VecDeque::new(),
            log,
            turn,
            timer,
            turn_log,
            db_path: db_path.to_path_buf(),
        })
    }

    /// One user turn (recall → respond → write → cite), logged and persisted. `note` is
    /// a display-only annotation. Returns the turn number.
    pub fn run_turn(&mut self, utterance: &str, note: &str) -> engram::Result<u64> {
        let turn = self.turn + 1;
        let mut m = run_turn(&mut self.memory, self.llm.as_mut(), &self.timer, &self.cfg.glob, turn, utterance);
        m.db_size_bytes = self.db_size_bytes();
        let entry = json!({"turn": turn, "utterance": utterance, "note": note, "timestamp": self.memory.now(), "system": m.detail()});
        let keep = self.cfg.glob.max_turn_log; // sliding windows: bounded RAM and DB
        self.metrics.push_back(m);
        while self.metrics.len() > self.cfg.glob.max_metrics_history {
            self.metrics.pop_front();
        }
        self.turn = turn;
        self.turn_log.save(&entry, keep)?;
        self.log.push(entry);
        let excess = self.log.len().saturating_sub(keep);
        self.log.drain(..excess);
        Ok(turn)
    }

    /// Memory consolidation: ≤ `budget` LLM verdicts (`None` = the configured budget).
    pub fn dream(&mut self, budget: Option<usize>) -> engram::Result<Vec<Value>> {
        let llm = &mut self.llm;
        let reports = self.memory.dream(&mut |r: &engram::DreamRequest| llm.adjudicate(r), budget)?;
        Ok(reports
            .iter()
            .map(|r| {
                let before: Vec<Value> = r.before.iter().map(|b| json!({"id": b.id, "text": b.text, "R": b.retrievability})).collect();
                let after: Vec<Value> = r.after.iter().map(|g| json!({"id": g.id, "text": g.text, "stability": g.stability.round() as i64})).collect();
                let mut v = json!({"action": r.action.as_str(), "before": before, "after": after});
                match r.action {
                    DreamAction::Replace => v["absorbed"] = json!(r.absorbed),
                    DreamAction::Error => v["error"] = json!(r.error),
                    DreamAction::Keep => {}
                }
                v
            })
            .collect())
    }

    /// Replays `items` along a virtual timeline anchored at the wall clock, or at
    /// `epoch` (Unix seconds) for a reproducible run: each `advance` cumulatively pushes
    /// the clock forward, and `on_turn` sees every finished turn. The clock stays virtual
    /// afterwards, so a following dream sees the seeded memories in its past.
    pub fn replay(&mut self, items: &[SeedItem], epoch: Option<f64>, mut on_turn: impl FnMut(&Session, u64, &SeedItem)) -> engram::Result<()> {
        let offset = ManualClock::new(0.0);
        let (wall, o) = (system_clock(), offset.clone());
        match epoch {
            Some(t) => self.memory.set_clock(Box::new(move || t + o.get())),
            None => self.memory.set_clock(Box::new(move || wall() + o.get())),
        }
        items.iter().try_for_each(|item| {
            offset.advance(parse_duration(&item.advance));
            let turn = self.run_turn(&item.text, &item.note)?;
            on_turn(self, turn, item);
            Ok(())
        })
    }

    /// Bytes of the database file and its WAL / SHM companions.
    pub fn db_size_bytes(&self) -> u64 {
        ["", "-wal", "-shm"].iter().filter_map(|s| std::fs::metadata(format!("{}{s}", self.db_path.display())).ok()).map(|m| m.len()).sum()
    }

    /// Final engine figures, as the CLI reports them.
    pub fn final_stats(&self) -> Value {
        let s = self.memory.stats();
        json!({"records": s.records, "labile": s.labile, "unindexed": s.unindexed,
               "vector_mb": round_to(self.memory.vector_mb(), 3), "db_kb": round_to(self.db_size_bytes() as f64 / 1024.0, 1)})
    }

    /// The metrics rows.
    pub fn metric_rows(&self) -> Vec<Value> {
        self.metrics.iter().map(TurnMetrics::row).collect()
    }

    /// The per-turn invariants over the metrics rows.
    pub fn invariants(&self) -> Value {
        invariants(&self.metric_rows(), self.cfg.memory.budget_chars, self.cfg.memory.capacity)
    }

    /// Every trace, strongest first, as `ltm run --inspect` prints it.
    pub fn snapshot(&self) -> Vec<Value> {
        let m = &self.memory;
        let now = m.now();
        let mut rows: Vec<(f64, &engram::Trace)> = m.memories().iter().map(|t| (m.strength(t, now), t)).collect();
        rows.sort_by(|a, b| b.0.total_cmp(&a.0));
        rows.iter()
            .map(|(strength, t)| {
                json!({"id": t.id, "text": t.text, "local": fmt_local(t.created_at, &t.tz), "tz": t.tz,
                       "stability_days": round_to(t.stability / 86_400.0, 2), "R": round_to(m.retrievability(t, now), 3),
                       "strength_days": round_to(strength / 86_400.0, 2), "consolidated": t.consolidated,
                       "last_recall": t.last_recall, "model_id": t.model_id})
            })
            .collect()
    }
}
