//! The app session: assembles the runtime (embedder, LLM, SQLite store, engine,
//! metrics) and owns the turn log. The web server and the CLI drive the app through it.

use super::llm::{client, Llm};
use super::seed::{parse_duration, SeedItem};
use super::settings::AppConfig;
use super::turn::{invariants, run_turn, TurnMetrics};
use super::{round_to, EmbedTimer, Timed};
use crate::{fmt_local, system_clock, DreamAction, Embedder, Engram, ManualClock, SqliteStore};
use rusqlite::Connection;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};

/// The UI's turn log, in its own table beside `memory` (same file, own connection).
struct TurnLog(Connection);

impl TurnLog {
    fn open(path: &Path) -> crate::Result<Self> {
        let conn = Connection::open(path)?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS turn_log (turn INTEGER PRIMARY KEY, utterance TEXT NOT NULL, \
             note TEXT NOT NULL DEFAULT '', timestamp REAL NOT NULL, system_json TEXT NOT NULL DEFAULT '{}');",
        )?;
        Ok(Self(conn))
    }

    fn save(&mut self, e: &Value, keep: usize) -> crate::Result<()> {
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
    fn load(&self, keep: usize) -> crate::Result<Vec<Value>> {
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

    fn clear(&self) -> crate::Result<()> {
        self.0.execute("DELETE FROM turn_log", [])?;
        Ok(())
    }
}

/// A seed replay in progress: the engine runs on wall time plus `offset`.
pub struct Replay {
    offset: ManualClock,
}

/// One running app instance.
pub struct Session {
    pub cfg: AppConfig,
    pub memory: Engram,
    pub llm: Box<dyn Llm>,
    pub embedding_status: String,
    pub metrics: VecDeque<TurnMetrics>,
    pub log: Vec<Value>,
    pub turn: u64,
    pub seeded: bool,
    pub last_dream: Vec<Value>,
    timer: EmbedTimer,
    turn_log: TurnLog,
    db_path: PathBuf,
}

impl Session {
    /// Assembles the configured runtime over the app database.
    pub fn open(cfg: AppConfig, wipe: bool) -> crate::Result<Self> {
        // SPEC §2: a missing model is no reason not to start — memories are kept unindexed
        // and indexed once the model is back.
        let (embedder, status) = super::embedder(&cfg.glob).unwrap_or_else(|e| (Box::new(super::Offline), format!("ERROR — {e}")));
        let llm = client(&cfg.glob);
        Self::with_parts(cfg, embedder, status, llm, &super::settings::db_path(), wipe)
    }

    /// Assembles a session from explicit parts. `wipe` starts from an empty database
    /// (file delete, then a clear in case the file was locked); otherwise the persisted
    /// turn log is restored.
    pub fn with_parts(cfg: AppConfig, embedder: Box<dyn Embedder>, embedding_status: String, llm: Box<dyn Llm>, db_path: &Path, wipe: bool) -> crate::Result<Self> {
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
            seeded: false,
            last_dream: Vec::new(),
            timer,
            turn_log,
            db_path: db_path.to_path_buf(),
        })
    }

    /// Timestamp of the oldest turn still in the log.
    pub fn start_time(&self) -> Option<f64> {
        self.log.first().and_then(|e| e["timestamp"].as_f64())
    }

    /// Erases memories, turn log and metrics while keeping the database open.
    pub fn reset(&mut self) -> crate::Result<()> {
        self.memory.reset()?;
        self.turn_log.clear()?;
        self.memory.set_clock(system_clock()); // back to real time after a seed replay
        self.metrics.clear();
        self.turn = 0;
        self.log.clear();
        self.seeded = false;
        self.last_dream.clear();
        Ok(())
    }

    /// One user turn (recall → respond → write → cite), logged and persisted. `note` is
    /// a display-only annotation. Returns the turn number.
    pub fn run_turn(&mut self, utterance: &str, note: &str) -> crate::Result<u64> {
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
    pub fn dream(&mut self, budget: Option<usize>) -> crate::Result<&[Value]> {
        let llm = &mut self.llm;
        let reports = self.memory.dream(&mut |r: &crate::DreamRequest| llm.adjudicate(r), budget)?;
        self.last_dream = reports
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
            .collect();
        Ok(&self.last_dream)
    }

    /// Starts replaying seed utterances on a virtual clock anchored at the wall clock,
    /// or at `epoch` (Unix seconds) for a reproducible run.
    pub fn start_replay(&mut self, epoch: Option<f64>) -> Replay {
        let offset = ManualClock::new(0.0);
        let (wall, o) = (system_clock(), offset.clone());
        match epoch {
            Some(t) => self.memory.set_clock(Box::new(move || t + o.get())),
            None => self.memory.set_clock(Box::new(move || wall() + o.get())),
        }
        Replay { offset }
    }

    /// Advances the virtual clock by the item's `advance` and runs it as a turn.
    pub fn replay_step(&mut self, replay: &Replay, item: &SeedItem) -> crate::Result<u64> {
        replay.offset.advance(parse_duration(&item.advance));
        self.run_turn(&item.text, &item.note)
    }

    /// Ends a replay. Headless runs keep the virtual clock (`restore_clock = false`) so a
    /// following dream sees the seeded memories in its past rather than its future.
    pub fn finish_replay(&mut self, restore_clock: bool) {
        if restore_clock {
            self.memory.set_clock(system_clock());
        }
        self.seeded = true;
    }

    /// Replays `items` along a virtual timeline (each `advance` cumulatively pushes the
    /// clock forward); `on_turn` sees every finished turn. The replay is finished even
    /// when a turn fails.
    pub fn replay(&mut self, items: &[SeedItem], epoch: Option<f64>, mut on_turn: impl FnMut(&Session, u64, &SeedItem), restore_clock: bool) -> crate::Result<()> {
        let replay = self.start_replay(epoch);
        let result = items.iter().try_for_each(|item| {
            let turn = self.replay_step(&replay, item)?;
            on_turn(self, turn, item);
            Ok(())
        });
        self.finish_replay(restore_clock);
        result
    }

    /// Bytes of the database file and its WAL / SHM companions.
    pub fn db_size_bytes(&self) -> u64 {
        ["", "-wal", "-shm"].iter().filter_map(|s| std::fs::metadata(format!("{}{s}", self.db_path.display())).ok()).map(|m| m.len()).sum()
    }

    /// Final engine figures, as reported by the API, the CLI and the dump.
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

    /// Every trace, strongest first, as the DB tab shows it.
    pub fn snapshot(&self) -> Vec<Value> {
        let m = &self.memory;
        let now = m.now();
        let mut rows: Vec<(f64, &crate::Trace)> = m.memories().iter().map(|t| (m.strength(t, now), t)).collect();
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

    /// The clusters the next dream would adjudicate, as the DB tab previews them.
    pub fn cluster_preview(&mut self) -> Vec<Value> {
        let now = self.memory.now();
        let clusters = self.memory.clusters(None);
        clusters
            .iter()
            .map(|c| {
                let members: Vec<Value> = c
                    .iter()
                    .map(|x| json!({"id": x.id, "text": x.text, "R": round_to(self.memory.retrievability(x, now), 2),
                                    "stability_days": round_to(x.stability / 86_400.0, 1)}))
                    .collect();
                json!({"seed": c[0].id.chars().take(10).collect::<String>(), "member_count": c.len(), "members": members})
            })
            .collect()
    }
}
