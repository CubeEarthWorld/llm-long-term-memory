//! SQLite substrate: one `memory` table (SPEC §2) plus a ring of pre-dream snapshots.

use crate::store::{Store, Write};
use crate::{Result, Trace};
use rusqlite::types::ValueRef;
use rusqlite::{params, Connection};
use std::path::{Path, PathBuf};

const SCHEMA: &str = "CREATE TABLE IF NOT EXISTS memory (
  id TEXT PRIMARY KEY,
  text TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  tz TEXT NOT NULL,
  last_recall INTEGER NOT NULL,
  stability REAL NOT NULL,
  consolidated INTEGER NOT NULL,
  model_id TEXT NOT NULL,
  vector BLOB NOT NULL,
  cue TEXT NOT NULL DEFAULT ''
);";

/// Snapshot files kept before dreams.
const SNAPSHOT_GENERATIONS: usize = 8;

/// A single-file SQLite store (WAL, one fsync per checkpoint rather than per commit).
pub struct SqliteStore {
    conn: Connection,
    path: Option<PathBuf>,
    snapshots: u32,
}

impl SqliteStore {
    /// Opens (creating directories and schema as needed) the database at `path`.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        Self::init(conn, Some(path.to_path_buf()))
    }

    /// A private in-memory database.
    pub fn open_in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?, None)
    }

    fn init(conn: Connection, path: Option<PathBuf>) -> Result<Self> {
        conn.execute_batch(SCHEMA)?;
        Ok(Self { conn, path, snapshots: 0 })
    }

    /// The database file (`None` in memory).
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// The underlying connection, for applications that keep their own tables beside
    /// the memory table.
    pub fn connection(&self) -> &Connection {
        &self.conn
    }
}

/// A dynamically typed cell as a number (`None` for text that is not one, or NULL).
fn number(v: ValueRef<'_>) -> Option<f64> {
    match v {
        ValueRef::Integer(i) => Some(i as f64),
        ValueRef::Real(r) => Some(r),
        ValueRef::Text(t) => std::str::from_utf8(t).ok()?.trim().parse().ok(),
        _ => None,
    }
}

fn text(v: ValueRef<'_>) -> Option<String> {
    match v {
        ValueRef::Text(t) => Some(String::from_utf8_lossy(t).into_owned()),
        ValueRef::Null => None,
        ValueRef::Integer(i) => Some(i.to_string()),
        ValueRef::Real(r) => Some(r.to_string()),
        ValueRef::Blob(_) => None,
    }
}

impl Store for SqliteStore {
    /// Ordered by id (ULIDs sort by time): `INSERT OR REPLACE` re-inserts a row, so
    /// rowid order would move a rehearsed trace to the end. SQLite is dynamically
    /// typed, so a hand-edited or legacy row can hold a non-numeric stability or a
    /// vector that is not whole float32s; such a row is skipped, not fatal.
    fn load_all(&self) -> Result<Vec<Trace>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, text, created_at, tz, last_recall, stability, consolidated, model_id, vector, cue FROM memory ORDER BY id",
        )?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(r) = rows.next()? {
            let cell = |i: usize| r.get_ref(i).unwrap_or(ValueRef::Null);
            let vector = match cell(8) {
                ValueRef::Blob(b) if b.len() % 4 == 0 => {
                    b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
                }
                _ => {
                    eprintln!("skipping unreadable memory row {:?}: vector is not float32s", text(cell(0)));
                    continue;
                }
            };
            let trace = (|| {
                Some(Trace {
                    id: text(cell(0))?,
                    text: text(cell(1))?,
                    created_at: number(cell(2))? as i64,
                    tz: text(cell(3))?,
                    last_recall: number(cell(4))? as i64,
                    stability: number(cell(5))?,
                    consolidated: number(cell(6))? != 0.0,
                    model_id: text(cell(7))?,
                    vector,
                    cue: text(cell(9)).unwrap_or_default(),
                })
            })();
            match trace {
                Some(t) => out.push(t),
                None => eprintln!("skipping unreadable memory row {:?}", text(cell(0))),
            }
        }
        Ok(out)
    }

    fn write(&mut self, writes: &[Write<'_>]) -> Result<()> {
        let tx = self.conn.transaction()?;
        {
            let mut put = tx.prepare_cached(
                "INSERT OR REPLACE INTO memory(id,text,created_at,tz,last_recall,stability,consolidated,model_id,vector,cue) \
                 VALUES(?,?,?,?,?,?,?,?,?,?)",
            )?;
            let mut remove = tx.prepare_cached("DELETE FROM memory WHERE id=?")?;
            for w in writes {
                match *w {
                    Write::Put(t) => {
                        let blob: Vec<u8> = t.vector.iter().flat_map(|x| x.to_le_bytes()).collect();
                        put.execute(params![
                            t.id, t.text, t.created_at, t.tz, t.last_recall, t.stability, t.consolidated, t.model_id, blob, t.cue
                        ])?;
                    }
                    Write::Remove(id) => {
                        remove.execute([id])?;
                    }
                }
            }
        }
        tx.commit()?;
        Ok(())
    }

    fn clear(&mut self) -> Result<()> {
        self.conn.execute("DELETE FROM memory", [])?;
        Ok(())
    }

    /// Copies the database into `snapshots/` beside it and keeps the newest 8.
    fn backup(&mut self) -> Option<String> {
        let path = self.path.clone()?;
        let result = (|| -> Result<String> {
            let dir = path.parent().unwrap_or(Path::new(".")).join("snapshots");
            std::fs::create_dir_all(&dir)?;
            self.snapshots += 1;
            let millis = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis();
            let dst = dir.join(format!("snap_{millis}_{:04}.db", self.snapshots));
            // Page copy (online backup API): far cheaper than VACUUM INTO, which rebuilds.
            self.conn.backup(rusqlite::MAIN_DB, &dst, None)?;
            let mut snaps: Vec<PathBuf> = std::fs::read_dir(&dir)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("snap_") && n.ends_with(".db")))
                .collect();
            snaps.sort();
            for old in &snaps[..snaps.len().saturating_sub(SNAPSHOT_GENERATIONS)] {
                let _ = std::fs::remove_file(old);
            }
            Ok(dst.to_string_lossy().into_owned())
        })();
        result.map_err(|e| eprintln!("pre-dream backup failed: {e}")).ok() // the dream is the only deleter
    }
}
