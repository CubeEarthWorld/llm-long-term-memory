//! The durable substrate (SPEC §2). The engine keeps every trace in RAM; a store
//! only persists, so its contract is four operations.

use crate::{Result, Trace};

/// One change inside an atomic [`Store::write`].
#[derive(Clone, Copy, Debug)]
pub enum Write<'a> {
    /// Insert or replace the trace with this id.
    Put(&'a Trace),
    /// Delete the trace with this id (no-op when absent).
    Remove(&'a str),
}

/// Persistence for traces. One engine owns one store.
pub trait Store: Send {
    /// Every trace, oldest id first (ids sort by time, so this is insertion order).
    fn load_all(&self) -> Result<Vec<Trace>>;
    /// Applies `writes` in order as one atomic commit — one engine operation is one
    /// write, so a crash never leaves half an operation behind.
    fn write(&mut self, writes: &[Write<'_>]) -> Result<()>;
    /// Deletes every trace.
    fn clear(&mut self) -> Result<()>;
    /// Snapshots the store before a dream (the only irreversible rewrite); returns
    /// where it went. Stores without durability have nothing to snapshot.
    fn backup(&mut self) -> Option<String> {
        None
    }
}

/// A store that lives and dies with the process — for tests, benchmarks and
/// ephemeral agents. Every write is O(1) per row.
#[derive(Debug, Default)]
pub struct MemoryStore {
    rows: std::collections::HashMap<String, Trace>,
}

impl Store for MemoryStore {
    fn load_all(&self) -> Result<Vec<Trace>> {
        let mut rows: Vec<Trace> = self.rows.values().cloned().collect();
        rows.sort_unstable_by(|a, b| a.id.cmp(&b.id));
        Ok(rows)
    }

    fn write(&mut self, writes: &[Write<'_>]) -> Result<()> {
        for w in writes {
            match *w {
                Write::Put(t) => {
                    self.rows.insert(t.id.clone(), t.clone());
                }
                Write::Remove(id) => {
                    self.rows.remove(id);
                }
            }
        }
        Ok(())
    }

    fn clear(&mut self) -> Result<()> {
        self.rows.clear();
        Ok(())
    }
}
