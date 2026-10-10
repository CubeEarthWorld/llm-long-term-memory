//! The storage seam (SPEC §2). A storage persists traces and answers the few questions
//! the engine asks — a trace by id or text, a handful of ordered views and similarity
//! search — so the engine holds no copy of the memory and opening costs nothing. Every
//! view is one index range; nothing here needs a full pass. [`crate::Store`] is the
//! bundled implementation; implement [`Storage`] to keep memories in any database.

use crate::{Result, Trace};

/// An ordered selection of traces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View<'a> {
    /// Every trace, oldest id first.
    All,
    /// The trace with this id.
    Id(&'a str),
    /// The trace with exactly this text.
    Text(&'a str),
    /// Labile traces indexed under `model_id`, oldest id first: the dream's queue.
    Labile { model_id: &'a str },
    /// Traces not indexed under `model_id`, newest id first (within each other model):
    /// re-embedded newest first.
    Stale { model_id: &'a str },
    /// Traces of stability band `band` ([`Trace::band`]) stated at or before `created_by`,
    /// least recently recalled first (ties: oldest id): the band's eviction order.
    Band { band: u8, created_by: i64 },
    /// Traces stated after `created_after` (still inside their grace period), earliest
    /// first (ties: oldest id).
    Young { created_after: i64 },
}

/// One similarity search result.
#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    /// Cosine between the query and the trace's vector.
    pub cos: f64,
    pub trace: Trace,
}

/// One change inside an atomic [`Storage::write`].
#[derive(Clone, Copy, Debug)]
pub enum Write<'a> {
    /// Insert the trace, or replace the row with its id.
    Insert(&'a Trace),
    /// Update the state of a trace (strengthening, consolidation); text and vector stay.
    State { id: &'a str, last_recall: i64, stability: f64, consolidated: bool },
    /// Attach a (re-)embedded vector.
    Index { id: &'a str, model_id: &'a str, vector: &'a [f32] },
    /// Delete the trace with this id (no-op when absent).
    Remove(&'a str),
}

/// Persistence and lookup for traces. One engine owns one storage. Ids sort in creation
/// order (ULIDs), so "oldest id first" is insertion order.
pub trait Storage: Send {
    /// Up to `limit` traces of `view`, in the view's order.
    fn find(&self, view: View<'_>, limit: usize) -> Result<Vec<Trace>>;
    /// How many traces `view` holds.
    fn count(&self, view: View<'_>) -> Result<usize>;
    /// Up to `k` traces indexed under `model_id` with `cos(vector, q) ≥ min_cos` (and
    /// `created_at ≤ stated_by` when given — applied inside the search, not after it),
    /// highest cosine first, ties oldest id first. `q` is unit length. An approximate
    /// index may miss some.
    fn near(&self, q: &[f32], model_id: &str, min_cos: f64, k: usize, stated_by: Option<i64>) -> Result<Vec<Hit>>;
    /// Applies `writes` in order as one atomic commit — one engine operation is one
    /// write, so a crash never leaves half an operation behind.
    fn write(&mut self, writes: &[Write<'_>]) -> Result<()>;
    /// Deletes every trace.
    fn clear(&mut self) -> Result<()>;
}
