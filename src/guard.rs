//! The one place the engine's trust in a [`Storage`] ends. A storage is a seam: any
//! database, any vector index, written by anyone. [`Guarded`] holds the contract every
//! answer must meet — at most `limit` rows, the row asked for, no row twice, hits at or
//! stated by the time asked (the cosine is not taken on trust: the engine recomputes it) — so the engine's own code
//! (which counts, indexes and trusts positions) never sees an answer that breaks it.

use crate::storage::{Hit, Storage, View, Write};
use crate::{Result, Trace};
use std::collections::HashSet;

/// A [`Storage`] whose answers are held to the contract.
pub struct Guarded(pub Box<dyn Storage>);

/// `rows` without a second occurrence of any id.
fn distinct<T>(rows: Vec<T>, id: impl Fn(&T) -> &str) -> Vec<T> {
    let mut seen: HashSet<String> = HashSet::new();
    rows.into_iter().filter(|r| seen.insert(id(r).to_owned())).collect()
}

impl Storage for Guarded {
    fn find(&self, view: View<'_>, limit: usize) -> Result<Vec<Trace>> {
        let rows = self.0.find(view, limit)?;
        let rows: Vec<Trace> = match view {
            View::Id(id) => rows.into_iter().filter(|t| t.id == id).collect(),
            View::Text(text) => rows.into_iter().filter(|t| t.text == text).collect(),
            View::After(after) | View::Labile { after, .. } => rows.into_iter().filter(|t| t.id.as_str() > after).collect(),
            _ => rows,
        };
        Ok(distinct(rows, |t| &t.id).into_iter().take(limit).collect())
    }

    fn count(&self, view: View<'_>) -> Result<usize> {
        self.0.count(view)
    }

    fn near(&self, q: &[f32], model_id: &str, min_cos: f64, k: usize, stated_by: Option<i64>) -> Result<Vec<Hit>> {
        let hits = self.0.near(q, model_id, min_cos, k, stated_by)?;
        let hits = hits.into_iter().filter(|h| stated_by.is_none_or(|s| h.trace.created_at <= s)).collect();
        Ok(distinct(hits, |h: &Hit| &h.trace.id).into_iter().take(k).collect())
    }

    fn write(&mut self, writes: &[Write<'_>]) -> Result<()> {
        self.0.write(writes)
    }

    fn clear(&mut self) -> Result<()> {
        self.0.clear()
    }
}
