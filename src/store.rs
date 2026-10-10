//! The bundled storage, pure Rust: one [redb](https://docs.rs/redb) file (ACID, B-trees)
//! holding the traces, one table per view, and an inverted-file (IVF) similarity index.
//!
//! Vectors live in lists, one per centroid. A search reads the lists whose centroids
//! are nearest the query, so it touches a few lists rather than every vector; a list
//! that grows past [`LIST_MAX`] is split in two by 2-means over its own members, so
//! the index refines itself as it grows and never needs a rebuild. Nothing is loaded
//! at open except, lazily, the centroids (a few per thousand traces). With one list —
//! under [`LIST_MAX`] traces of a model — search is exact.

use crate::storage::{Hit, Storage, View, Write};
use crate::vector::{dot, normalized};
use crate::{Result, Trace};
use redb::{Database, ReadableTable, TableDefinition, WriteTransaction};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;
use std::sync::RwLock;

/// Vectors per similarity list before it is split in two.
pub const LIST_MAX: u64 = 512;
/// Pages redb keeps in RAM (the OS page cache keeps the rest warm).
const CACHE_BYTES: usize = 1 << 30;
/// Vectors a similarity search reads, from the lists with the nearest centroids: a
/// model with fewer is searched exactly, a larger one in about the same time.
const PROBE_VECTORS: u64 = 8192;

/// id → the trace without its vector (JSON).
const ROWS: TableDefinition<&str, &[u8]> = TableDefinition::new("rows");
/// text → id.
const TEXT: TableDefinition<&str, &str> = TableDefinition::new("text");
/// (model, id) of the labile traces.
const LABILE: TableDefinition<(&str, &str), ()> = TableDefinition::new("labile");
/// (model, id) of every trace.
const MODEL: TableDefinition<(&str, &str), ()> = TableDefinition::new("model");
/// (band, last_recall, id) → created_at.
const BAND: TableDefinition<(u8, i64, &str), i64> = TableDefinition::new("band");
/// (created_at, id).
const CREATED: TableDefinition<(i64, &str), ()> = TableDefinition::new("created");
/// id → the list its vector is in.
const LISTED: TableDefinition<&str, u32> = TableDefinition::new("listed");
/// (model, list, id) → created_at (8 bytes) + vector (f32 little-endian).
const LISTS: TableDefinition<(&str, u32, &str), &[u8]> = TableDefinition::new("lists");
/// (model, list) → size (8 bytes) + centroid (f32 little-endian).
const CENTROIDS: TableDefinition<(&str, u32), &[u8]> = TableDefinition::new("centroids");
/// "count", "next:<model>".
const META: TableDefinition<&str, u64> = TableDefinition::new("meta");

/// One list's centroid and size.
#[derive(Clone, Debug)]
struct List {
    id: u32,
    size: u64,
    centroid: Vec<f32>,
}

/// The bundled [`Storage`]: a redb file with an IVF index (see the module docs).
pub struct Store {
    db: Database,
    lists: RwLock<HashMap<String, Vec<List>>>,
}

fn floats(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn meta(t: &Trace) -> Vec<u8> {
    json!([t.text, t.created_at, t.tz, t.last_recall, t.stability, t.consolidated, t.cue, t.model_id]).to_string().into_bytes()
}

fn trace(id: &str, row: &[u8], vector: Vec<f32>) -> Option<Trace> {
    let v: Value = serde_json::from_slice(row).ok()?;
    Some(Trace {
        id: id.to_owned(),
        text: v[0].as_str()?.to_owned(),
        created_at: v[1].as_i64()?,
        tz: v[2].as_str()?.to_owned(),
        last_recall: v[3].as_i64()?,
        stability: v[4].as_f64()?,
        consolidated: v[5].as_bool()?,
        cue: v[6].as_str()?.to_owned(),
        model_id: v[7].as_str()?.to_owned(),
        vector,
    })
}

fn list_entry(created_at: i64, v: &[f32]) -> Vec<u8> {
    created_at.to_le_bytes().into_iter().chain(bytes(v)).collect()
}

fn centroid_entry(size: u64, c: &[f32]) -> Vec<u8> {
    size.to_le_bytes().into_iter().chain(bytes(c)).collect()
}

fn decode_list(list: u32, raw: &[u8]) -> List {
    List { id: list, size: u64::from_le_bytes(raw[..8].try_into().expect("8 bytes")), centroid: floats(&raw[8..]) }
}

/// Creates every table (a no-op for those that exist).
fn create_tables(tx: &WriteTransaction) -> Result<()> {
    tx.open_table(ROWS)?;
    tx.open_table(TEXT)?;
    tx.open_table(LABILE)?;
    tx.open_table(MODEL)?;
    tx.open_table(BAND)?;
    tx.open_table(CREATED)?;
    tx.open_table(LISTED)?;
    tx.open_table(LISTS)?;
    tx.open_table(CENTROIDS)?;
    tx.open_table(META)?;
    Ok(())
}

impl Store {
    /// Opens (or creates) the store file at `path`.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        Self::init(Database::builder().set_cache_size(CACHE_BYTES).create(path)?)
    }

    /// A store that lives and dies with the process (tests, benchmarks, ephemeral agents).
    pub fn in_memory() -> Result<Self> {
        Self::init(Database::builder().create_with_backend(redb::backends::InMemoryBackend::new())?)
    }

    fn init(db: Database) -> Result<Self> {
        let tx = db.begin_write()?;
        create_tables(&tx)?;
        tx.commit()?;
        Ok(Self { db, lists: RwLock::new(HashMap::new()) })
    }

    /// The lists of `model` (centroids are read once, then kept).
    fn lists_of(&self, model: &str) -> Result<Vec<List>> {
        if let Some(l) = self.lists.read().unwrap_or_else(|e| e.into_inner()).get(model) {
            return Ok(l.clone());
        }
        let rx = self.db.begin_read()?;
        let table = rx.open_table(CENTROIDS)?;
        let mut lists = Vec::new();
        for item in table.range((model, 0u32)..=(model, u32::MAX))? {
            let (k, v) = item?;
            lists.push(decode_list(k.value().1, v.value()));
        }
        self.lists.write().unwrap_or_else(|e| e.into_inner()).insert(model.to_owned(), lists.clone());
        Ok(lists)
    }

    /// Full traces for `ids`, in that order (missing ids are skipped).
    fn rows(&self, ids: &[String]) -> Result<Vec<Trace>> {
        let rx = self.db.begin_read()?;
        let (rows, listed, lists) = (rx.open_table(ROWS)?, rx.open_table(LISTED)?, rx.open_table(LISTS)?);
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            let Some(row) = rows.get(id.as_str())? else { continue };
            let Some(mut t) = trace(id, row.value(), Vec::new()) else { continue };
            if let Some(list) = listed.get(id.as_str())?
                && let Some(entry) = lists.get((t.model_id.as_str(), list.value(), id.as_str()))?
            {
                t.vector = floats(&entry.value()[8..]);
            }
            out.push(t);
        }
        Ok(out)
    }

    /// Ids of `view`, in the view's order, at most `limit`.
    fn ids(&self, view: View<'_>, limit: usize) -> Result<Vec<String>> {
        let rx = self.db.begin_read()?;
        let mut out = Vec::new();
        match view {
            View::All => {
                for item in rx.open_table(ROWS)?.iter()?.take(limit) {
                    out.push(item?.0.value().to_owned());
                }
            }
            View::Id(id) => {
                if rx.open_table(ROWS)?.get(id)?.is_some() {
                    out.push(id.to_owned());
                }
            }
            View::Text(text) => {
                if let Some(id) = rx.open_table(TEXT)?.get(text)? {
                    out.push(id.value().to_owned());
                }
            }
            View::Labile { model_id } => {
                for item in rx.open_table(LABILE)?.range((model_id, "")..)?.take(limit) {
                    let (k, _) = item?;
                    let (m, id) = k.value();
                    if m != model_id {
                        break;
                    }
                    out.push(id.to_owned());
                }
            }
            View::Stale { model_id } => {
                let table = rx.open_table(MODEL)?;
                for range in [table.range(..(model_id, ""))?, table.range((model_id, "\u{10FFFF}")..)?] {
                    for item in range.rev().take(limit) {
                        out.push(item?.0.value().1.to_owned());
                    }
                }
                out.sort_unstable_by(|a, b| b.cmp(a));
                out.truncate(limit);
            }
            View::Band { band, created_by } => {
                for item in rx.open_table(BAND)?.range((band, i64::MIN, "")..=(band, i64::MAX, "\u{10FFFF}"))? {
                    if out.len() >= limit {
                        break;
                    }
                    let (k, created_at) = item?;
                    if created_at.value() <= created_by {
                        out.push(k.value().2.to_owned());
                    }
                }
            }
            View::Young { created_after } => {
                for item in rx.open_table(CREATED)?.range((created_after.saturating_add(1), "")..)?.take(limit) {
                    out.push(item?.0.value().1.to_owned());
                }
            }
        }
        Ok(out)
    }
}

/// One write transaction's tables, and the lists it changed.
struct Tx<'t> {
    rows: redb::Table<'t, &'static str, &'static [u8]>,
    text: redb::Table<'t, &'static str, &'static str>,
    labile: redb::Table<'t, (&'static str, &'static str), ()>,
    model: redb::Table<'t, (&'static str, &'static str), ()>,
    band: redb::Table<'t, (u8, i64, &'static str), i64>,
    created: redb::Table<'t, (i64, &'static str), ()>,
    listed: redb::Table<'t, &'static str, u32>,
    lists: redb::Table<'t, (&'static str, u32, &'static str), &'static [u8]>,
    centroids: redb::Table<'t, (&'static str, u32), &'static [u8]>,
    meta: redb::Table<'t, &'static str, u64>,
    touched: HashMap<String, Vec<List>>,
}

impl Tx<'_> {
    fn get(&self, id: &str) -> Result<Option<Trace>> {
        let Some(row) = self.rows.get(id)? else { return Ok(None) };
        let mut t = trace(id, row.value(), Vec::new()).ok_or("unreadable row")?;
        if let Some(list) = self.listed.get(id)?
            && let Some(entry) = self.lists.get((t.model_id.as_str(), list.value(), id))?
        {
            t.vector = floats(&entry.value()[8..]);
        }
        Ok(Some(t))
    }

    fn bump(&mut self, key: &str, delta: i64) -> Result<u64> {
        let n = self.meta.get(key)?.map_or(0, |v| v.value()).saturating_add_signed(delta);
        self.meta.insert(key, n)?;
        Ok(n)
    }

    /// The lists of `model` as this transaction sees them.
    fn lists_of(&mut self, model: &str) -> Result<&mut Vec<List>> {
        if !self.touched.contains_key(model) {
            let mut lists = Vec::new();
            for item in self.centroids.range((model, 0u32)..=(model, u32::MAX))? {
                let (k, v) = item?;
                lists.push(decode_list(k.value().1, v.value()));
            }
            self.touched.insert(model.to_owned(), lists);
        }
        Ok(self.touched.get_mut(model).expect("just inserted"))
    }

    fn save_list(&mut self, model: &str, l: &List) -> Result<()> {
        if l.size == 0 {
            self.centroids.remove((model, l.id))?;
        } else {
            self.centroids.insert((model, l.id), centroid_entry(l.size, &l.centroid).as_slice())?;
        }
        Ok(())
    }

    /// Files the vector of `t` in the list with the nearest centroid; splits it when full.
    fn list(&mut self, t: &Trace) -> Result<()> {
        let model = t.model_id.as_str();
        let lists = self.lists_of(model)?;
        let best = lists
            .iter()
            .enumerate()
            .filter(|(_, l)| l.centroid.len() == t.vector.len())
            .max_by(|a, b| dot(&a.1.centroid, &t.vector).total_cmp(&dot(&b.1.centroid, &t.vector)))
            .map(|(i, _)| i);
        let at = match best {
            Some(i) => i,
            None => {
                let id = u32::try_from(self.bump(&format!("next:{model}"), 1)?)?;
                let lists = self.lists_of(model)?;
                lists.push(List { id, size: 0, centroid: t.vector.clone() });
                lists.len() - 1
            }
        };
        let lists = self.lists_of(model)?;
        lists[at].size += 1;
        let l = lists[at].clone();
        self.lists.insert((model, l.id, t.id.as_str()), list_entry(t.created_at, &t.vector).as_slice())?;
        self.listed.insert(t.id.as_str(), l.id)?;
        self.save_list(model, &l)?;
        if l.size > LIST_MAX {
            self.split(model, l.id)?;
        }
        Ok(())
    }

    /// Takes `id`'s vector out of its list.
    fn unlist(&mut self, id: &str, model: &str) -> Result<()> {
        let Some(list) = self.listed.remove(id)?.map(|g| g.value()) else { return Ok(()) };
        self.lists.remove((model, list, id))?;
        let lists = self.lists_of(model)?;
        if let Some(i) = lists.iter().position(|l| l.id == list) {
            lists[i].size = lists[i].size.saturating_sub(1);
            let l = lists[i].clone();
            if l.size == 0 {
                lists.remove(i);
            }
            self.save_list(model, &l)?;
        }
        Ok(())
    }

    /// Splits a full list in two by 2-means over its members (a few passes over at most
    /// `LIST_MAX + 1` vectors: the cost of one split is bounded, whatever the store size).
    fn split(&mut self, model: &str, list: u32) -> Result<()> {
        let mut members: Vec<(String, i64, Vec<f32>)> = Vec::new();
        for item in self.lists.range((model, list, "")..=(model, list, "\u{10FFFF}"))? {
            let (k, v) = item?;
            let raw = v.value();
            members.push((k.value().2.to_owned(), i64::from_le_bytes(raw[..8].try_into()?), floats(&raw[8..])));
        }
        let Some(first) = members.first().map(|m| m.2.clone()) else { return Ok(()) };
        let far = members.iter().min_by(|a, b| dot(&a.2, &first).total_cmp(&dot(&b.2, &first))).expect("members").2.clone();
        let mut c = [first, far];
        let mut side = vec![false; members.len()];
        for _ in 0..4 {
            for (s, m) in side.iter_mut().zip(&members) {
                *s = dot(&m.2, &c[1]) > dot(&m.2, &c[0]);
            }
            for (k, cent) in c.iter_mut().enumerate() {
                let mut sum = vec![0.0f64; cent.len()];
                for (m, _) in members.iter().zip(&side).filter(|(_, s)| **s == (k == 1)) {
                    for (a, &x) in sum.iter_mut().zip(&m.2) {
                        *a += f64::from(x);
                    }
                }
                *cent = normalized(&sum.iter().map(|&x| x as f32).collect::<Vec<_>>());
            }
        }
        let moved = side.iter().filter(|&&s| s).count() as u64;
        if moved == 0 || moved == members.len() as u64 {
            return Ok(()); // indivisible (duplicates): leave it whole
        }
        let new = u32::try_from(self.bump(&format!("next:{model}"), 1)?)?;
        for (m, _) in members.iter().zip(&side).filter(|(_, s)| **s) {
            self.lists.remove((model, list, m.0.as_str()))?;
            self.lists.insert((model, new, m.0.as_str()), list_entry(m.1, &m.2).as_slice())?;
            self.listed.insert(m.0.as_str(), new)?;
        }
        let lists = self.lists_of(model)?;
        let i = lists.iter().position(|l| l.id == list).expect("listed");
        lists[i].size -= moved;
        lists[i].centroid = c[0].clone();
        let (old, fresh) = (lists[i].clone(), List { id: new, size: moved, centroid: c[1].clone() });
        lists.push(fresh.clone());
        self.save_list(model, &old)?;
        self.save_list(model, &fresh)
    }

    fn link(&mut self, t: &Trace) -> Result<()> {
        self.rows.insert(t.id.as_str(), meta(t).as_slice())?;
        if self.text.get(t.text.as_str())?.is_none() {
            self.text.insert(t.text.as_str(), t.id.as_str())?;
        }
        self.model.insert((t.model_id.as_str(), t.id.as_str()), ())?;
        if !t.consolidated {
            self.labile.insert((t.model_id.as_str(), t.id.as_str()), ())?;
        }
        self.band.insert((t.band(), t.last_recall, t.id.as_str()), t.created_at)?;
        self.created.insert((t.created_at, t.id.as_str()), ())?;
        if !t.vector.is_empty() {
            self.list(t)?;
        }
        Ok(())
    }

    fn unlink(&mut self, t: &Trace) -> Result<()> {
        self.rows.remove(t.id.as_str())?;
        if self.text.get(t.text.as_str())?.is_some_and(|id| id.value() == t.id) {
            self.text.remove(t.text.as_str())?;
        }
        self.model.remove((t.model_id.as_str(), t.id.as_str()))?;
        self.labile.remove((t.model_id.as_str(), t.id.as_str()))?;
        self.band.remove((t.band(), t.last_recall, t.id.as_str()))?;
        self.created.remove((t.created_at, t.id.as_str()))?;
        self.unlist(&t.id, &t.model_id)
    }

    fn apply(&mut self, w: &Write<'_>) -> Result<()> {
        match *w {
            Write::Insert(t) => {
                match self.get(&t.id)? {
                    Some(old) => self.unlink(&old)?,
                    None => {
                        self.bump("count", 1)?;
                    }
                }
                self.link(t)
            }
            Write::State { id, last_recall, stability, consolidated } => {
                let Some(old) = self.get(id)? else { return Ok(()) };
                self.unlink(&old)?;
                self.link(&Trace { last_recall, stability, consolidated, ..old })
            }
            Write::Index { id, model_id, vector } => {
                let Some(old) = self.get(id)? else { return Ok(()) };
                self.unlink(&old)?;
                self.link(&Trace { model_id: model_id.to_owned(), vector: vector.to_vec(), ..old })
            }
            Write::Remove(id) => {
                if let Some(old) = self.get(id)? {
                    self.unlink(&old)?;
                    self.bump("count", -1)?;
                }
                Ok(())
            }
        }
    }
}

impl Storage for Store {
    fn find(&self, view: View<'_>, limit: usize) -> Result<Vec<Trace>> {
        self.rows(&self.ids(view, limit)?)
    }

    fn count(&self, view: View<'_>) -> Result<usize> {
        let rx = self.db.begin_read()?;
        Ok(match view {
            View::All => rx.open_table(META)?.get("count")?.map_or(0, |v| v.value()) as usize,
            View::Young { created_after } => rx.open_table(CREATED)?.range((created_after.saturating_add(1), "")..)?.count(),
            View::Stale { model_id } => {
                let table = rx.open_table(MODEL)?;
                table.range(..(model_id, ""))?.count() + table.range((model_id, "\u{10FFFF}")..)?.count()
            }
            other => self.ids(other, usize::MAX)?.len(),
        })
    }

    /// Reads the lists with the nearest centroids until [`PROBE_VECTORS`] vectors have been
    /// read, so the cost is bounded (plus one dot product per centroid), not the store size.
    fn near(&self, q: &[f32], model_id: &str, min_cos: f64, k: usize, stated_by: Option<i64>) -> Result<Vec<Hit>> {
        let mut lists = self.lists_of(model_id)?;
        lists.retain(|l| l.centroid.len() == q.len());
        lists.sort_by(|a, b| dot(&b.centroid, q).total_cmp(&dot(&a.centroid, q)));
        let rx = self.db.begin_read()?;
        let table = rx.open_table(LISTS)?;
        let mut best: Vec<(f64, String)> = Vec::new();
        let mut buf = vec![0.0f32; q.len()]; // one buffer: no allocation per vector read
        let mut read = 0;
        for l in &lists {
            if read >= PROBE_VECTORS {
                break;
            }
            read += l.size;
            for item in table.range((model_id, l.id, "")..=(model_id, l.id, "\u{10FFFF}"))? {
                let (key, v) = item?;
                let raw = v.value();
                if stated_by.is_some_and(|s| i64::from_le_bytes(raw[..8].try_into().expect("8 bytes")) > s) || raw.len() != 8 + 4 * q.len() {
                    continue;
                }
                for (x, c) in buf.iter_mut().zip(raw[8..].as_chunks::<4>().0) {
                    *x = f32::from_le_bytes(*c);
                }
                let cos = dot(&buf, q);
                if cos >= min_cos {
                    best.push((cos, key.value().2.to_owned()));
                }
            }
        }
        let order = |a: &(f64, String), b: &(f64, String)| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1));
        if best.len() > k && k > 0 {
            best.select_nth_unstable_by(k - 1, order);
        }
        best.truncate(k);
        best.sort_by(order);
        let ids: Vec<String> = best.iter().map(|b| b.1.clone()).collect();
        let traces = self.rows(&ids)?;
        Ok(traces.into_iter().zip(best).map(|(trace, (cos, _))| Hit { cos, trace }).collect())
    }

    /// One atomic commit. Durability is "eventual" (like SQLite's WAL with
    /// `synchronous = NORMAL`): a crash can lose the last commits, never corrupt the file.
    fn write(&mut self, writes: &[Write<'_>]) -> Result<()> {
        let mut wx = self.db.begin_write()?;
        wx.set_durability(redb::Durability::Eventual);
        let touched = {
            let mut tx = Tx {
                rows: wx.open_table(ROWS)?,
                text: wx.open_table(TEXT)?,
                labile: wx.open_table(LABILE)?,
                model: wx.open_table(MODEL)?,
                band: wx.open_table(BAND)?,
                created: wx.open_table(CREATED)?,
                listed: wx.open_table(LISTED)?,
                lists: wx.open_table(LISTS)?,
                centroids: wx.open_table(CENTROIDS)?,
                meta: wx.open_table(META)?,
                touched: HashMap::new(),
            };
            for w in writes {
                tx.apply(w)?;
            }
            tx.touched
        };
        wx.commit()?;
        let mut cache = self.lists.write().unwrap_or_else(|e| e.into_inner());
        cache.extend(touched);
        Ok(())
    }

    fn clear(&mut self) -> Result<()> {
        let wx = self.db.begin_write()?;
        wx.delete_table(ROWS)?;
        wx.delete_table(TEXT)?;
        wx.delete_table(LABILE)?;
        wx.delete_table(MODEL)?;
        wx.delete_table(BAND)?;
        wx.delete_table(CREATED)?;
        wx.delete_table(LISTED)?;
        wx.delete_table(LISTS)?;
        wx.delete_table(CENTROIDS)?;
        wx.delete_table(META)?;
        create_tables(&wx)?;
        wx.commit()?;
        self.lists.write().unwrap_or_else(|e| e.into_inner()).clear();
        Ok(())
    }
}
