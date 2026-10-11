//! Round 3, lane B: SPEC-vs-code conformance of the numbers SPEC states, and the documents
//! around the engine (README examples, README API names). Each test names the sentence it checks.
//! See audit/round3_b.md for the full conformance matrix and audit/lean/R3B.lean for the
//! arithmetic behind `band_window_total_reads_match_spec_section_3`.

mod support;

use engram::{Hit, Result, Storage, Store, Trace, View, Write};
use std::sync::{Arc, Mutex};
use support::{build_with, cfg, temp_dir, FakeEmbedder, DAY, START};

/// `Store` plus a log of how many rows each `find(View::Band)` returned.
struct Counting {
    inner: Store,
    band_reads: Arc<Mutex<Vec<(u8, usize)>>>,
}

impl Storage for Counting {
    fn find(&self, view: View<'_>, limit: usize) -> Result<Vec<Trace>> {
        let rows = self.inner.find(view, limit)?;
        if let View::Band { band, .. } = view {
            self.band_reads.lock().unwrap().push((band, rows.len()));
        }
        Ok(rows)
    }
    fn count(&self, view: View<'_>) -> Result<usize> {
        self.inner.count(view)
    }
    fn near(&self, q: &[f32], model_id: &str, min_cos: f64, k: usize, stated_by: Option<i64>) -> Result<Vec<Hit>> {
        self.inner.near(q, model_id, min_cos, k, stated_by)
    }
    fn write(&mut self, writes: &[Write<'_>]) -> Result<()> {
        self.inner.write(writes)
    }
    fn clear(&mut self) -> Result<()> {
        self.inner.clear()
    }
}

/// `rows` strong, old traces that all fall in one stability band (S = 1400 s: band 20, rank
/// `log2 S` = 10.45 against the band's lower bound 10.0, so every window keeps growing until
/// it hits its cap), stored behind the engine's back. `idle` = seconds since their last recall.
/// One `remember` then has to evict `rows + 1 - capacity` traces. Returns the rows each
/// `find(Band)` returned for band 20.
fn band_reads_for_one_eviction(rows: usize, capacity: usize, idle: f64) -> Vec<usize> {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut store = Counting { inner: Store::in_memory().unwrap(), band_reads: log.clone() };
    let mut embedder = FakeEmbedder::new();
    let traces: Vec<Trace> = (0..rows)
        .map(|i| {
            let text = format!("row{i}");
            Trace {
                id: format!("{i:06}"),
                vector: embedder.embed(&text),
                text,
                created_at: (START - 30.0 * DAY) as i64,
                tz: "UTC;+00:00".into(),
                last_recall: (START - idle) as i64,
                stability: 1400.0,
                consolidated: true,
                cue: String::new(),
                model_id: "fake/token-overlap".into(),
            }
        })
        .collect();
    assert!(traces.iter().all(|t| t.band() == 20), "setup: one band");
    let refs: Vec<Write<'_>> = traces.iter().map(Write::Insert).collect();
    store.write(&refs).unwrap();
    let (mut m, _clock) = build_with(engram::Config { capacity, ..cfg() }, Box::new(FakeEmbedder::new()), Box::new(store));
    log.lock().unwrap().clear();
    m.remember("brand new fact", 1.0, "").unwrap();
    
    log.lock().unwrap().iter().filter(|(b, _)| *b == 20).map(|(_, n)| *n).collect()
}

/// SPEC §3: "1 帯の窓は 128 件（excess + 8 がそれより大きければそれ）が上限（窓は 4 倍ずつ広げて
/// 読み直すので、読み出しの総量は上限の約 1.4 倍まで）".
/// The 128 cap on one read is true. The 1.4x total is not: at excess = 23 the window is read as
/// 31, 124, 128 rows = 283 = 2.2 x the cap (audit/lean/R3B.lean proves 283 is the worst case).
#[test]
fn band_window_total_reads_match_spec_section_3() {
    let reads = band_reads_for_one_eviction(300, 278, 0.0); // excess = 301 - 278 = 23
    assert!(reads.iter().all(|&n| n <= 128), "a single read exceeds BAND_CAP: {reads:?}");
    let total: usize = reads.iter().sum();
    // SPEC §3 (corrected): the windows grow 4x each time, so one band is read at most about 2.2 x the cap.
    assert!(total as f64 <= 2.3 * 128.0, "one eviction read {reads:?} = {total} rows ({:.2} x the cap)", total as f64 / 128.0);
}

/// SPEC §3: the first read is `excess + 8` rows, and a window whose `excess`-th weakest rank is
/// already <= -64 (dead) is not widened. (Holds.)
#[test]
fn dead_band_is_not_widened_and_first_window_is_excess_plus_8() {
    let reads = band_reads_for_one_eviction(300, 278, 100.0 * 1400.0); // rank = 10.45 - 100 < -64
    assert_eq!(reads, [31]);
}

/// SPEC §3: "excess + 8 がそれより大きければそれ" - a window larger than 128 is read once, whole. (Holds.)
#[test]
fn window_larger_than_128_is_read_once() {
    let reads = band_reads_for_one_eviction(300, 100, 0.0); // excess = 201, window = 209
    assert_eq!(reads, [209]);
}

/// SPEC §8: "id は 48bit ミリ秒（…それ以後は id が巻き戻る）". With the monotonic generator the
/// ids do NOT roll back after the clamp, nor when time goes backwards: the SPEC sentence is
/// wrong for a live engine (it holds only across a restart). Documents the actual behaviour.
#[test]
fn ids_do_not_roll_back_past_the_48_bit_clamp_in_one_process() {
    let mut ids = engram::ulid_ids();
    let far = [ids(i64::MAX, ""), ids(i64::MAX, ""), ids((1i64 << 48) - 1, ""), ids(5, "")];
    assert!(far.windows(2).all(|w| w[0] < w[1]), "{far:?}");
    assert!(far.iter().all(|id| id.len() == 26));
}

/// README.md / README.ja.md "Using the library": the snippet, verbatim, must compile against the
/// public API and run.
#[test]
fn readme_quickstart_compiles_and_runs() {
    fn quickstart(my_embedder: FakeEmbedder, llm_reply: &str, my_llm: impl Fn(&str, &str) -> String, path: &str) -> engram::Result<String> {
        use engram::{Config, DreamDecision, DreamRequest, Engram, Store};
        let mut memory = Engram::open(Box::new(Store::open(path.as_ref())?), Box::new(my_embedder), Config::default(), "Asia/Tokyo")?;
        memory.remember("The user moved to Osaka in April 2026.", 1.0, "Where does the user live?")?;
        let pack = memory.recall("where do I live now?")?.pack_text; // inject into your prompt
        memory.cite(llm_reply)?; // strengthen what was used
        memory.dream(&mut |req: &DreamRequest| {
            // any LLM in JSON mode
            DreamDecision::parse(&my_llm(engram::DREAM_SYSTEM_PROMPT, &req.prompt())).ok_or("unparsable".into())
        }, None)?;
        Ok(pack)
    }
    let dir = temp_dir("readme");
    let path = dir.join("memory.redb");
    let pack = quickstart(FakeEmbedder::new(), "", |_, _| r#"{"action":"keep"}"#.into(), path.to_str().unwrap()).unwrap();
    assert!(pack.is_empty() || pack.contains("《id:"));
}

/// README.md and README.ja.md: "All 19 engine parameters live in `engram::Config` (JSON round trip
/// with `to_json` / `merge_json`)". `Config` has no such methods (grep src/ bench/src/: none).
#[test]
fn readmes_name_only_methods_that_exist() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for file in ["README.md", "README.ja.md", "README.zh.md"] {
        let text = std::fs::read_to_string(root.join(file)).unwrap();
        for stale in ["to_json", "merge_json"] {
            assert!(!text.contains(stale), "{file} mentions `{stale}`, which engram::Config does not have");
        }
    }
}

/// `Engram::clusters` doc: "The clusters the next dream would hand to the LLM". It returns the
/// cluster of every one of the next 8 x budget seeds (the bench's supersede measurement relies on
/// that), while the dream judges at most `budget` of them: documents the real behaviour (the doc
/// comment, not the code, needs the correction).
#[test]
fn clusters_preview_lists_more_clusters_than_the_dream_would_judge() {
    let (mut m, _clock) = build_with(cfg(), Box::new(FakeEmbedder::new()), Box::new(Store::in_memory().unwrap()));
    for g in 0..3 {
        let base = format!("aaa{g} bbb{g} ccc{g} ddd{g}");
        support::insert(&mut m, &base, 1.0, "");
        support::insert(&mut m, &format!("{base} eee{g}"), 1.0, "");
    }
    let preview = m.clusters(Some(1)).unwrap().len();
    let mut judged = 0;
    m.dream(&mut |_: &engram::DreamRequest| { judged += 1; Ok(engram::DreamDecision::keep()) }, Some(1)).unwrap();
    assert_eq!((preview, judged), (3, 1));
}
