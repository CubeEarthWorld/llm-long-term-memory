//! Seed utterances: the built-in scenario, the `advance` duration syntax and the
//! user-editable CSV (`data/seed.csv`, columns text / note / advance).
//!
//! The scenario mixes stable facts, low-value noise and an update sequence along a
//! *virtual* timeline. `advance` moves the virtual clock *before* the utterance: a
//! number with a unit — `s`, `m`, `h`, `d`, `w`, `y` (365 days; no month unit) — or a
//! bare number of days. It exercises (SPEC §3–§5): noise the LLM should not save; #16
//! restating #2 (rehearsal); #5 → #8 (a meeting moved: a labile pair for the dream);
//! #3 / #6 decaying across the ~10-year gap yet still recalled by #14 / #15; and the
//! meeting / product utterances forming a cluster for consolidation.

use std::path::Path;

/// One seed utterance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeedItem {
    pub text: String,
    pub note: String,
    pub advance: String,
}

impl SeedItem {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({"text": self.text, "note": self.note, "advance": self.advance})
    }
}

const DEFAULT_SEED: [(&str, &str); 17] = [
    ("こんにちは。今日は何をしようかな。", "0"),
    ("私はアイスクリームが好きです。特に抹茶味が好物です。", "0"),
    ("学生時代はずっとバスケットボール部に所属していました。", "30m"),
    ("今日はいい天気ですね。", "2h"),
    ("来週の木曜日に重大な新商品の開発会議が入っています。", "6h"),
    ("新商品のコードネームは「あおぞら」です。", "1d"),
    ("明日は晴れみたいです。", "1d"),
    ("新商品の開発会議は再来週の水曜日に移動しました。", "2d"),
    ("うーん、何をしようかな。", "1d"),
    ("マインクラフトで松明ってどうやって作るんだっけ。", "0"),
    ("再来週に会議って入っていましたっけ？", "0"),
    ("おひさしぶり、最近どうしてる？しばらく話してなかったね。", "5y"),
    ("今日もいい天気だなあ。特に予定はないかな。", "5y"),
    ("ところで、私が学生時代にやっていた部活って何だっけ？", "0"),
    ("新商品「あおぞら」のことって、まだ覚えてる？", "0"),
    ("私はアイスクリームが好きです。", "10y"),
    ("私の好きな食べ物は何か覚えてる？", "500y"),
];

/// The built-in scenario.
pub fn default_seed() -> Vec<SeedItem> {
    clean(DEFAULT_SEED.iter().map(|&(text, advance)| (text.to_owned(), String::new(), advance.to_owned())))
}

/// `'5y' '8d' '12h' '30m'` → seconds; a bare number means days; `'0'`, `''`, junk → 0.
pub fn parse_duration(spec: &str) -> f64 {
    let s = spec.trim().to_lowercase();
    let unit = match s.chars().last() {
        Some('s') => 1.0,
        Some('m') => 60.0,
        Some('h') => 3600.0,
        Some('d') => 86_400.0,
        Some('w') => 604_800.0,
        Some('y') => 31_536_000.0,
        _ => return s.parse::<f64>().map_or(0.0, |v| v * 86_400.0),
    };
    s[..s.len() - 1].trim().parse::<f64>().map_or(0.0, |v| v * unit)
}

/// Normalises (text, note, advance) triples into seed items; empty texts are dropped.
pub fn clean(raw: impl IntoIterator<Item = (String, String, String)>) -> Vec<SeedItem> {
    raw.into_iter()
        .filter(|(t, _, _)| !t.trim().is_empty())
        .map(|(t, n, a)| SeedItem {
            text: t.trim().to_owned(),
            note: n.trim().to_owned(),
            advance: Some(a.trim()).filter(|a| !a.is_empty()).unwrap_or("0").to_owned(),
        })
        .collect()
}

/// Seed items from JSON objects with text / note / advance.
pub fn from_json(items: &[serde_json::Value]) -> Vec<SeedItem> {
    let field = |v: &serde_json::Value, k: &str| match &v[k] {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    };
    clean(items.iter().map(|v| (field(v, "text"), field(v, "note"), field(v, "advance"))))
}

/// RFC 4180 records (quoted fields, doubled quotes, CRLF or LF).
fn csv_records(text: &str) -> Vec<Vec<String>> {
    let (mut records, mut row, mut field) = (Vec::new(), Vec::new(), String::new());
    let (mut quoted, mut chars) = (false, text.chars().peekable());
    while let Some(c) = chars.next() {
        match (quoted, c) {
            (true, '"') if chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            (true, '"') => quoted = false,
            (true, c) => field.push(c),
            (false, '"') if field.is_empty() => quoted = true,
            (false, ',') => row.push(std::mem::take(&mut field)),
            (false, '\r') if chars.peek() == Some(&'\n') => {}
            (false, '\n' | '\r') => {
                row.push(std::mem::take(&mut field));
                records.push(std::mem::take(&mut row));
            }
            (false, c) => field.push(c),
        }
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        records.push(row);
    }
    records.retain(|r| !(r.len() == 1 && r[0].is_empty()));
    records
}

/// Tolerant CSV: a header row with a `text` column (plus optional note / advance), or
/// header-less `text[,advance]` rows.
pub fn parse_csv(text: &str) -> Vec<SeedItem> {
    let records = csv_records(text.trim_start_matches('\u{FEFF}'));
    let Some(header) = records.first() else { return Vec::new() };
    let col = |name: &str| header.iter().position(|h| h.trim().eq_ignore_ascii_case(name));
    let get = |r: &[String], i: Option<usize>| i.and_then(|i| r.get(i)).cloned().unwrap_or_default();
    match col("text") {
        Some(t) => {
            let (n, a) = (col("note"), col("advance"));
            clean(records[1..].iter().map(|r| (get(r, Some(t)), get(r, n), get(r, a))))
        }
        None => clean(records.iter().map(|r| (get(r, Some(0)), String::new(), get(r, Some(1))))),
    }
}

/// The saved CSV if present and parseable, otherwise the built-in scenario.
pub fn load(path: &Path) -> Vec<SeedItem> {
    let items = std::fs::read_to_string(path).map(|t| parse_csv(&t)).unwrap_or_default();
    if items.is_empty() { default_seed() } else { items }
}

fn csv_cell(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_owned() }
}

/// Persists the items (UTF-8 with BOM, so spreadsheet apps read them) to survive restarts.
pub fn save(path: &Path, items: &[SeedItem]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut out = String::from("\u{FEFF}text,note,advance\r\n");
    for i in items {
        out.push_str(&format!("{},{},{}\r\n", csv_cell(&i.text), csv_cell(&i.note), csv_cell(&i.advance)));
    }
    std::fs::write(path, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(text: &str, note: &str, advance: &str) -> SeedItem {
        SeedItem { text: text.into(), note: note.into(), advance: advance.into() }
    }

    #[test]
    fn durations() {
        let got: Vec<f64> = ["0", "", "30m", "2h", "1d", "3", "5y", "junk", "1.5d", "2W"].iter().map(|s| parse_duration(s)).collect();
        assert_eq!(got, [0.0, 0.0, 1800.0, 7200.0, 86_400.0, 3.0 * 86_400.0, 5.0 * 31_536_000.0, 0.0, 129_600.0, 1_209_600.0]);
    }

    #[test]
    fn csv_round_trip_keeps_notes() {
        let path = std::env::temp_dir().join(format!("engram-seed-{}.csv", std::process::id()));
        let _ = std::fs::remove_file(&path);
        assert_eq!(load(&path), default_seed()); // no file → built-in scenario
        let items = clean([(" a,b ".into(), "重要".into(), "5y".into()), (" ".into(), String::new(), String::new()), ("c \"q\"".into(), String::new(), String::new())]);
        assert_eq!(items, [item("a,b", "重要", "5y"), item("c \"q\"", "", "0")]);
        save(&path, &items).unwrap();
        assert_eq!(load(&path), items);
        assert_eq!(parse_csv("\u{FEFF}text,note,advance\nx,n,1d\n"), [item("x", "n", "1d")]);
        assert_eq!(parse_csv("x,2h\ny\n"), [item("x", "", "2h"), item("y", "", "0")]);
        assert_eq!(parse_csv("advance,TEXT\r\n3d,\"multi\nline\"\r\n"), [item("multi\nline", "", "3d")]);
        let _ = std::fs::remove_file(&path);
    }
}
