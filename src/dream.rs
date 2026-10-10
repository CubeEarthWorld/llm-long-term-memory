//! The sleep-phase contract between the engine and an LLM (SPEC §5): the engine
//! hands over a cluster, the [`Adjudicator`] answers keep or replace.

use crate::json::{json_texts, relaxed_json};
use crate::Result;
use serde_json::{json, Value};

/// System prompt for the bundled consolidation prompt.
pub const DREAM_SYSTEM_PROMPT: &str = "You are a memory consolidation engine. Return JSON only.";

const DREAM_INSTRUCTION: &str = concat!(
    "あなたは長期記憶を睡眠中に整理する統合エンジンです(夢フェーズ)。\n",
    "現在時刻: {current_time}\n",
    "**先頭の記憶が新しく符号化された痕跡**で、続く記憶はそれが再活性化した、より古い記憶です。",
    "各記憶には id・内容時刻(local_time/timezone)・想起可能性 R があります。",
    "local_time はその記憶が述べられた時点の時刻です。話題が近いだけの無関係な記憶も混ざります。\n\n",
    "厳守: 入力に存在しない事実を書かないこと(作話禁止)。\n",
    "厳守: 「今日」「明日」「来週」などの相対時間表現は、その記憶の local_time を基準に",
    "絶対日付(YYYY-MM-DD、できれば曜日も)へ変換し、新しい text に相対表現を残さないこと。\n\n",
    "先頭の記憶が、古い記憶のうち**同じ主体の同じ事柄**を更新・訂正・重複しているものを選んでください:\n",
    "- replace: 選んだ古い記憶の id を ids に挙げ、先頭の記憶とそれらを統合した新しい記憶を memories に書く。",
    "矛盾は local_time が新しい記憶を優先し、変化は命題に書き込む(例『2025年は東京、2026年に大阪へ転居』)。",
    "現在時刻より前に終わった予定は過去の事実として書き直す(例『2026年7月に旅行予定』→『2026年7月に旅行した』)。",
    "1つの記憶に複数の事実が詰まっていれば独立した記憶へ分ける。異なる事柄を無理に1つへまとめない。",
    "該当する古い記憶が無くても、先頭の記憶自身を書き直すべきなら ids を空にして replace してよい。\n",
    "- keep: 先頭の記憶はそのままでよく、更新する古い記憶も無い場合。\n\n",
    "話題が近いだけで別の事柄を述べている記憶は ids に挙げないこと(挙げなければ一切変更されません)。\n",
    "各新記憶 text は代名詞を含まない自己完結文・170字以内。「〜時点で確認」のような確認時刻のメタ情報は書かない(事実が変化した場合の日付だけを書く)。出力は JSON オブジェクトのみ:\n",
    "{\"action\": \"replace\", \"ids\": [\"...\", \"...\"], \"memories\": [\"...\", \"...\"]} または {\"action\": \"keep\"}\n\n",
    "# 記憶(先頭が新しい痕跡)\n{listing}\n",
);

/// One trace inside a [`DreamRequest`].
#[derive(Clone, Debug, PartialEq)]
pub struct DreamMember {
    /// Trace id.
    pub id: String,
    /// Proposition text.
    pub text: String,
    /// When it was stated, local (`'2026-05-20 14:00 +09:00'`): the reference for any
    /// relative date words in `text`.
    pub local_time: String,
    /// Stored `'IANA;+HH:MM'` field.
    pub timezone: String,
    /// Retrievability at dream time, rounded to 2 decimals.
    pub retrievability: f64,
}

/// One cluster awaiting a verdict: the seed (newest evidence) first, then the older
/// traces its cue reactivated.
#[derive(Clone, Debug, PartialEq)]
pub struct DreamRequest {
    /// Current local time, same format as [`DreamMember::local_time`].
    pub current_time: String,
    /// The cluster, seed first.
    pub members: Vec<DreamMember>,
}

impl DreamRequest {
    /// The members as the pretty-printed JSON the bundled prompt embeds.
    pub fn listing(&self) -> String {
        let members: Vec<Value> = self
            .members
            .iter()
            .map(|m| json!({"id": m.id, "text": m.text, "local_time": m.local_time, "timezone": m.timezone, "R": m.retrievability}))
            .collect();
        serde_json::to_string_pretty(&members).expect("JSON values serialise")
    }

    /// The complete consolidation prompt (Japanese), ready for any LLM in JSON mode;
    /// pair it with [`DREAM_SYSTEM_PROMPT`] and parse the answer with [`DreamDecision::parse`].
    pub fn prompt(&self) -> String {
        let time = if self.current_time.is_empty() { "(不明)" } else { &self.current_time };
        DREAM_INSTRUCTION.replace("{listing}", &self.listing()).replace("{current_time}", time)
    }
}

/// A verdict: replace the seed and the named candidates with `memories`, or keep
/// (empty `memories`). Candidates not named are untouched.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DreamDecision {
    /// Replacement propositions (empty = keep).
    pub memories: Vec<String>,
    /// Ids of the older candidates the seed supersedes.
    pub absorbed_ids: Vec<String>,
}

impl DreamDecision {
    /// "Leave the cluster as it is."
    pub fn keep() -> Self {
        Self::default()
    }

    /// Parses an LLM answer leniently (fences, prose, `{text}` items). Keep is only an
    /// explicit `"action": "keep"`; anything else must carry at least one text. `None`
    /// for every other answer — empty, unparsable, or `{"memories": []}` with no keep —
    /// so a truncated or confused model is retried by the next dream, never read as
    /// "keep" (SPEC §5).
    pub fn parse(raw: &str) -> Option<Self> {
        let obj = relaxed_json(raw)?;
        let obj = obj.as_object()?;
        let action = obj.get("action").map_or(String::new(), |a| a.as_str().map_or_else(|| a.to_string(), str::to_owned));
        if action.trim().eq_ignore_ascii_case("keep") {
            return Some(Self::keep());
        }
        let memories = json_texts(obj.get("memories").filter(|m| m.is_array())?);
        if memories.is_empty() {
            return None;
        }
        let absorbed_ids = match obj.get("ids").and_then(Value::as_array) {
            Some(ids) => ids
                .iter()
                .map(|i| i.as_str().map_or_else(|| i.to_string(), str::to_owned))
                .filter(|i| !i.is_empty())
                .collect(),
            _ => Vec::new(),
        };
        Some(Self { memories, absorbed_ids })
    }
}

/// The dream-phase LLM: keep or replace one cluster. An error leaves the seed labile,
/// so the next dream retries it.
pub trait Adjudicator {
    /// Judges one cluster.
    fn adjudicate(&mut self, request: &DreamRequest) -> Result<DreamDecision>;
}

impl<F: FnMut(&DreamRequest) -> Result<DreamDecision>> Adjudicator for F {
    fn adjudicate(&mut self, request: &DreamRequest) -> Result<DreamDecision> {
        self(request)
    }
}

#[cfg(test)]
mod tests {
    use super::DreamDecision;

    #[test]
    fn parser_is_lenient_but_never_reads_silence_as_keep() {
        assert_eq!(DreamDecision::parse("```json\n{\"action\":\"keep\"}\n```"), Some(DreamDecision::keep()));
        let v = DreamDecision::parse(r#"x {"action":"replace","ids":["A","B"],"memories":["a",{"text":"b"},""]} y"#).unwrap();
        assert_eq!((v.memories, v.absorbed_ids), (vec!["a".into(), "b".into()], vec!["A".into(), "B".into()]));
        assert!(DreamDecision::parse(r#"{"action":"replace","memories":["a"]}"#).unwrap().absorbed_ids.is_empty());
        assert_eq!(DreamDecision::parse(r#"{"action":"replace","memories":[]}"#), None);
        assert_eq!(DreamDecision::parse(r#"{"memories":[]}"#), None); // no action is not keep
        assert_eq!(DreamDecision::parse(r#"{"action":1,"memories":[" "]}"#), None);
        assert_eq!(DreamDecision::parse(r#"{"action":"replace"}"#), None);
        assert_eq!(DreamDecision::parse("garbage"), None);
        assert_eq!(DreamDecision::parse(""), None);
    }
}
