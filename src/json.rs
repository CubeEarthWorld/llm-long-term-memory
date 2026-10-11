//! Lenient JSON reading for LLM output: code fences and surrounding prose are tolerated.

use serde_json::Value;

/// Openers tried before giving up on a text.
const SPAN_TRIES: usize = 8;

/// Best-effort parse: strips ```` ``` ```` fences, then tries the whole text, then the
/// first `{…}` / `[…]` span. `None` when nothing parses.
pub fn relaxed_json(text: &str) -> Option<Value> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let unfenced = text
        .lines()
        .map(|l| {
            let l = l.strip_prefix("```json").or_else(|| l.strip_prefix("```")).unwrap_or(l);
            l.strip_suffix("```").unwrap_or(l)
        })
        .collect::<Vec<_>>()
        .join("\n");
    let unfenced = unfenced.trim();
    if let Ok(v) = serde_json::from_str(unfenced) {
        return Some(v);
    }
    // The leftmost opener that has a closer after it, through the last such closer (one
    // pass: the last closers are found once, not once per opener).
    let (last_brace, last_bracket) = (unfenced.rfind('}'), unfenced.rfind(']'));
    unfenced
        .char_indices()
        .filter_map(|(i, c)| {
            let close = match c {
                '{' => last_brace,
                '[' => last_bracket,
                _ => return None,
            };
            close.filter(|&j| j > i).map(|j| &unfenced[i..=j])
        })
        .take(SPAN_TRIES) // prose such as "[reason] {…}": the next opener may be the object
        .find_map(|span| serde_json::from_str(span).ok())
}

/// The first `max` non-empty texts of a JSON list whose items are strings or
/// `{"text": "…"}` objects. A number, boolean, list or nested object is not a text.
pub fn json_texts(v: &Value, max: usize) -> Vec<String> {
    let Some(items) = v.as_array() else { return Vec::new() };
    items
        .iter()
        .filter_map(|item| match item {
            Value::String(s) => Some(s.as_str()),
            Value::Object(o) => o.get("text").and_then(Value::as_str),
            _ => None,
        })
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .take(max)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tolerates_fences_and_prose() {
        assert_eq!(relaxed_json("```json\n{\"a\":1}\n```"), Some(json!({"a": 1})));
        assert_eq!(relaxed_json("x {\"a\":[1]} y"), Some(json!({"a": [1]})));
        assert_eq!(relaxed_json("garbage"), None);
        assert_eq!(relaxed_json(""), None);
        assert_eq!(json_texts(&json!(["a", " ", {"text": "b"}, null, 1, true, [2], {"text": 3}]), 8), ["a", "b"]);
        assert_eq!(json_texts(&json!(["a", "b", "c"]), 2), ["a", "b"]);
    }
}
