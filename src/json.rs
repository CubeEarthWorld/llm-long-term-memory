//! Lenient JSON reading for LLM output: code fences and surrounding prose are tolerated.

use serde_json::Value;

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
    let span = unfenced.char_indices().find_map(|(i, c)| {
        let close = match c {
            '{' => last_brace,
            '[' => last_bracket,
            _ => return None,
        };
        close.filter(|&j| j > i).map(|j| &unfenced[i..=j])
    })?;
    serde_json::from_str(span).ok()
}

/// The non-empty texts of a JSON list whose items are strings or `{"text": …}` objects.
pub fn json_texts(v: &Value) -> Vec<String> {
    let Some(items) = v.as_array() else { return Vec::new() };
    items
        .iter()
        .map(|item| match item {
            Value::Object(o) => o.get("text").map_or(String::new(), scalar),
            other => scalar(other),
        })
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect()
}

fn scalar(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
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
        assert_eq!(json_texts(&json!(["a", " ", {"text": "b"}, null])), ["a", "b"]);
    }
}
