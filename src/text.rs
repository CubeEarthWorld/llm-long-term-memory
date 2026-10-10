//! Text hygiene shared by the write and dream paths, and query cue splitting.

/// The whitespace class shared with the Dart engine: Unicode `White_Space`, the
/// information separators U+001C–U+001F and U+FEFF.
fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1C}'..='\u{1F}').contains(&c) || c == '\u{FEFF}'
}

/// Truncates to `max_chars` characters, preferring a sentence/clause boundary in the
/// second half of the cut.
pub fn shorten(text: &str, max_chars: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_chars {
        return text.to_owned();
    }
    let cut = &chars[..max_chars];
    for sep in ['。', '．', '.', '、', ' '] {
        if let Some(idx) = cut.iter().rposition(|&c| c == sep)
            && idx as f64 > max_chars as f64 * 0.5
        {
            return cut[..=idx].iter().collect();
        }
    }
    cut.iter().collect()
}

/// Normalises a proposition for storage: strips the pack delimiters `《》` (so a
/// stored text can never spoof an injected id), collapses whitespace runs to one
/// space, trims and shortens.
pub fn clean_text(text: &str, max_chars: usize) -> String {
    let mut out = String::with_capacity(text.len());
    let mut space = false;
    for c in text.chars().filter(|&c| c != '《' && c != '》') {
        if is_space(c) {
            space = true;
        } else {
            if space && !out.is_empty() {
                out.push(' ');
            }
            space = false;
            out.push(c);
        }
    }
    shorten(&out, max_chars).trim().to_owned() // a boundary cut keeps its separator
}

/// Splits a query into cues (lines, then sentences) so a trace relevant to any part
/// of a long multi-topic turn can surface. More than `max_cues` pieces are coalesced
/// into `max_cues` contiguous buckets (nothing is dropped).
pub fn cues(text: &str, max_cues: usize) -> Vec<String> {
    let mut parts: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut flush = |cur: &mut String| {
        let t = cur.trim();
        if !t.is_empty() {
            parts.push(t.to_owned());
        }
        cur.clear();
    };
    let mut it = text.chars().peekable();
    while let Some(c) = it.next() {
        if c == '\n' {
            flush(&mut cur);
            while it.peek() == Some(&'\n') {
                it.next();
            }
            continue;
        }
        cur.push(c);
        if matches!(c, '。' | '！' | '？') {
            flush(&mut cur);
        } else if matches!(c, '.' | '!' | '?') && it.peek().is_some_and(|&n| is_space(n)) {
            flush(&mut cur);
            while it.peek().is_some_and(|&n| is_space(n)) {
                it.next();
            }
        }
    }
    flush(&mut cur);
    let n = parts.len();
    if n <= max_cues {
        return parts;
    }
    (0..max_cues).map(|i| parts[i * n / max_cues..(i + 1) * n / max_cues].join(" ")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cues_split_lines_and_sentences() {
        assert_eq!(cues("a。b！c", 8), ["a。", "b！", "c"]);
        assert_eq!(cues("one. two? three!\nfour", 8), ["one.", "two?", "three!", "four"]);
        assert_eq!(cues("1.5 kg is fine", 8), ["1.5 kg is fine"]);
        assert_eq!(cues("a\nb\nc\nd\ne", 2), ["a b", "c d e"]);
        assert!(cues("  \n ", 8).is_empty());
    }

    #[test]
    fn shorten_and_clean() {
        assert_eq!(shorten("abcdefghij", 20), "abcdefghij");
        assert_eq!(shorten("これは文です。これは長い文です。おわり", 11), "これは文です。");
        assert_eq!(shorten(&"a".repeat(30), 10), "a".repeat(10));
        assert_eq!(clean_text(" x 《y》\t z ", 100), "x y z");
        assert_eq!(clean_text("\u{FEFF}a\u{1C}b\u{3000}c", 100), "a b c");
    }
}
