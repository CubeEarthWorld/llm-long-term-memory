//! Claude through Claude Code (`claude -p`), with a prompt-keyed response cache.
//!
//! Every answer is stored under a hash of (model, system prompt, prompt), so a test or
//! benchmark that replays the same scenario asks the model once and then reuses the
//! same data. When the CLI cannot answer (not installed, not signed in) a missing
//! prompt is appended to `<cache>.pending.json` and the call fails — a dream leaves the
//! seed labile, a turn answers without saving — so the pending prompts can be answered
//! out of band (any Claude client) and merged into the cache.

use crate::llm::{build_prompt, parse_texts, retry, CallError, Converse, Llm, ToolFn, EXTRACT_INSTRUCTION, EXTRACT_SYSTEM_PROMPT, SYSTEM_PROMPT};
use crate::settings::{root, Settings};
use engram::{relaxed_json, DreamDecision, DreamRequest, DREAM_SYSTEM_PROMPT};
use serde_json::{json, Map, Value};
use std::io::Write as _;
use std::path::PathBuf;
use std::process::{Command, Stdio};

/// How the conversation turn returns its tool calls: one JSON object instead of a
/// function-calling round trip.
const TOOL_PROTOCOL: &str = concat!(
    "\n\n# 出力形式\n",
    "ツールは呼べません。代わりに次の JSON オブジェクトだけを返してください:\n",
    "{\"reply\": \"ユーザーへの応答（使った記憶の《id:...》引用を含む）\", ",
    "\"save_memory\": [{\"text\": \"保存する自己完結した1命題\", \"cue\": \"この事実を後で問い直す質問文（値・固有名詞・日付を含めない）\", \"salience\": 1}], ",
    "\"delete_memory\": [\"削除する記憶の id\"]}\n",
    "保存・削除が無ければ空配列にしてください。",
);

/// Longest one CLI call may take.
const CLI_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

/// FNV-1a over the parts (stable across runs and platforms, unlike `RandomState`).
fn key(parts: &[&str]) -> String {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for p in parts {
        for b in p.bytes().chain([0xFF]) {
            h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01B3);
        }
    }
    format!("{h:016x}")
}

/// Claude Haiku (or any Claude model) via the Claude Code CLI.
pub struct ClaudeCode {
    cli: String,
    model: String,
    cache_path: PathBuf,
    cache: Map<String, Value>,
    cli_error: Option<String>,
}

impl ClaudeCode {
    pub fn new(s: &Settings) -> Self {
        let path = PathBuf::from(&s.llm_cache);
        let cache_path = if path.is_absolute() { path } else { root().join(path) };
        let cache = std::fs::read_to_string(&cache_path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
        Self { cli: s.claude_cli.clone(), model: s.claude_model.clone(), cache_path, cache, cli_error: None }
    }

    fn pending_path(&self) -> PathBuf {
        self.cache_path.with_extension("pending.json")
    }

    fn save(&self, path: &PathBuf, map: &Map<String, Value>) {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        // Write a sibling file, then rename it over the old one: a crash never leaves half a cache.
        let tmp = path.with_extension("tmp");
        let written = std::fs::write(&tmp, serde_json::to_string_pretty(map).unwrap_or_default()).and_then(|()| std::fs::rename(&tmp, path));
        if let Err(e) = written {
            eprintln!("could not write {}: {e}", path.display());
        }
    }

    /// One headless Claude Code call without tools or session state.
    fn call_cli(&self, system: &str, prompt: &str) -> Result<String, CallError> {
        let mut child = Command::new(&self.cli)
            .args(["-p", "--model", &self.model, "--output-format", "json", "--tools", "", "--no-session-persistence", "--system-prompt", system])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| CallError::Fatal(format!("cannot run {}: {e}", self.cli)))?;
        child.stdin.take().expect("piped").write_all(prompt.as_bytes()).map_err(|e| CallError::Fatal(e.to_string()))?;
        // A hung CLI must not hold the session forever: give up after CLI_TIMEOUT.
        let deadline = std::time::Instant::now() + CLI_TIMEOUT;
        while child.try_wait().map_err(|e| CallError::Transient(e.to_string()))?.is_none() {
            if std::time::Instant::now() > deadline {
                let _ = child.kill();
                return Err(CallError::Transient(format!("claude did not answer within {} s", CLI_TIMEOUT.as_secs())));
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        let out = child.wait_with_output().map_err(|e| CallError::Transient(e.to_string()))?;
        let body: Value = serde_json::from_slice(&out.stdout)
            .map_err(|_| CallError::Transient(format!("no JSON from claude: {}", String::from_utf8_lossy(&out.stderr).chars().take(200).collect::<String>())))?;
        let result = body["result"].as_str().unwrap_or("").to_owned();
        if body["is_error"] == json!(true) {
            let fatal = result.contains("authenticate") || result.contains("model");
            return Err(if fatal { CallError::Fatal(result) } else { CallError::Transient(result) });
        }
        if result.trim().is_empty() {
            return Err(CallError::Transient("EmptyResponse: empty body".into()));
        }
        Ok(result)
    }

    /// The cached answer, else a live one (then cached), else a pending entry and an error.
    fn ask(&mut self, kind: &str, system: &str, prompt: &str) -> Result<String, CallError> {
        let k = key(&[&self.model, system, prompt]);
        // A 64-bit key can collide: an entry answers only the exact prompt it was made for.
        let exact = |e: &&Value| e["model"] == json!(self.model) && e["system"] == json!(system) && e["prompt"] == json!(prompt);
        if let Some(r) = self.cache.get(&k).filter(exact).and_then(|e| e["response"].as_str()) {
            // Cache maintenance: LLM_CACHE_USED lists the keys a run replayed, so entries no
            // scenario asks for any more can be pruned.
            if let Some(path) = std::env::var_os("LLM_CACHE_USED")
                && let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path)
            {
                let _ = writeln!(f, "{k}");
            }
            return Ok(r.to_owned());
        }
        let live = match &self.cli_error {
            Some(e) => Err(CallError::Fatal(e.clone())),
            None => retry(kind, std::thread::sleep, || self.call_cli(system, prompt)),
        };
        let entry = json!({"kind": kind, "model": self.model, "system": system, "prompt": prompt});
        match live {
            Ok(r) => {
                let mut e = entry;
                e["response"] = json!(r);
                self.cache.insert(k, e);
                self.save(&self.cache_path, &self.cache);
                Ok(r)
            }
            Err(err) => {
                if matches!(err, CallError::Fatal(_)) {
                    self.cli_error = Some(err.to_string()); // do not retry a dead CLI every call
                }
                let pending_path = self.pending_path();
                let mut pending: Map<String, Value> =
                    std::fs::read_to_string(&pending_path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
                pending.insert(k, entry);
                self.save(&pending_path, &pending);
                Err(CallError::Fatal(format!("no cached answer and claude unavailable ({err}); prompt queued in {}", pending_path.display())))
            }
        }
    }
}

impl Llm for ClaudeCode {
    fn status(&self) -> String {
        format!("OK - claude-code:{} (cache {})", self.model, self.cache.len())
    }

    fn model(&self) -> String {
        self.model.clone()
    }

    fn converse(&mut self, pack: &str, user: &str, tool: &mut ToolFn<'_>, current_time: &str) -> Converse {
        let prompt = build_prompt(pack, user, current_time);
        let system = format!("{SYSTEM_PROMPT}{TOOL_PROTOCOL}");
        let raw = match self.ask("converse", &system, &prompt) {
            Ok(r) => r,
            Err(e) => return Converse { text: format!("[LLM error] {e}"), prompt },
        };
        let Some(Value::Object(obj)) = relaxed_json(&raw) else {
            return Converse { text: raw.trim().to_owned(), prompt }; // plain prose: an answer without tools
        };
        for s in obj.get("save_memory").and_then(Value::as_array).into_iter().flatten() {
            let args = if s.is_object() { s.clone() } else { json!({"text": s}) };
            tool("save_memory", &args);
        }
        for d in obj.get("delete_memory").and_then(Value::as_array).into_iter().flatten() {
            let id = d.as_str().map_or_else(|| d["id"].clone(), |s| json!(s));
            tool("delete_memory", &json!({"id": id}));
        }
        Converse { text: obj.get("reply").and_then(Value::as_str).unwrap_or("").trim().to_owned(), prompt }
    }

    fn extract(&mut self, user: &str, assistant: &str, current_time: &str, known: &str) -> Vec<String> {
        let known = if known.trim().is_empty() { "(なし)" } else { known.trim() };
        let time = if current_time.is_empty() { "(不明)" } else { current_time };
        let instruction = EXTRACT_INSTRUCTION
            .replace("{user_text}", user)
            .replace("{assistant_text}", assistant)
            .replace("{known}", known)
            .replace("{current_time}", time);
        self.ask("extract", EXTRACT_SYSTEM_PROMPT, &instruction).map(|r| parse_texts(&r)).unwrap_or_default()
    }

    fn adjudicate(&mut self, request: &DreamRequest) -> engram::Result<DreamDecision> {
        if request.members.is_empty() {
            return Ok(DreamDecision::keep());
        }
        let raw = self.ask("dream", DREAM_SYSTEM_PROMPT, &request.prompt())?;
        DreamDecision::parse(&raw).ok_or_else(|| "unparsable dream verdict".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_answers_are_replayed_and_misses_are_queued() {
        let dir = std::env::temp_dir().join(format!("engram-claude-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let s = Settings {
            llm_cache: dir.join("cache.json").to_string_lossy().into_owned(),
            claude_cli: "definitely-not-a-claude-binary".into(),
            ..Settings::default()
        };
        let mut c = ClaudeCode::new(&s);
        let req = DreamRequest { current_time: "t".into(), members: vec![engram::DreamMember { id: "A".into(), text: "x".into(), local_time: "t".into(), timezone: "UTC;+00:00".into(), retrievability: 1.0 }] };
        assert!(c.adjudicate(&req).is_err()); // no CLI, no cache: queued
        let pending: Map<String, Value> = serde_json::from_str(&std::fs::read_to_string(dir.join("cache.pending.json")).unwrap()).unwrap();
        let (k, mut e) = pending.into_iter().next().unwrap();
        assert_eq!(e["kind"], "dream");
        e["response"] = json!(r#"{"action":"keep"}"#);
        std::fs::write(dir.join("cache.json"), serde_json::to_string(&json!({k: e})).unwrap()).unwrap();
        let mut c = ClaudeCode::new(&s);
        assert_eq!(c.adjudicate(&req).unwrap(), DreamDecision::keep()); // replayed, no CLI needed
        let mut saved = Vec::new();
        let k2 = key(&[&c.model, &format!("{SYSTEM_PROMPT}{TOOL_PROTOCOL}"), &build_prompt("", "hi", "now")]);
        let system = format!("{SYSTEM_PROMPT}{TOOL_PROTOCOL}");
        c.cache.insert(k2.clone(), json!({"model": c.model, "system": system, "prompt": "another prompt", "response": "{}"}));
        assert!(c.converse("", "hi", &mut |_, _| json!({}), "now").text.starts_with("[LLM error]")); // a colliding key is a miss
        c.cache.insert(k2, json!({"model": c.model, "system": system, "prompt": build_prompt("", "hi", "now"),
                                  "response": r#"{"reply":"ok","save_memory":[{"text":"t","cue":"q"}],"delete_memory":["X"]}"#}));
        let r = c.converse("", "hi", &mut |n, a| {
            saved.push((n.to_owned(), a.clone()));
            json!({})
        }, "now");
        assert_eq!(r.text, "ok");
        assert_eq!(saved.len(), 2);
        assert_eq!(saved[1].1["id"], "X");
    }
}
