//! Claude through Claude Code (`claude -p`), with a prompt-keyed response cache.
//!
//! Every answer is stored under a hash of (model, system prompt, prompt), so a test or
//! benchmark that replays the same scenario asks the model once and then reuses the
//! same data. When the CLI cannot answer (not installed, not signed in) a missing
//! prompt is appended to `<cache>.pending.json` and the call fails — a dream leaves the
//! seed labile, a turn answers without saving — so the pending prompts can be answered
//! out of band (any Claude client) and merged into the cache.

use crate::llm::{build_prompt, fill, parse_texts, retry, CallError, Converse, Llm, ToolFn, EXTRACT_INSTRUCTION, EXTRACT_SYSTEM_PROMPT, SYSTEM_PROMPT};
use crate::settings::{root, Settings};
use engram::{relaxed_json, DreamDecision, DreamRequest, DREAM_SYSTEM_PROMPT};
use serde_json::{json, Map, Value};
use std::io::{Read, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

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
const CLI_TIMEOUT: Duration = Duration::from_secs(600);

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

/// Whether a CLI error fails the same way on every retry: not signed in, or a model the
/// CLI does not know. Everything else (overload, timeouts, network) is worth a retry.
fn is_fatal(msg: &str) -> bool {
    const AUTH: [&str; 7] = ["authenticat", "/login", "logged in", "oauth", "credential", "expired", "api key"];
    const MODEL: [&str; 3] = ["model not found", "invalid model", "unknown model"];
    let m = msg.to_lowercase();
    AUTH.iter().chain(&MODEL).any(|w| m.contains(w))
}

fn classify(msg: String) -> CallError {
    if is_fatal(&msg) { CallError::Fatal(msg) } else { CallError::Transient(msg) }
}

/// A scratch file removed when dropped.
struct TempFile(PathBuf);

impl TempFile {
    fn new(text: &str) -> std::io::Result<Self> {
        static N: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!("ltm-system-{}-{}.txt", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        std::fs::write(&path, text)?;
        Ok(Self(path))
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Starts `cli args…` (plus the system prompt) with all three pipes. On Windows an
/// extensionless `cli` that is not found is retried as npm's `<cli>.cmd` shim, which runs
/// through `cmd.exe`; a multi-line argument cannot pass there, so the system prompt goes
/// in a file (kept alive by the returned guard).
fn spawn(cli: &str, args: &[&str], system: Option<&str>) -> std::io::Result<(Child, Option<TempFile>)> {
    let run = |exe: &str, extra: &[&str]| Command::new(exe).args(args).args(extra).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn();
    let inline = system.map_or_else(Vec::new, |s| vec!["--system-prompt", s]);
    match run(cli, &inline) {
        Err(e) if cfg!(windows) && e.kind() == std::io::ErrorKind::NotFound && Path::new(cli).extension().is_none() => {
            let shim = format!("{cli}.cmd");
            let Some(system) = system else { return Ok((run(&shim, &[])?, None)) };
            let file = TempFile::new(system)?;
            let child = run(&shim, &["--system-prompt-file", &file.0.to_string_lossy()])?;
            Ok((child, Some(file)))
        }
        r => Ok((r?, None)),
    }
}

fn drain(mut r: impl Read + Send + 'static) -> JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut b = Vec::new();
        let _ = r.read_to_end(&mut b);
        b
    })
}

/// A JSON object file (missing = empty). One that exists but cannot be read is moved
/// aside to `<file>.corrupt-<unix>.json`, so the next save cannot overwrite its entries.
fn load_map(path: &Path) -> Result<Map<String, Value>, String> {
    let err = match std::fs::read(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Map::new()),
        Err(e) => e.to_string(),
        Ok(bytes) => match serde_json::from_slice(&bytes) {
            Ok(m) => return Ok(m),
            Err(e) => e.to_string(),
        },
    };
    let unix = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let aside = path.with_extension(format!("corrupt-{unix}.json"));
    std::fs::rename(path, &aside).map_err(|e| format!("{} is damaged ({err}) and could not be moved aside: {e}", path.display()))?;
    eprintln!("{} is damaged ({err}); moved to {}", path.display(), aside.display());
    Ok(Map::new())
}

/// What one CLI run left: its exit status, both outputs, and whether stdin took the
/// whole input.
struct Output {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    wrote: std::io::Result<()>,
}

/// Claude Haiku (or any Claude model) via the Claude Code CLI.
pub struct ClaudeCode {
    cli: String,
    model: String,
    cache_path: PathBuf,
    cache: Map<String, Value>,
    cli_error: Option<String>,
    timeout: Duration,
}

impl ClaudeCode {
    /// Fails only when a damaged cache cannot be moved out of the way.
    pub fn new(s: &Settings) -> engram::Result<Self> {
        let path = PathBuf::from(&s.llm_cache);
        let cache_path = if path.is_absolute() { path } else { root().join(path) };
        let cache = load_map(&cache_path)?;
        Ok(Self { cli: s.claude_cli.clone(), model: s.claude_model.clone(), cache_path, cache, cli_error: None, timeout: CLI_TIMEOUT })
    }

    fn pending_path(&self) -> PathBuf {
        self.cache_path.with_extension("pending.json")
    }

    /// Writes `map` atomically: a crash never leaves half a cache.
    fn save(&self, path: &Path, map: &Map<String, Value>) {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Err(e) = crate::write_atomic(path, serde_json::to_string_pretty(map).unwrap_or_default()) {
            eprintln!("could not write {}: {e}", path.display());
        }
    }

    /// Runs the CLI once with `input` on stdin. Every pipe is served by its own thread, so
    /// a CLI that writes more than a pipe buffer cannot stall; a hung one is killed and
    /// reaped after `timeout`.
    fn run(&self, args: &[&str], system: Option<&str>, input: &str) -> Result<Output, CallError> {
        let (mut child, _system_file) = spawn(&self.cli, args, system).map_err(|e| CallError::Fatal(format!("cannot run {}: {e}", self.cli)))?;
        let (stdout, stderr) = (drain(child.stdout.take().expect("piped")), drain(child.stderr.take().expect("piped")));
        let (mut stdin, input) = (child.stdin.take().expect("piped"), input.to_owned());
        let writer = std::thread::spawn(move || stdin.write_all(input.as_bytes())); // closes stdin when done
        let deadline = Instant::now() + self.timeout;
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(20)),
                waited => {
                    let _ = child.kill();
                    let _ = child.wait();
                    // The pipe threads finish on their own: a grandchild (the `.cmd` shim's
                    // node) may hold the pipes open after the kill.
                    return Err(CallError::Transient(match waited {
                        Err(e) => format!("waiting for claude: {e}"),
                        _ => format!("claude did not answer within {} s", self.timeout.as_secs()),
                    }));
                }
            }
        };
        let wrote = writer.join().unwrap_or(Ok(()));
        Ok(Output { status, stdout: stdout.join().unwrap_or_default(), stderr: stderr.join().unwrap_or_default(), wrote })
    }

    /// One headless Claude Code call without tools or session state.
    fn call_cli(&self, system: &str, prompt: &str) -> Result<String, CallError> {
        let args = ["-p", "--model", &self.model, "--output-format", "json", "--tools", "", "--no-session-persistence"];
        let out = self.run(&args, Some(system), prompt)?;
        let Ok(body) = serde_json::from_slice::<Value>(&out.stdout) else {
            // An early exit (e.g. not signed in) breaks the prompt pipe; stderr says why.
            let stderr = String::from_utf8_lossy(&out.stderr).trim().chars().take(300).collect::<String>();
            return Err(classify(match out.wrote {
                Err(e) => format!("claude exited before reading the prompt ({e}): {stderr}"),
                Ok(()) => format!("no JSON from claude ({}): {stderr}", out.status),
            }));
        };
        let result = body["result"].as_str().unwrap_or("").to_owned();
        if body["is_error"] == json!(true) {
            return Err(classify(result));
        }
        if result.trim().is_empty() {
            return Err(CallError::Transient("EmptyResponse: empty body".into()));
        }
        Ok(result)
    }

    /// The cached answer, else a live one (then cached), else a pending entry and an error.
    fn ask(&mut self, kind: &str, system: &str, prompt: &str) -> Result<String, CallError> {
        let k = key(&[&self.model, system, prompt]);
        // Cache maintenance: LLM_CACHE_USED lists the keys a run used (replayed or fetched
        // live), so entries no scenario asks for any more can be pruned.
        let mark_used = |k: &str| {
            if let Some(path) = std::env::var_os("LLM_CACHE_USED")
                && let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path)
            {
                let _ = writeln!(f, "{k}");
            }
        };
        // A 64-bit key can collide: an entry answers only the exact prompt it was made for.
        let exact = |e: &&Value| e["model"] == json!(self.model) && e["system"] == json!(system) && e["prompt"] == json!(prompt);
        if let Some(r) = self.cache.get(&k).filter(exact).and_then(|e| e["response"].as_str()) {
            mark_used(&k);
            return Ok(r.to_owned());
        }
        let live = match &self.cli_error {
            Some(e) => Err(CallError::Fatal(e.clone())),
            None => retry(kind, std::thread::sleep, || self.call_cli(system, prompt)),
        };
        let entry = json!({"kind": kind, "model": self.model, "system": system, "prompt": prompt});
        match live {
            Ok(r) => {
                mark_used(&k);
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
                let mut pending = load_map(&pending_path).map_err(|e| CallError::Fatal(format!("claude unavailable ({err}); prompt not queued: {e}")))?;
                pending.insert(k, entry);
                self.save(&pending_path, &pending);
                Err(CallError::Fatal(format!("no cached answer and claude unavailable ({err}); prompt queued in {}", pending_path.display())))
            }
        }
    }
}

impl Llm for ClaudeCode {
    /// Asks the CLI for its version (no model call): ERROR when it cannot run.
    fn status(&self) -> String {
        if let Some(e) = &self.cli_error {
            return format!("ERROR - {e}");
        }
        match self.run(&["--version"], None, "") {
            Ok(o) if o.status.success() => format!("OK - claude-code:{} ({}, cache {})", self.model, String::from_utf8_lossy(&o.stdout).trim(), self.cache.len()),
            Ok(o) => format!("ERROR - {} --version exited with {}: {}", self.cli, o.status, String::from_utf8_lossy(&o.stderr).trim()),
            Err(e) => format!("ERROR - {e}"),
        }
    }

    fn model(&self) -> String {
        self.model.clone()
    }

    fn converse(&mut self, pack: &str, user: &str, tool: &mut ToolFn<'_>, current_time: &str) -> Converse {
        let prompt = build_prompt(pack, user, current_time);
        let system = format!("{SYSTEM_PROMPT}{TOOL_PROTOCOL}");
        let raw = match self.ask("converse", &system, &prompt) {
            Ok(r) => r,
            Err(e) => return Converse { text: format!("[LLM error] {e}"), prompt, failed: true },
        };
        let Some(Value::Object(obj)) = relaxed_json(&raw) else {
            return Converse { text: raw.trim().to_owned(), prompt, failed: false }; // plain prose: an answer without tools
        };
        for s in obj.get("save_memory").and_then(Value::as_array).into_iter().flatten() {
            let args = if s.is_object() { s.clone() } else { json!({"text": s}) };
            tool("save_memory", &args);
        }
        for d in obj.get("delete_memory").and_then(Value::as_array).into_iter().flatten() {
            let id = d.as_str().map_or_else(|| d["id"].clone(), |s| json!(s));
            tool("delete_memory", &json!({"id": id}));
        }
        Converse { text: obj.get("reply").and_then(Value::as_str).unwrap_or("").trim().to_owned(), prompt, failed: false }
    }

    fn extract(&mut self, user: &str, assistant: &str, current_time: &str, known: &str) -> Vec<String> {
        let known = if known.trim().is_empty() { "(なし)" } else { known.trim() };
        let time = if current_time.is_empty() { "(不明)" } else { current_time };
        let instruction = fill(EXTRACT_INSTRUCTION, &[("user_text", user), ("assistant_text", assistant), ("known", known), ("current_time", time)]);
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

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("engram-claude-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn settings(dir: &Path, cli: &str) -> Settings {
        Settings { llm_cache: dir.join("cache.json").to_string_lossy().into_owned(), claude_cli: cli.into(), ..Settings::default() }
    }

    #[test]
    fn cached_answers_are_replayed_and_misses_are_queued() {
        let dir = temp("cache");
        let s = settings(&dir, "definitely-not-a-claude-binary");
        let mut c = ClaudeCode::new(&s).unwrap();
        assert!(c.status().starts_with("ERROR")); // no CLI: not OK, whatever the cache holds
        let req = DreamRequest { current_time: "t".into(), members: vec![engram::DreamMember { id: "A".into(), text: "x".into(), local_time: "t".into(), timezone: "UTC;+00:00".into(), retrievability: 1.0 }] };
        assert!(c.adjudicate(&req).is_err()); // no CLI, no cache: queued
        let pending: Map<String, Value> = serde_json::from_str(&std::fs::read_to_string(dir.join("cache.pending.json")).unwrap()).unwrap();
        let (k, mut e) = pending.into_iter().next().unwrap();
        assert_eq!(e["kind"], "dream");
        e["response"] = json!(r#"{"action":"keep"}"#);
        std::fs::write(dir.join("cache.json"), serde_json::to_string(&json!({k: e})).unwrap()).unwrap();
        let mut c = ClaudeCode::new(&s).unwrap();
        assert_eq!(c.adjudicate(&req).unwrap(), DreamDecision::keep()); // replayed, no CLI needed
        let mut saved = Vec::new();
        let k2 = key(&[&c.model, &format!("{SYSTEM_PROMPT}{TOOL_PROTOCOL}"), &build_prompt("", "hi", "now")]);
        let system = format!("{SYSTEM_PROMPT}{TOOL_PROTOCOL}");
        c.cache.insert(k2.clone(), json!({"model": c.model, "system": system, "prompt": "another prompt", "response": "{}"}));
        let miss = c.converse("", "hi", &mut |_, _| json!({}), "now");
        assert!(miss.failed && miss.text.starts_with("[LLM error]")); // a colliding key is a miss
        c.cache.insert(k2, json!({"model": c.model, "system": system, "prompt": build_prompt("", "hi", "now"),
                                  "response": r#"{"reply":"ok","save_memory":[{"text":"t","cue":"q"}],"delete_memory":["X"]}"#}));
        let r = c.converse("", "hi", &mut |n, a| {
            saved.push((n.to_owned(), a.clone()));
            json!({})
        }, "now");
        assert_eq!((r.text.as_str(), r.failed), ("ok", false));
        assert_eq!(saved.len(), 2);
        assert_eq!(saved[1].1["id"], "X");
    }

    #[test]
    fn a_damaged_cache_is_moved_aside_not_overwritten() {
        let dir = temp("corrupt");
        std::fs::write(dir.join("cache.json"), "{\"k\": {\"response\": \"tru").unwrap();
        std::fs::write(dir.join("cache.pending.json"), "not json").unwrap();
        let mut c = ClaudeCode::new(&settings(&dir, "definitely-not-a-claude-binary")).unwrap();
        assert!(c.cache.is_empty());
        assert!(c.adjudicate(&DreamRequest { current_time: "t".into(), members: vec![engram::DreamMember { id: "A".into(), text: "x".into(), local_time: "t".into(), timezone: "UTC;+00:00".into(), retrievability: 1.0 }] }).is_err());
        let names: Vec<String> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        let kept: Vec<String> = names.iter().filter(|n| n.contains(".corrupt-")).map(|n| std::fs::read_to_string(dir.join(n)).unwrap()).collect();
        assert_eq!(kept.len(), 2, "{names:?}");
        assert!(kept.contains(&"not json".to_owned()) && kept.iter().any(|t| t.contains("\"tru")));
    }

    #[test]
    fn only_lasting_failures_are_fatal() {
        for m in ["Invalid API key · Please run /login", "OAuth token has expired", "Not logged in", "Failed to authenticate", "model not found: claude-x"] {
            assert!(matches!(classify(m.into()), CallError::Fatal(_)), "{m}");
        }
        for m in ["Overloaded: the model is busy, try again", "API Error: 529", "request timed out", "model output was empty"] {
            assert!(matches!(classify(m.into()), CallError::Transient(_)), "{m}");
        }
    }

    /// A stand-in CLI: drains stdin, floods stderr past any pipe buffer, then answers (or,
    /// with `hang`, sleeps). Extensionless on Windows, so it is reached through the `.cmd`
    /// fallback.
    fn fake_cli(dir: &Path, hang: bool) -> String {
        let base = dir.join(if hang { "hang" } else { "fake" });
        #[cfg(windows)]
        {
            let body = if hang {
                "@echo off\r\nping -n 30 127.0.0.1 >nul\r\n".to_owned()
            } else {
                let line = "x".repeat(200);
                format!("@echo off\r\nif \"%1\"==\"--version\" (echo 9.9.9 fake& exit /b 0)\r\nmore >nul\r\nfor /l %%i in (1,1,1500) do @echo {line} 1>&2\r\necho {{\"result\":\"answer\",\"is_error\":false}}\r\n")
            };
            std::fs::write(base.with_extension("cmd"), body).unwrap();
        }
        #[cfg(not(windows))]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let body = if hang {
                "#!/bin/sh\nsleep 30\n".to_owned()
            } else {
                "#!/bin/sh\n[ \"$1\" = --version ] && { echo 9.9.9 fake; exit 0; }\ncat >/dev/null\nhead -c 300000 /dev/zero | tr '\\0' x >&2\necho '{\"result\":\"answer\",\"is_error\":false}'\n".to_owned()
            };
            std::fs::write(&base, body).unwrap();
            std::fs::set_permissions(&base, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        base.to_string_lossy().into_owned()
    }

    #[test]
    fn a_chatty_cli_does_not_stall_and_a_hung_one_is_killed() {
        let dir = temp("fake");
        let c = ClaudeCode::new(&settings(&dir, &fake_cli(&dir, false))).unwrap();
        assert!(c.status().starts_with("OK") && c.status().contains("9.9.9 fake"), "{}", c.status());
        let t = Instant::now();
        let prompt = "p".repeat(200_000); // more than a pipe buffer in, ~300 kB of stderr out
        assert_eq!(c.call_cli("line 1\nline 2", &prompt).unwrap(), "answer");
        assert!(t.elapsed() < Duration::from_secs(20));
        let mut hung = ClaudeCode::new(&settings(&dir, &fake_cli(&dir, true))).unwrap();
        hung.timeout = Duration::from_millis(500);
        let t = Instant::now();
        assert!(matches!(hung.call_cli("s", "p"), Err(CallError::Transient(m)) if m.contains("did not answer")));
        assert!(t.elapsed() < Duration::from_secs(10));
    }
}
