//! The local web app: a JSON API plus the no-build React frontend (compiled into the
//! binary), served over a deliberately small HTTP/1.1 loop on 127.0.0.1.
//!
//! One session lives behind one mutex. Every mutating job runs on a background thread
//! as *the* running job (a second one gets 409); `/api/state` never takes the session
//! lock, so the UI stays responsive while a model loads, a seed replays or a dream runs.

use super::seed::{self, SeedItem};
use super::session::Session;
use super::settings::{seed_csv_path, AppConfig};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

const MAX_BODY: usize = 16 << 20;

const ASSETS: [(&str, &str, &str); 6] = [
    ("/index.html", "text/html; charset=utf-8", include_str!("../../frontend/index.html")),
    ("/app.js", "text/javascript; charset=utf-8", include_str!("../../frontend/app.js")),
    ("/styles.css", "text/css; charset=utf-8", include_str!("../../frontend/styles.css")),
    ("/vendor/react.production.min.js", "text/javascript; charset=utf-8", include_str!("../../frontend/vendor/react.production.min.js")),
    ("/vendor/react-dom.production.min.js", "text/javascript; charset=utf-8", include_str!("../../frontend/vendor/react-dom.production.min.js")),
    ("/vendor/htm.umd.js", "text/javascript; charset=utf-8", include_str!("../../frontend/vendor/htm.umd.js")),
];

/// An HTTP answer.
pub struct Response {
    pub status: u16,
    pub content_type: &'static str,
    pub body: Vec<u8>,
}

impl Response {
    fn json(status: u16, v: &Value) -> Self {
        Self { status, content_type: "application/json", body: v.to_string().into_bytes() }
    }
    fn ok(v: Value) -> Self {
        Self::json(200, &v)
    }
    /// The error body the frontend reads (`{"detail": ...}`).
    fn error(status: u16, detail: &str) -> Self {
        Self::json(status, &json!({"detail": detail}))
    }
}

type Reply = Result<Response, Response>;

/// What `/api/state` reports without touching the session.
#[derive(Default)]
struct JobState {
    ready: bool,
    running: bool,
    progress: String,
    error: Option<String>,
    init_error: Option<String>,
    turn: u64,
    seeded: bool,
    embedding: String,
    llm: String,
}

/// The server's shared state.
pub struct App {
    state: Mutex<JobState>,
    session: Mutex<Option<Session>>,
    seed: Mutex<Vec<SeedItem>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner()) // a panicked job must not wedge the UI
}

impl App {
    pub fn new() -> Arc<Self> {
        Arc::new(Self { state: Mutex::default(), session: Mutex::new(None), seed: Mutex::new(seed::load(&seed_csv_path())) })
    }

    /// Mirrors the session's counters into the lock-free state.
    fn sync(&self, s: &Session) {
        let mut st = lock(&self.state);
        st.turn = s.turn;
        st.seeded = s.seeded;
        st.embedding = s.embedding_status.clone();
        st.llm = s.llm.status();
    }

    fn require_idle(&self) -> Result<(), Response> {
        if lock(&self.state).running { Err(Response::error(409, "A job is already running.")) } else { Ok(()) }
    }

    /// The ready session, or 503.
    fn session(&self) -> Result<MutexGuard<'_, Option<Session>>, Response> {
        if !lock(&self.state).ready {
            return Err(Response::error(503, "Engine is still starting."));
        }
        let guard = lock(&self.session);
        if guard.is_none() {
            return Err(Response::error(503, "Engine is still starting."));
        }
        Ok(guard)
    }

    /// Builds or rebuilds the session (closing the old one). The startup call (`cfg`
    /// = `None`) is a no-op once a session exists: a reset is admitted while not ready
    /// and must not be undone by a slow startup finishing second.
    pub fn init_session(&self, cfg: Option<AppConfig>, wipe: bool) {
        let mut guard = lock(&self.session);
        if cfg.is_none() && guard.is_some() {
            return;
        }
        lock(&self.state).ready = false;
        *guard = None; // close the old store before a wipe deletes its files
        match Session::open(cfg.unwrap_or_default(), wipe) {
            Ok(s) => {
                self.sync(&s);
                *guard = Some(s);
                let mut st = lock(&self.state);
                st.init_error = None;
                st.ready = true;
            }
            Err(e) => {
                eprintln!("session init failed: {e}");
                lock(&self.state).init_error = Some(e.to_string());
            }
        }
        lock(&self.state).progress.clear();
    }

    /// Runs `job` on a thread as *the* running job (409 if one is running). `running`
    /// is claimed under the state lock before the thread starts, so two requests cannot
    /// both pass; the previous job's progress line is cleared there.
    fn run_job(self: &Arc<Self>, job: impl FnOnce(&App) -> crate::Result<()> + Send + 'static) -> Result<(), Response> {
        {
            let mut st = lock(&self.state);
            if st.running {
                return Err(Response::error(409, "A job is already running."));
            }
            st.running = true;
            st.progress.clear();
            st.error = None;
        }
        let app = Arc::clone(self);
        std::thread::spawn(move || {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job(&app)));
            let mut st = lock(&app.state);
            match outcome {
                Ok(Ok(())) => {}
                Ok(Err(e)) => st.error = Some(e.to_string()),
                Err(p) => st.error = Some(format!("panic: {}", p.downcast_ref::<&str>().copied().or(p.downcast_ref::<String>().map(String::as_str)).unwrap_or("?"))),
            }
            st.running = false;
        });
        Ok(())
    }

    /// Runs `f` on the session and mirrors its counters afterwards.
    fn with_session<T>(&self, f: impl FnOnce(&mut Session) -> crate::Result<T>) -> crate::Result<T> {
        let mut guard = lock(&self.session);
        let s = guard.as_mut().ok_or("no session")?;
        let out = f(s);
        self.sync(s);
        out
    }

    fn seed_job(self: &Arc<Self>, reset: bool) -> Result<(), Response> {
        drop(self.session()?);
        let items = lock(&self.seed).clone();
        self.run_job(move |app| {
            let replay = app.with_session(|s| {
                if reset {
                    s.reset()?;
                }
                Ok(s.start_replay(None))
            })?;
            // The session lock is taken per turn, so readers can poll between turns.
            let result = items.iter().try_for_each(|item| {
                let turn = app.with_session(|s| s.replay_step(&replay, item))?;
                lock(&app.state).progress = format!("turn {turn}/{}: {}...", items.len(), item.text.chars().take(16).collect::<String>());
                std::thread::sleep(Duration::from_millis(20));
                Ok::<_, crate::Error>(())
            });
            app.with_session(|s| {
                s.finish_replay(true);
                Ok(())
            })?;
            result
        })
    }

    /// Routes one request.
    pub fn handle(self: &Arc<Self>, method: &str, path: &str, body: &[u8]) -> Response {
        let parsed = if body.iter().all(u8::is_ascii_whitespace) { Ok(json!({})) } else { serde_json::from_slice::<Value>(body) };
        let Ok(body) = parsed else { return Response::error(422, "request body is not JSON") };
        match self.route(method, path, &body) {
            Ok(r) | Err(r) => r,
        }
    }

    fn route(self: &Arc<Self>, method: &str, path: &str, body: &Value) -> Reply {
        match (method, path) {
            ("GET", "/api/state") => {
                let st = lock(&self.state);
                Ok(Response::ok(json!({
                    "ready": st.ready, "running": st.running, "progress": st.progress, "error": st.error,
                    "init_error": st.init_error, "turn": st.turn, "seeded": st.seeded, "embedding": st.embedding, "llm": st.llm,
                })))
            }
            ("GET", "/api/config") => Ok(Response::ok(lock(&self.session).as_ref().map_or_else(|| AppConfig::default().to_json(), |s| s.cfg.to_json()))),
            ("POST", "/api/reset") => {
                // Allowed while not ready, so a bad configuration (init_error) can be corrected.
                let cfg = AppConfig::from_json(&body["config"]).map_err(|e| Response::error(400, &e))?;
                self.run_job(move |app| {
                    app.init_session(Some(cfg), true);
                    Ok(())
                })?;
                Ok(Response::ok(json!({"ok": true})))
            }
            ("POST", "/api/reset-db") => {
                let mut guard = self.session()?; // idle check under the lock: a seed replay admits between turns
                self.require_idle()?;
                let s = guard.as_mut().expect("ready");
                s.reset().map_err(|e| Response::error(500, &e.to_string()))?;
                self.sync(s);
                Ok(Response::ok(json!({"ok": true})))
            }
            ("POST", "/api/seed") => {
                self.seed_job(body["reset"].as_bool().unwrap_or(true))?;
                Ok(Response::ok(json!({"ok": true})))
            }
            ("POST", "/api/turn") => {
                let text = body["text"].as_str().ok_or_else(|| Response::error(422, "`text` is required"))?.trim().to_owned();
                if text.is_empty() {
                    return Err(Response::error(400, "空の発話です。"));
                }
                drop(self.session()?);
                self.run_job(move |app| app.with_session(|s| s.run_turn(&text, "").map(drop)))?;
                Ok(Response::ok(json!({"ok": true})))
            }
            ("POST", "/api/dream") => {
                let budget = body["budget"].as_i64().map(|b| b.max(1) as usize);
                self.require_idle()?; // the read below takes the lock a running job may hold for minutes
                let n = self.session()?.as_ref().expect("ready").memory.stats().labile;
                if n == 0 {
                    return Ok(Response::ok(json!({"ok": true, "n": 0,
                        "message": "Dream 対象のクラスタがありません（不安定な記憶に近傍がない＝整理済み）"})));
                }
                self.run_job(move |app| {
                    let (done, replaced) = app.with_session(|s| {
                        let r = s.dream(budget)?;
                        Ok((r.len(), r.iter().filter(|x| x["action"] == "replace").count()))
                    })?;
                    lock(&app.state).progress = format!("dreamed {done} cluster(s), {replaced} consolidated");
                    Ok(())
                })?;
                Ok(Response::ok(json!({"ok": true, "n": n})))
            }
            ("GET", "/api/dream-log") => Ok(Response::ok(json!({"results": lock(&self.session).as_ref().map_or(vec![], |s| s.last_dream.clone())}))),
            ("GET", "/api/turns-detail") => {
                let guard = lock(&self.session);
                Ok(Response::ok(json!({"start_time": guard.as_ref().and_then(Session::start_time), "turns": guard.as_ref().map_or(vec![], |s| s.log.clone())})))
            }
            ("GET", "/api/db") => Ok(Response::ok(match self.session() {
                Ok(g) => {
                    let s = g.as_ref().expect("ready");
                    json!({"stats": s.final_stats(), "tables": {"memory": s.snapshot()}})
                }
                Err(_) => json!({"stats": {}, "tables": {}}),
            })),
            ("GET", "/api/clusters") => Ok(Response::ok(match self.session() {
                Ok(mut g) => {
                    let c = g.as_mut().expect("ready").cluster_preview();
                    json!({"clusters": c, "total_clusters": c.len()})
                }
                Err(_) => json!({"clusters": []}),
            })),
            ("GET", "/api/metrics") => Ok(Response::ok(match lock(&self.session).as_ref() {
                Some(s) => json!({"invariants": s.invariants(), "rows": s.metric_rows()}),
                None => json!({"rows": [], "invariants": {}}),
            })),
            ("GET", "/api/seed-utterances") => {
                let items: Vec<Value> = lock(&self.seed).iter().enumerate().map(|(i, s)| {
                    let mut v = s.to_json();
                    v["i"] = json!(i + 1);
                    v
                }).collect();
                Ok(Response::ok(json!({"utterances": items})))
            }
            ("POST", "/api/seed-utterances") => {
                let items = seed::from_json(body["items"].as_array().map_or(&[][..], Vec::as_slice));
                if items.is_empty() {
                    return Err(Response::error(400, "少なくとも1件の発話が必要です。"));
                }
                self.set_seed(items)
            }
            ("POST", "/api/seed-utterances/reset") => self.set_seed(seed::default_seed()),
            ("POST", "/api/seed-utterances/import") => {
                let items = seed::parse_csv(body["csv"].as_str().unwrap_or(""));
                if items.is_empty() {
                    return Err(Response::error(400, "CSVから有効な発話を読み取れませんでした（text列が必要です）。"));
                }
                Ok(Response::ok(json!({"n": items.len(), "items": items.iter().map(SeedItem::to_json).collect::<Vec<_>>()})))
            }
            ("GET", p) => {
                let p = if p == "/" { "/index.html" } else { p };
                ASSETS.iter().find(|a| a.0 == p).map(|&(_, ct, body)| Response { status: 200, content_type: ct, body: body.as_bytes().to_vec() }).ok_or_else(|| Response::error(404, "Not Found"))
            }
            _ => Err(Response::error(405, "Method Not Allowed")),
        }
    }

    /// Replaces the editable seed list and persists it; the lock keeps two saves from
    /// interleaving into a mixed file.
    fn set_seed(&self, items: Vec<SeedItem>) -> Reply {
        let mut seed = lock(&self.seed);
        seed::save(&seed_csv_path(), &items).map_err(|e| Response::error(500, &e.to_string()))?;
        let n = items.len();
        *seed = items;
        Ok(Response::ok(json!({"ok": true, "n": n})))
    }
}

/// One parsed request.
pub struct Request {
    pub method: String,
    pub path: String,
    pub host: String,
    pub origin: Option<String>,
    pub content_type: String,
    pub body: Vec<u8>,
}

/// The browser-facing guard. Any web page can make the user's browser send requests to
/// 127.0.0.1, so a request is served only when its `Host` is this server (defeats DNS
/// rebinding), any `Origin` is this server (defeats cross-site requests), and a POST
/// carries a JSON body (a cross-site JSON POST needs a CORS preflight, never answered).
pub fn check_request(r: &Request, port: u16) -> Result<(), Response> {
    let ours = |authority: &str| authority == format!("localhost:{port}") || authority == format!("127.0.0.1:{port}");
    if !ours(&r.host) {
        return Err(Response::error(403, "unexpected Host"));
    }
    if r.origin.as_deref().is_some_and(|o| !o.strip_prefix("http://").is_some_and(ours)) {
        return Err(Response::error(403, "cross-origin request refused"));
    }
    if r.method == "POST" && !r.content_type.to_ascii_lowercase().starts_with("application/json") {
        return Err(Response::error(415, "POST bodies must be application/json"));
    }
    Ok(())
}

/// Reads one request.
fn read_request(stream: &TcpStream) -> std::io::Result<Request> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let (method, target) = (parts.next().unwrap_or("").to_owned(), parts.next().unwrap_or("/"));
    let path = target.split('?').next().unwrap_or("/").to_owned();
    let (mut length, mut host, mut origin, mut content_type) = (0usize, String::new(), None, String::new());
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 || header.trim().is_empty() {
            break;
        }
        let Some((k, v)) = header.split_once(':') else { continue };
        let (k, v) = (k.trim().to_ascii_lowercase(), v.trim().to_owned());
        match k.as_str() {
            "content-length" => length = v.parse().unwrap_or(0),
            "host" => host = v,
            "origin" => origin = Some(v),
            "content-type" => content_type = v,
            _ => {}
        }
    }
    if length > MAX_BODY {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "body too large"));
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    Ok(Request { method, path, host, origin, content_type, body })
}

fn respond(mut stream: &TcpStream, r: &Response) -> std::io::Result<()> {
    let reason = match r.status {
        200 => "OK",
        400 => "Bad Request",
        403 => "Forbidden",
        415 => "Unsupported Media Type",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        422 => "Unprocessable Entity",
        503 => "Service Unavailable",
        _ => "Error",
    };
    // No caching: the frontend is edited in place during development.
    let head = format!(
        "HTTP/1.1 {} {reason}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store, no-cache, must-revalidate, max-age=0\r\nConnection: close\r\n\r\n",
        r.status,
        r.content_type,
        r.body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(&r.body)?;
    stream.flush()
}

fn serve_connection(app: &Arc<App>, stream: TcpStream, port: u16) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let response = match read_request(&stream) {
        Ok(r) => match check_request(&r, port) {
            Ok(()) => app.handle(&r.method, &r.path, &r.body),
            Err(refused) => refused,
        },
        Err(e) => Response::error(400, &e.to_string()),
    };
    let _ = respond(&stream, &response);
}

/// Serves the app on `127.0.0.1:port` until the process ends; the session is built on a
/// background thread so the UI can show "loading" while the model loads.
pub fn serve(port: u16, open_browser: bool) -> std::io::Result<()> {
    let listener = TcpListener::bind(("127.0.0.1", port)).map_err(|e| {
        eprintln!(
            "\n[error] could not start the web server on http://localhost:{port}\n        {e}\n        \
             The port is likely held by a previous instance: close it, or run start again (it frees a stale port).\n"
        );
        e
    })?;
    let app = App::new();
    let init = Arc::clone(&app);
    std::thread::spawn(move || init.init_session(None, false));
    let url = format!("http://localhost:{port}");
    println!("[run] {} at {url}  (Ctrl+C to stop)", super::TITLE);
    if open_browser {
        let url = url.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(1500));
            let _ = if cfg!(windows) {
                std::process::Command::new("cmd").args(["/C", "start", "", &url]).spawn()
            } else if cfg!(target_os = "macos") {
                std::process::Command::new("open").arg(&url).spawn()
            } else {
                std::process::Command::new("xdg-open").arg(&url).spawn()
            };
        });
    }
    for stream in listener.incoming().flatten() {
        let app = Arc::clone(&app);
        std::thread::spawn(move || serve_connection(&app, stream, port));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{check_request, Request};

    fn req(method: &str, host: &str, origin: Option<&str>, content_type: &str) -> Request {
        Request { method: method.into(), path: "/api/reset".into(), host: host.into(), origin: origin.map(Into::into), content_type: content_type.into(), body: Vec::new() }
    }

    #[test]
    fn only_this_origin_reaches_the_api() {
        let ok = |r: &Request| check_request(r, 8501).is_ok();
        assert!(ok(&req("GET", "localhost:8501", None, "")));
        assert!(ok(&req("POST", "127.0.0.1:8501", Some("http://127.0.0.1:8501"), "application/json")));
        assert!(!ok(&req("POST", "localhost:8501", Some("https://evil.example"), "application/json"))); // cross-site
        assert!(!ok(&req("POST", "localhost:8501", None, "text/plain"))); // a "simple" request needs no preflight
        assert!(!ok(&req("GET", "evil.example:8501", None, ""))); // DNS rebinding
        assert!(!ok(&req("POST", "localhost:8501", Some("http://localhost:8501.evil.example"), "application/json")));
    }
}
