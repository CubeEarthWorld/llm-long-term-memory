//! `ltm` — the LLM Long-Term Memory app: web UI, headless runner and benchmarks.

use engram::app::{bench, cli, server};

const USAGE: &str = "usage:
  ltm [serve] [--port N] [--no-browser]   web app (default http://localhost:8501)
  ltm run [...]                            headless seed replay / turns / dream (ltm run --help)
  ltm bench <forget|sweep|supersede|generality|cache|perf> [...]";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("run") => cli::main(&args[1..]),
        Some("bench") => bench::main(&args[1..]),
        Some("-h" | "--help" | "help") => {
            println!("{USAGE}");
            0
        }
        _ => {
            let rest = if args.first().map(String::as_str) == Some("serve") { &args[1..] } else { &args[..] };
            let mut port: u16 = engram::app::settings::env("PORT", "8501").parse().unwrap_or(8501);
            let mut browser = true;
            let mut it = rest.iter();
            while let Some(a) = it.next() {
                match a.as_str() {
                    "--port" => port = it.next().and_then(|p| p.parse().ok()).unwrap_or(port),
                    "--no-browser" => browser = false,
                    other => {
                        eprintln!("unknown argument {other}\n{USAGE}");
                        std::process::exit(2);
                    }
                }
            }
            i32::from(server::serve(port, browser).is_err())
        }
    };
    std::process::exit(code);
}
