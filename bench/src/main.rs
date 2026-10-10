//! `ltm` — tools around the engram crate: a headless scenario runner and benchmarks.

use ltm::{bench, cli};

const USAGE: &str = "usage:
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
            eprintln!("{USAGE}");
            2
        }
    };
    std::process::exit(code);
}
