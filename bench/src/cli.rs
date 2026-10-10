//! Headless runner: build the engine, optionally replay the seed scenario on a virtual
//! timeline, run extra turns (`--say`), dream (`--dream`), inspect (`--inspect`), and
//! write per-turn results and final statistics to JSON.

use crate::seed;
use crate::session::Session;
use crate::settings::{data_dir, seed_csv_path, AppConfig};
use serde_json::{json, Value};
use std::path::PathBuf;

const USAGE: &str = "usage: ltm run [--no-reset] [--no-seed] [--say TEXT]... [--dream [N]] [--epoch UNIX] [--inspect] [--out PATH]\n  --epoch fixes the virtual clock and the id sequence, so a rerun sends the same prompts (and replays cached answers)";

struct Args {
    reset: bool,
    seed: bool,
    say: Vec<String>,
    dream: Option<usize>,
    inspect: bool,
    epoch: Option<f64>,
    out: PathBuf,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args { reset: true, seed: true, say: Vec::new(), dream: None, inspect: false, epoch: None, out: data_dir().join("results.json") };
    let mut it = args.iter().peekable();
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().cloned().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "--reset" => a.reset = true,
            "--no-reset" => a.reset = false,
            "--seed" => a.seed = true,
            "--no-seed" => a.seed = false,
            "--say" => a.say.push(value("--say")?),
            "--out" => a.out = PathBuf::from(value("--out")?),
            "--inspect" => a.inspect = true,
            "--epoch" => a.epoch = Some(value("--epoch")?.parse().map_err(|_| "--epoch needs Unix seconds")?),
            "--dream" => {
                let n = it.peek().and_then(|v| v.parse::<usize>().ok());
                if n.is_some() {
                    it.next();
                }
                a.dream = Some(n.unwrap_or(5));
            }
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    Ok(a)
}

fn rule(c: char) -> String {
    c.to_string().repeat(78)
}

fn print_turn(s: &Session, turn: u64, utterance: &str) {
    println!("\n{}\nTURN {turn}  {utterance}\n{}", rule('='), rule('='));
    let Some(m) = s.metrics.iter().find(|m| m.turn == turn) else {
        println!("No metrics recorded.");
        return;
    };
    println!("\n-- {} --", crate::TITLE);
    println!("  応答: {}", m.response.trim().chars().take(400).collect::<String>());
    println!("  書込: {}", m.write_note);
    println!("  records={}  pack={}字/{}件  time={:.0}ms (llm {:.0})", m.records(), m.pack_chars(), m.recalled.len(), m.total_ms(), m.llm_ms);
    if m.recalled.is_empty() {
        println!("  想起なし");
        return;
    }
    println!("  想起された記憶 ({}) / 引用 {}:", m.recalled.len(), m.cited.len());
    for r in &m.recalled {
        let extra: Vec<String> = r.as_object().into_iter().flatten().filter(|(k, _)| !matches!(k.as_str(), "id" | "text" | "score")).map(|(k, v)| format!("{k}={v}")).collect();
        let text: String = r["text"].as_str().unwrap_or("").chars().take(80).collect();
        println!("     - [{:.2}] {text}  ({})", r["score"].as_f64().unwrap_or(0.0), extra.join(" "));
    }
}

fn print_dream(results: &[Value]) {
    println!("\n{}\nDREAM  ({} クラスタ処理)\n{}", rule('='), results.len(), rule('='));
    if results.is_empty() {
        println!("  対象クラスタなし（不安定な記憶に近傍がない）");
        return;
    }
    let cut = |v: &Value| v.as_str().unwrap_or("").chars().take(70).collect::<String>();
    for r in results {
        let err = r["error"].as_str().map_or(String::new(), |e| format!("  error={e}"));
        println!("\n-- action={}{err}", r["action"].as_str().unwrap_or("?"));
        let before = r["before"].as_array().cloned().unwrap_or_default();
        println!("   統合元 ({}):", before.len());
        for b in &before {
            println!("     - [R={}] {}", b["R"], cut(&b["text"]));
        }
        let after = r["after"].as_array().cloned().unwrap_or_default();
        if after.is_empty() {
            println!("   統合後: （維持）");
        } else {
            println!("   統合後 ({}):", after.len());
            for a in &after {
                println!("     + [S={}s] {}", a["stability"], cut(&a["text"]));
            }
        }
    }
}

/// `ltm run …`. Returns the process exit code.
pub fn main(args: &[String]) -> i32 {
    let a = match parse(args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    let cfg = AppConfig::default();
    println!("[init] embedding={} -> ./model", cfg.glob.embedding_model);
    let mut s = match Session::open(cfg, a.reset) {
        Ok(s) => s,
        Err(e) => {
            println!("\n[FATAL] embedding model load failed: {e}");
            return 2;
        }
    };
    if let Some(t) = a.epoch {
        s.memory.set_ids(crate::ids::reproducible(t.to_bits()));
    }
    println!("[init] model={}", s.llm.model());
    println!("[init] embedding: {}", s.embedding_status);
    println!("[init] llm: {}", s.llm.status());
    let run = (|| -> engram::Result<()> {
        if a.seed {
            s.replay(&seed::load(&seed_csv_path()), a.epoch, |s, t, item| print_turn(s, t, &item.text))?;
        }
        for text in &a.say {
            let t = s.run_turn(text, "")?;
            print_turn(&s, t, text);
        }
        if let Some(budget) = a.dream.filter(|&b| b > 0) {
            print_dream(&s.dream(Some(budget))?);
        }
        if a.inspect {
            let dump = serde_json::to_string_pretty(&s.snapshot()?)?;
            println!("{}", dump.chars().take(8000).collect::<String>());
        }
        println!("\n{}\n# サマリ\n{}", rule('#'), rule('#'));
        if !s.metrics.is_empty() {
            for (label, ok) in s.invariants().as_object().into_iter().flatten() {
                println!("  {label} : {ok}");
            }
        }
        println!("  {}: {}", crate::TITLE, s.final_stats());
        let keys = ["response", "write_note", "records", "pack_chars", "times", "recalled"];
        let turns: Vec<Value> = s
            .log
            .iter()
            .map(|r| {
                let system: serde_json::Map<String, Value> = keys.iter().filter_map(|k| r["system"].get(*k).map(|v| ((*k).to_owned(), v.clone()))).collect();
                json!({"turn": r["turn"], "utterance": r["utterance"], "note": r["note"], "system": system})
            })
            .collect();
        if let Some(dir) = a.out.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&a.out, serde_json::to_string_pretty(&json!({"turns": turns, "final": s.final_stats()}))?)?;
        println!("\n[saved] {}", a.out.display());
        Ok(())
    })();
    match run {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("[error] {e}");
            1
        }
    }
}
