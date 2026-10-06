mod context;
mod llm;
mod parse;
mod profile;
mod tools;

use context::Context;
use profile::{EditFormat, Profile};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

const HELP: &str = "tern: minimal coding agent

usage:
  tern                      interactive session
  tern -p \"task\"            run one request and exit (exit 0 done, 2 step limit, 1 error)
options:
  --profile NAME            use a named profile instead of matching on model name
  --stats-file PATH         write a JSON line with token usage and outcome when done
  --list-profiles           show available profiles and exit
env: TERN_BASE_URL TERN_MODEL TERN_API_KEY TERN_PROFILE TERN_CTX TERN_MAX_TOKENS
     TERN_MAX_STEPS TERN_CHECK TERN_YOLO TERN_CONFIG";

/// Static per profile: no timestamps or directory listings that change
/// between requests, so system prompt + tool schemas form a cacheable prefix.
fn system_prompt(p: &Profile) -> String {
    let find = if p.tool_enabled("glob") { "grep/glob" } else { "grep" };
    let change = if p.edit_format == EditFormat::Whole {
        "To change a file, read it, then write its complete new content."
    } else {
        "Change files with edit, not by rewriting them."
    };
    let mut s = format!(
        "You are a coding agent in the user's repository. \
Find code with {find} before reading; read only the ranges you need. {change} \
Don't repeat file contents back in replies. \
Be terse. When finished, reply with a one or two sentence summary."
    );
    if !p.prompt_extra.is_empty() {
        s.push('\n');
        s.push_str(&p.prompt_extra);
    }
    s
}

#[derive(Debug, PartialEq)]
enum Outcome {
    Done,
    StepLimit,
    Error,
}

struct Args {
    prompt: Option<String>,
    profile: Option<String>,
    stats_file: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args { prompt: None, profile: None, stats_file: None };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = |name: &str| it.next().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "-p" | "--prompt" => a.prompt = Some(val("-p")?),
            "--profile" => a.profile = Some(val("--profile")?),
            "--stats-file" => a.stats_file = Some(val("--stats-file")?),
            "--list-profiles" => {
                for p in profile::all()? {
                    let m = if p.matches.is_empty() { "(select by name)".into() } else { p.matches.join(", ") };
                    println!("{:<14} {:<7} ctx {:<7} {}", p.name, format!("{:?}", p.edit_format).to_lowercase(), p.ctx, m);
                }
                std::process::exit(0);
            }
            "-h" | "--help" => {
                println!("{HELP}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}; see --help")),
        }
    }
    Ok(a)
}

fn main() {
    let (args, cfg) = match parse_args().and_then(|a| llm::Config::new(a.profile.clone()).map(|c| (a, c))) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let p = &cfg.profile;
    let mut ctx = Context::new(&system_prompt(p), p.ctx);
    let mut tools = tools::Tools::new(cfg.yolo, p);
    let schemas = tools::schemas(p);

    eprintln!(
        "tern · {} @ {} · profile {} · {:?} edits · budget {} tokens",
        cfg.model, cfg.base_url, p.name, p.edit_format, p.ctx
    );

    if let Some(task) = &args.prompt {
        ctx.push(json!({"role": "user", "content": task}));
        let (outcome, steps) = run_turn(&cfg, &mut ctx, &mut tools, &schemas);
        if let Some(path) = &args.stats_file {
            let (requests, t) = ctx.totals();
            let line = json!({
                "profile": p.name, "model": cfg.model, "outcome": format!("{outcome:?}").to_lowercase(),
                "steps": steps, "requests": requests, "prompt_tokens": t.prompt,
                "cached_tokens": t.cached, "completion_tokens": t.completion,
            });
            if let Err(e) = std::fs::write(path, format!("{line}\n")) {
                eprintln!("error writing stats: {e}");
            }
        }
        std::process::exit(match outcome {
            Outcome::Done => 0,
            Outcome::StepLimit => 2,
            Outcome::Error => 1,
        });
    }

    eprintln!("commands: /stats /exit");
    loop {
        print!("\n> ");
        let _ = io::stdout().flush();
        let mut line = String::new();
        if io::stdin().lock().read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        match line.trim() {
            "" => continue,
            "/exit" => break,
            "/stats" => ctx.print_stats(),
            input => {
                ctx.push(json!({"role": "user", "content": input}));
                run_turn(&cfg, &mut ctx, &mut tools, &schemas);
            }
        }
    }
}

fn run_turn(cfg: &llm::Config, ctx: &mut Context, tools: &mut tools::Tools, schemas: &Value) -> (Outcome, usize) {
    let lenient = cfg.profile.lenient_parsing;
    for step in 1..=cfg.max_steps {
        ctx.maintain(cfg, tools);

        let sent = ctx.raw_estimate();
        let resp = match llm::chat(cfg, &ctx.messages(), Some(schemas)) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("error: {e}");
                return (Outcome::Error, step);
            }
        };
        let u = resp.usage;
        eprintln!("  [{} in, {} cached, {} out]", u.prompt, u.cached, u.completion);
        ctx.record_usage(u, sent);

        let mut msg = resp.message;
        if lenient {
            let n = parse::recover_text_calls(&mut msg, tools::ALL_TOOLS);
            if n > 0 {
                eprintln!("  [recovered {n} tool call(s) written as text]");
            }
        }
        if let Some(text) = msg["content"].as_str().filter(|t| !t.trim().is_empty()) {
            println!("{}", text.trim());
        }
        let calls = msg["tool_calls"].as_array().cloned().unwrap_or_default();
        ctx.push(msg);
        if calls.is_empty() {
            return (Outcome::Done, step);
        }

        let mut changed = false;
        for c in calls {
            let id = c["id"].as_str().unwrap_or("");
            let name = c["function"]["name"].as_str().unwrap_or("");
            let raw = c["function"]["arguments"].as_str().unwrap_or("{}");
            let (label, out) = match parse::parse_args(raw, lenient) {
                Ok(args) => {
                    let label = format!("{name} {}", tools::summarize_args(&args));
                    eprintln!("  · {label}");
                    (label, tools.run(name, &args))
                }
                // Tell the model what went wrong instead of guessing; most
                // models fix malformed JSON on the next try.
                Err(e) => (name.to_string(), format!("error: arguments were not valid JSON ({e})")),
            };
            changed |= matches!(name, "edit" | "write") && out.starts_with("ok");
            ctx.push_tool_result(id, label, out);
        }

        // One check per step, after all of the step's edits, attached to the
        // last result so the model sees it before deciding what's next.
        if changed {
            if let Some(report) = tools.run_check() {
                eprintln!("  {}", report.lines().next().unwrap_or(""));
                ctx.append_to_last_tool_result(&report);
            }
        }
    }
    println!("[stopped after {} steps]", cfg.max_steps);
    (Outcome::StepLimit, cfg.max_steps)
}
