mod context;
mod llm;
mod lsp;
mod parse;
mod profile;
mod sandbox;
mod tools;

use context::Context;
use profile::{EditFormat, Profile};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

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
    // A profile with no edit/write/bash can only investigate, so it gets a
    // planning prompt instead of the (wrong, token-wasting) "change files" one.
    let read_only = !(p.tool_enabled("edit") || p.tool_enabled("write") || p.tool_enabled("bash"));
    if read_only {
        let mut s = format!(
            "You are a planning agent in the user's repository. Find code with {find} and read \
only the ranges you need; you cannot modify files. Produce a concise, step-by-step \
implementation plan naming the files and functions to change. Don't repeat file contents \
back. When finished, reply with the plan as your summary."
        );
        if !p.prompt_extra.is_empty() {
            s.push('\n');
            s.push_str(&p.prompt_extra);
        }
        // Returns early: a read-only profile can't delegate (no subagents in
        // practice), so it skips the edit-oriented and `task` delegation prose.
        return s;
    }
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
    if !p.subagents.is_empty() {
        s.push_str(
            " Delegate self-contained subtasks with the task tool; you pay only for the \
result it returns, not the subagent's steps. Give complete, standalone instructions.",
        );
    }
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
        let (outcome, steps, _) = run_turn(&cfg, &mut ctx, &mut tools, &schemas, 0);
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
                run_turn(&cfg, &mut ctx, &mut tools, &schemas, 0);
            }
        }
    }
}

/// How deeply subagents may nest. Child profiles default to no `subagents`,
/// so this is a backstop against a misconfigured recursive delegation.
const MAX_SUBAGENT_DEPTH: usize = 2;

/// How many times `require_check_pass` will force the model to keep working
/// past a "done" with a failing check. A check that can't be made to pass
/// must not burn the whole step budget resending the context every retry.
const MAX_CHECK_RETRIES: usize = 3;

/// One tool call, parsed once so it can be logged, executed (possibly on
/// another thread), and paired back to its id in order.
struct Call {
    id: String,
    name: String,
    label: String,
    args: Result<Value, String>,
}

impl Call {
    fn parse(c: &Value, lenient: bool) -> Call {
        let id = c["id"].as_str().unwrap_or("").to_string();
        let name = c["function"]["name"].as_str().unwrap_or("").to_string();
        let raw = c["function"]["arguments"].as_str().unwrap_or("{}");
        let args = parse::parse_args(raw, lenient);
        let label = match &args {
            Ok(a) => format!("{name} {}", tools::summarize_args(a)),
            Err(_) => name.clone(),
        };
        Call { id, name, label, args }
    }

    /// A subagent delegation with valid arguments (the only thing we parallelize).
    fn is_subagent(&self) -> bool {
        self.name == "task" && self.args.is_ok()
    }
}

/// Execute one call on the current thread. `task` is handled here, not in
/// Tools, because spawning a subagent needs Config, which Tools doesn't hold.
fn run_call(cfg: &llm::Config, tools: &mut tools::Tools, c: &Call, depth: usize, ind: &str) -> String {
    match &c.args {
        Ok(args) => {
            eprintln!("{ind}  · {}", c.label);
            if c.name == "task" {
                run_subagent(cfg, args, depth, None)
            } else {
                tools.run(&c.name, args)
            }
        }
        // Tell the model what went wrong instead of guessing; most models fix
        // malformed JSON on the next try.
        Err(e) => format!("error: arguments were not valid JSON ({e})"),
    }
}

/// Runs the agent loop to completion. `depth` is 0 for the top-level agent and
/// increments for each nested subagent; it controls log indentation and
/// whether assistant text prints to stdout (only the top level does). Returns
/// the outcome, step count, and the final assistant text (the summary a parent
/// receives when this run was a subagent).
fn run_turn(
    cfg: &llm::Config,
    ctx: &mut Context,
    tools: &mut tools::Tools,
    schemas: &Value,
    depth: usize,
) -> (Outcome, usize, String) {
    let lenient = cfg.profile.lenient_parsing;
    let ind = "  ".repeat(depth);
    // Tool names the lenient parser will recognise in text form: the always-on
    // set plus any the profile opts into.
    let mut known: Vec<&str> = tools::ALL_TOOLS.to_vec();
    if !cfg.profile.subagents.is_empty() {
        known.push("task");
    }
    if !cfg.profile.lsp.is_empty() {
        known.push("def");
        known.push("refs");
    }
    let mut last_text = String::new();
    // Tracked across the whole turn for `require_check_pass`: whether the agent
    // actually changed anything (a red check it never touched isn't its to fix),
    // and how many times we've already forced it past a failing check.
    let mut made_edits = false;
    let mut check_retries = 0usize;

    for step in 1..=cfg.max_steps {
        ctx.maintain(cfg, tools);

        let sent = ctx.raw_estimate();
        let resp = match llm::chat(cfg, &ctx.messages(), Some(schemas)) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{ind}error: {e}");
                return (Outcome::Error, step, last_text);
            }
        };
        let u = resp.usage;
        eprintln!("{ind}  [{} in, {} cached, {} out]", u.prompt, u.cached, u.completion);
        ctx.record_usage(u, sent);

        let mut msg = resp.message;
        if lenient {
            let n = parse::recover_text_calls(&mut msg, &known);
            if n > 0 {
                eprintln!("{ind}  [recovered {n} tool call(s) written as text]");
            }
        }
        if let Some(text) = msg["content"].as_str().filter(|t| !t.trim().is_empty()) {
            let text = text.trim().to_string();
            if depth == 0 {
                println!("{text}");
            }
            last_text = text;
        }
        let calls = msg["tool_calls"].as_array().cloned().unwrap_or_default();
        ctx.push(msg);
        if calls.is_empty() {
            // Don't let the model declare victory while the check is failing —
            // but only if it actually changed something this turn, and only up
            // to MAX_CHECK_RETRIES so an unfixable check can't loop to the step
            // limit resending the whole context each time.
            if cfg.profile.require_check_pass && made_edits && check_retries < MAX_CHECK_RETRIES {
                if let Some((false, report)) = tools.run_check() {
                    check_retries += 1;
                    eprintln!("{ind}  [not done — check still failing, continuing ({check_retries}/{MAX_CHECK_RETRIES})]");
                    ctx.push(json!({"role": "user", "content":
                        format!("Not finished: the check is still failing. Keep working; do not stop until it passes.\n{report}")}));
                    continue;
                }
            }
            return (Outcome::Done, step, last_text);
        }

        // Parse every call once up front, then execute. Results are collected
        // by call index so they can be pushed back in order (each tool_call_id
        // must pair with its result) regardless of execution order.
        let parsed: Vec<Call> = calls.iter().map(|c| Call::parse(c, lenient)).collect();
        let mut outs: Vec<Option<String>> = vec![None; parsed.len()];

        let subagent_calls = parsed.iter().filter(|c| c.is_subagent()).count();
        if cfg.profile.parallel_subagents && subagent_calls >= 2 {
            // Independent subagents run concurrently (scope lets the threads
            // borrow cfg/args without 'static); each builds its own `tools`, so
            // there's no shared mutable state between them. They run in batches
            // of `max_parallel` so a step with many `task` calls doesn't open
            // all the threads/connections at once; 0 means no clamp.
            // Each parallel subagent edits a private sandbox copy of the tree,
            // not the shared one, so their writes can't clobber each other; its
            // changes are merged back after it joins (files two touched are
            // withheld — see sandbox.rs).
            let idx: Vec<usize> = parsed.iter().enumerate().filter(|(_, c)| c.is_subagent()).map(|(i, _)| i).collect();
            let batch = if cfg.profile.max_parallel == 0 { idx.len() } else { cfg.profile.max_parallel };
            for chunk in idx.chunks(batch) {
                // Snapshot per chunk: a later chunk is seeded from (and diffed
                // against) the tree the earlier chunks already merged into.
                let base = sandbox::snapshot(Path::new("."));
                let mut boxes: Vec<(usize, Option<sandbox::Sandbox>)> = Vec::new();
                for &i in chunk {
                    match sandbox::Sandbox::create(Path::new(".")) {
                        Ok(sb) => boxes.push((i, Some(sb))),
                        // Degrade to the shared tree rather than failing the task.
                        Err(e) => {
                            eprintln!("{ind}  [sandbox unavailable ({e}); {} runs in the shared tree]", parsed[i].label);
                            boxes.push((i, None));
                        }
                    }
                }
                std::thread::scope(|scope| {
                    let mut handles = Vec::new();
                    for (i, sb) in &boxes {
                        let (i, c) = (*i, &parsed[*i]);
                        eprintln!("{ind}  · {}", c.label);
                        let args = c.args.as_ref().unwrap();
                        let root = sb.as_ref().map(|s| s.dir.clone());
                        handles.push((i, scope.spawn(move || run_subagent(cfg, args, depth, root))));
                    }
                    for (i, h) in handles {
                        outs[i] = Some(h.join().unwrap_or_else(|_| "error: subagent thread panicked".into()));
                    }
                });
                // Merge each sandbox back, telling any subagent whose file a
                // sibling also changed that its write was withheld.
                let per: Vec<(usize, sandbox::Changes)> =
                    boxes.iter().filter_map(|(i, sb)| sb.as_ref().map(|s| (*i, s.changes(&base)))).collect();
                let conflicts = sandbox::conflicts(&per);
                for (i, ch) in &per {
                    let dir = &boxes.iter().find(|(j, _)| j == i).unwrap().1.as_ref().unwrap().dir;
                    let note = match sandbox::merge(dir, ch, &conflicts, Path::new(".")) {
                        Ok(s) if !s.is_empty() => format!(
                            "\n[conflict: your changes to {} were not applied — another parallel subagent changed the same file(s); split the work or run them sequentially]",
                            s.join(", ")
                        ),
                        Ok(_) => continue,
                        Err(e) => format!("\n[warning: merging your changes back failed: {e}]"),
                    };
                    if let Some(o) = outs[*i].as_mut() {
                        o.push_str(&note);
                    }
                }
            }
            // Non-subagent calls run after the subagents have joined, not
            // alongside them: the parent's own edits/bash would otherwise race
            // a subagent mutating the shared working tree.
            for (i, c) in parsed.iter().enumerate() {
                if !c.is_subagent() {
                    outs[i] = Some(run_call(cfg, tools, c, depth, &ind));
                }
            }
        } else {
            for (i, c) in parsed.iter().enumerate() {
                outs[i] = Some(run_call(cfg, tools, c, depth, &ind));
            }
        }

        let mut changed = false;
        for (i, c) in parsed.iter().enumerate() {
            let out = outs[i].take().unwrap_or_default();
            // A subagent may have edited files on disk, so run the parent's
            // check afterward too.
            changed |= (matches!(c.name.as_str(), "edit" | "write") && out.starts_with("ok")) || c.name == "task";
            ctx.push_tool_result(&c.id, c.label.clone(), out);
        }

        made_edits |= changed;

        // One check per step, after all of the step's edits, attached to the
        // last result so the model sees it before deciding what's next.
        if changed {
            if let Some((_, report)) = tools.run_check() {
                eprintln!("{ind}  {}", report.lines().next().unwrap_or(""));
                ctx.append_to_last_tool_result(&report);
            }
        }
    }
    if depth == 0 {
        println!("[stopped after {} steps]", cfg.max_steps);
    } else {
        eprintln!("{ind}[subagent stopped after {} steps]", cfg.max_steps);
    }
    (Outcome::StepLimit, cfg.max_steps, last_text)
}

/// Run a delegated subtask in its own context and budget, returning only its
/// final summary to the caller. The subagent's reads/edits/output never enter
/// the parent's history — that isolation is the whole point. `root`, when set,
/// roots its file ops in a private sandbox copy of the tree (parallel mode).
fn run_subagent(cfg: &llm::Config, a: &Value, depth: usize, root: Option<PathBuf>) -> String {
    if depth + 1 > MAX_SUBAGENT_DEPTH {
        return format!("error: subagent depth limit ({MAX_SUBAGENT_DEPTH}) reached");
    }
    let role = a["role"].as_str().unwrap_or("");
    let description = a["description"].as_str().unwrap_or("");
    if description.trim().is_empty() {
        return "error: task needs a non-empty description".into();
    }
    if !cfg.profile.subagents.iter().any(|s| s == role) {
        return format!("error: unknown role '{role}'; available: {}", cfg.profile.subagents.join(", "));
    }
    let child_profile = match profile::select(&cfg.model, Some(role)) {
        Ok(p) => p,
        Err(e) => return format!("error: {e}"),
    };
    let child_cfg = cfg.for_subagent(child_profile);
    let p = &child_cfg.profile;
    let mut child_ctx = Context::new(&system_prompt(p), p.ctx);
    let mut child_tools = tools::Tools::new(child_cfg.yolo, p);
    if let Some(root) = root {
        child_tools = child_tools.in_dir(root);
    }
    let child_schemas = tools::schemas(p);

    eprintln!("{}╭─ subagent [{role}] {}", "  ".repeat(depth), tools::clip(description, 60));
    child_ctx.push(json!({"role": "user", "content": description}));
    let (outcome, steps, summary) = run_turn(&child_cfg, &mut child_ctx, &mut child_tools, &child_schemas, depth + 1);
    let (_, t) = child_ctx.totals();
    eprintln!("{}╰─ subagent [{role}] {outcome:?} in {steps} steps (~{} prompt tok)", "  ".repeat(depth), t.prompt);

    let summary = if summary.trim().is_empty() { "(subagent returned no summary)" } else { summary.trim() };
    format!("[subagent {role}: {outcome:?}, {steps} steps]\n{summary}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn call_parse_identifies_what_can_be_parallelized() {
        let task = json!({"id": "a", "function":
            {"name": "task", "arguments": r#"{"role":"small","description":"x"}"#}});
        let c = Call::parse(&task, true);
        assert!(c.is_subagent() && c.id == "a" && c.label.starts_with("task"));

        let read = json!({"id": "b", "function": {"name": "read", "arguments": r#"{"path":"p"}"#}});
        assert!(!Call::parse(&read, true).is_subagent());

        // A malformed task isn't parallelized — it runs inline so run_call can
        // report the parse error to the model.
        let bad = json!({"id": "c", "function": {"name": "task", "arguments": "{not json"}});
        assert!(!Call::parse(&bad, false).is_subagent());
    }

    #[test]
    fn read_only_profile_gets_a_planning_prompt() {
        let mut p = Profile::default();
        p.tools = vec!["read".into(), "grep".into()];
        let s = system_prompt(&p);
        assert!(s.contains("planning agent") && s.contains("cannot modify"));
        assert!(!s.contains("edit"));
        // A full-access profile still gets the edit-oriented prompt.
        assert!(system_prompt(&Profile::default()).contains("edit"));
    }
}
