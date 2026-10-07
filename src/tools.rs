//! Tools, each designed to return the fewest tokens that still let the
//! model make its next decision.

use crate::profile::{EditFormat, Profile, PromptStyle};
use regex::Regex;
use serde_json::{json, Value};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;
use wait_timeout::ChildExt;
use walkdir::WalkDir;

const READ_DEFAULT: usize = 200; // lines per read unless the model asks
const READ_MAX: usize = 500;
const LINE_MAX: usize = 300; // chars; minified/generated lines get clipped
const GREP_MAX: usize = 50;
const GLOB_MAX: usize = 100;
const BASH_HEAD: usize = 40;
const BASH_TAIL: usize = 60;
const SKIP_DIRS: &[&str] = &["target", "node_modules", "build", "dist", "__pycache__"];

pub const ALL_TOOLS: &[&str] = &["read", "edit", "write", "grep", "glob", "bash"];

/// Tool schemas are sent on *every* request, so descriptions are short by
/// default (~400 tokens for all six). The "guided" style adds a short example
/// to each; that costs a few hundred tokens more, which small models repay by
/// making fewer malformed calls.
pub fn schemas(p: &Profile) -> Value {
    let guided = p.prompt == PromptStyle::Guided;
    let whole = p.edit_format == EditFormat::Whole;
    let f = |name: &str, terse: &str, example: &str, props: Value, req: &[&str]| {
        let desc = if guided { format!("{terse} Example: {example}") } else { terse.to_string() };
        json!({"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}
        }})
    };
    let s = json!({"type": "string"});
    let i = json!({"type": "integer"});
    let write_desc = if whole {
        "Write a file's complete content. To change a file: read it, then write the full new version."
    } else {
        "Create or overwrite a whole file. Prefer edit for changes."
    };
    let all = [
        f("read", "Read file lines, numbered. Default 200 lines; page with offset.",
          r#"{"path": "src/main.rs", "offset": 1, "limit": 100}"#,
          json!({"path": s, "offset": {"type": "integer", "description": "1-based"}, "limit": i}), &["path"]),
        f("edit", "Replace exact text. old must match once (or set all). Returns only status.",
          r#"{"path": "src/a.rs", "old": "let x = 1;", "new": "let x = 2;"}"#,
          json!({"path": s, "old": s, "new": s, "all": {"type": "boolean"}}), &["path", "old", "new"]),
        f("write", write_desc, r#"{"path": "notes.txt", "content": "hello\n"}"#,
          json!({"path": s, "content": s}), &["path", "content"]),
        f("grep", "Regex search. Returns path:line: text, max 50.",
          r#"{"pattern": "fn parse_\\w+", "path": "src"}"#,
          json!({"pattern": s, "path": {"type": "string", "description": "dir, default ."}, "glob": s}), &["pattern"]),
        f("glob", "List files matching a glob like src/**/*.rs.", r#"{"pattern": "**/*.toml"}"#,
          json!({"pattern": s}), &["pattern"]),
        f("bash", "Run a shell command. Long output is truncated in the middle.",
          r#"{"command": "cargo test 2>&1 | tail -20"}"#,
          json!({"command": s}), &["command"]),
    ];
    let mut out: Vec<Value> =
        all.into_iter().filter(|t| p.tool_enabled(t["function"]["name"].as_str().unwrap())).collect();
    // `task` is offered only to agents configured with subagent roles, so an
    // ordinary agent's tool set (and cacheable prefix) is unchanged.
    if !p.subagents.is_empty() {
        let roles: Vec<Value> = p.subagents.iter().map(|r| json!(r)).collect();
        out.push(json!({"type": "function", "function": {
            "name": "task",
            "description": "Delegate a self-contained subtask to a subagent with a fresh context. \
You get back only its final summary, not its intermediate work, so give complete standalone instructions.",
            "parameters": {"type": "object", "properties": {
                "description": {"type": "string", "description": "the complete, standalone subtask"},
                "role": {"type": "string", "enum": roles, "description": "which subagent to use"}
            }, "required": ["description", "role"]}
        }}));
    }
    // `def`/`refs` are offered only when a language server is configured.
    if !p.lsp.is_empty() {
        let multi = p.lsp.len() > 1;
        let lang_prop = if multi {
            json!({"type": "string", "description": "file extension picking the server, e.g. rs"})
        } else {
            json!({"type": "string", "description": "optional; one server is configured"})
        };
        let lang_desc = if multi { " Pass lang to pick the server." } else { "" };
        out.push(json!({"type": "function", "function": {
            "name": "def",
            "description": format!("Find where a symbol is defined (via the language server). Returns path:line: source.{lang_desc}"),
            "parameters": {"type": "object", "properties": {"symbol": s, "lang": lang_prop}, "required": ["symbol"]}
        }}));
        out.push(json!({"type": "function", "function": {
            "name": "refs",
            "description": format!("Find all references to a symbol (via the language server). Returns path:line: source.{lang_desc}"),
            "parameters": {"type": "object", "properties": {"symbol": s, "lang": lang_prop}, "required": ["symbol"]}
        }}));
    }
    Value::Array(out)
}

pub struct Tools {
    yolo: bool,
    profile: Profile,
    /// Directory every relative path (and bash/check/lsp) resolves against.
    /// "." for an ordinary agent; a copied sandbox dir for an isolated subagent.
    root: PathBuf,
    /// Hash of what each read range last returned. Lets a repeat read return a
    /// one-line stub instead of the same 200 lines again.
    seen: HashMap<String, u64>,
    /// Language servers, started lazily per extension and kept for the session
    /// (each kills its child on drop).
    lsp: HashMap<String, crate::lsp::Lsp>,
}

impl Tools {
    pub fn new(yolo: bool, profile: &Profile) -> Self {
        Tools { yolo, profile: profile.clone(), root: PathBuf::from("."), seen: HashMap::new(), lsp: HashMap::new() }
    }

    /// Root this agent's file ops at `root` (a sandbox copy of the tree), so a
    /// parallel subagent can't clobber another's edits in the shared tree.
    pub fn in_dir(mut self, root: PathBuf) -> Self {
        self.root = root;
        self
    }

    /// The directory this agent's file ops resolve against. For a parallel
    /// subagent whose own tools are already rooted in a sandbox, its nested
    /// parallel batch must snapshot/copy/merge against *this* root, not the
    /// process CWD, or the nested edits would leak into the real tree.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve a model-supplied relative path against the working root.
    fn path(&self, p: &str) -> PathBuf {
        self.root.join(p)
    }

    /// Called when old results are dropped from context: the model no longer
    /// has them, so "unchanged since last read" would be a lie.
    pub fn forget_reads(&mut self) {
        self.seen.clear();
    }

    pub fn run(&mut self, name: &str, a: &Value) -> String {
        if ALL_TOOLS.contains(&name) && !self.profile.tool_enabled(name) {
            return if name == "edit" {
                "error: edit is not available. Read the file, then write its full new content.".into()
            } else {
                format!("error: {name} is not available in this session")
            };
        }
        match name {
            "read" => self.read(a),
            "edit" => self.edit(a),
            "write" => self.write(a),
            "grep" => grep(a, &self.root),
            "glob" => glob_files(a, &self.root),
            "bash" => self.bash(a),
            "def" | "refs" => self.lsp_query(name, a),
            _ => format!("error: unknown tool {name}"),
        }
    }

    /// `def`/`refs` via a language server. Any failure degrades to a hint to use
    /// grep rather than stalling the agent.
    fn lsp_query(&mut self, kind: &str, a: &Value) -> String {
        if self.profile.lsp.is_empty() {
            return "error: no language server configured; use grep".into();
        }
        let symbol = arg(a, "symbol");
        if symbol.is_empty() {
            return "error: symbol is required".into();
        }
        // Pick the server: by `lang`, else the sole configured one, else ask.
        let ext = match a["lang"].as_str() {
            Some(l) if self.profile.lsp.contains_key(l) => l.to_string(),
            Some(l) => return format!("error: no language server configured for '{l}'"),
            None if self.profile.lsp.len() == 1 => self.profile.lsp.keys().next().unwrap().clone(),
            None => {
                let mut langs: Vec<&String> = self.profile.lsp.keys().collect();
                langs.sort();
                return format!(
                    "error: multiple language servers configured ({}); pass lang",
                    langs.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                );
            }
        };
        if !self.lsp.contains_key(&ext) {
            let cmd = self.profile.lsp[&ext].clone();
            let secs = if self.profile.bash_timeout == 0 { 60 } else { self.profile.bash_timeout };
            let root = self.root.canonicalize().unwrap_or_else(|_| self.root.clone());
            match crate::lsp::Lsp::start(&cmd, &root, Duration::from_secs(secs)) {
                Ok(l) => {
                    self.lsp.insert(ext.clone(), l);
                }
                Err(e) => return format!("error: lsp unavailable ({e}); use grep"),
            }
        }
        let server = self.lsp.get_mut(&ext).unwrap();
        let r = if kind == "def" { server.def(symbol, GREP_MAX) } else { server.refs(symbol, GREP_MAX) };
        match r {
            Ok(s) => s,
            Err(e) => format!("error: lsp {kind} failed ({e}); use grep"),
        }
    }

    fn read(&mut self, a: &Value) -> String {
        let path = arg(a, "path");
        let text = match fs::read_to_string(self.path(path)) {
            Ok(t) => t,
            Err(e) => return format!("error: {e}"),
        };
        let lines: Vec<&str> = text.lines().collect();
        let total = lines.len();
        let off = a["offset"].as_u64().unwrap_or(1).max(1) as usize;
        let lim = (a["limit"].as_u64().unwrap_or(READ_DEFAULT as u64) as usize).clamp(1, READ_MAX);
        if off - 1 > total {
            return format!("error: offset {off} is past end ({total} lines)");
        }
        let end = (off - 1 + lim).min(total);

        let mut out = String::new();
        for (i, l) in lines[off - 1..end].iter().enumerate() {
            let _ = writeln!(out, "{:>5} {}", off + i, clip(l, LINE_MAX));
        }
        if end < total {
            let _ = writeln!(out, "[{} more lines; offset={} to continue]", total - end, end + 1);
        }

        let key = format!("{path}:{off}:{lim}");
        let h = hash(&out);
        if self.seen.get(&key) == Some(&h) {
            return "[unchanged since your last read of this range]".into();
        }
        self.seen.insert(key, h);
        out
    }

    /// Returns a status line, never the file. Echoing the edited file back
    /// is the single biggest waste in naive agents.
    fn edit(&mut self, a: &Value) -> String {
        let (path, old, new) = (arg(a, "path"), arg(a, "old"), arg(a, "new"));
        let all = a["all"].as_bool().unwrap_or(false);
        if old.is_empty() {
            return "error: old is empty; use write to create files".into();
        }
        let text = match fs::read_to_string(self.path(path)) {
            Ok(t) => t,
            Err(e) => return format!("error: {e}"),
        };
        let n = text.matches(old).count();
        if n == 0 && self.profile.edit_format == EditFormat::Fuzzy {
            match fuzzy_replace(&text, old, new) {
                Ok((updated, line)) => {
                    if let Err(e) = fs::write(self.path(path), updated) {
                        return format!("error: {e}");
                    }
                    self.invalidate(path);
                    return format!("ok: 1 replacement in {path} at line {line} (matched ignoring indentation)");
                }
                Err(k) if k > 1 => {
                    return format!("error: old text matches {k} places ignoring indentation; include more context");
                }
                Err(_) => {}
            }
        }
        if n == 0 {
            let hint = if normalize(&text).contains(&normalize(old)) {
                " (it would match ignoring indentation; copy whitespace exactly)"
            } else {
                ""
            };
            return format!("error: old text not found in {path}{hint}");
        }
        if n > 1 && !all {
            return format!("error: old text matches {n} places; include more context or set all=true");
        }
        let line = text[..text.find(old).unwrap()].matches('\n').count() + 1;
        let updated = if all { text.replace(old, new) } else { text.replacen(old, new, 1) };
        if let Err(e) = fs::write(self.path(path), updated) {
            return format!("error: {e}");
        }
        self.invalidate(path);
        format!("ok: {} replacement(s) in {path}, first at line {line}", if all { n } else { 1 })
    }

    fn write(&mut self, a: &Value) -> String {
        let (path, content) = (arg(a, "path"), arg(a, "content"));
        let full = self.path(path);
        if let Some(dir) = full.parent().filter(|d| !d.as_os_str().is_empty()) {
            let _ = fs::create_dir_all(dir);
        }
        match fs::write(&full, content) {
            Ok(_) => {
                self.invalidate(path);
                format!("ok: wrote {path} ({} lines)", content.lines().count())
            }
            Err(e) => format!("error: {e}"),
        }
    }

    fn bash(&self, a: &Value) -> String {
        let cmd = arg(a, "command");
        if !self.yolo && !confirm(&format!("run `{cmd}`?")) {
            return "error: user declined this command".into();
        }
        match shell(cmd, self.profile.bash_timeout, &self.root) {
            Ok((code, text)) => {
                let shown = truncate_middle(&text, BASH_HEAD, BASH_TAIL);
                format!("exit {code}\n{shown}{}", spill(&text, BASH_HEAD, BASH_TAIL))
            }
            Err(e) => format!("error: {e}"),
        }
    }

    /// Runs the profile's check command (configured by the user, so no
    /// confirmation). A pass costs the model one short line; only failures
    /// carry output.
    pub fn run_check(&self) -> Option<(bool, String)> {
        let cmd = self.profile.check.as_deref()?;
        Some(match shell(cmd, self.profile.bash_timeout, &self.root) {
            Ok((0, _)) => (true, format!("[check `{cmd}`: passed]")),
            Ok((code, text)) => (false, format!("[check `{cmd}`: exit {code}]\n{}", truncate_middle(&text, 30, 30))),
            Err(e) => (false, format!("[check `{cmd}` could not run: {e}]")),
        })
    }

    fn invalidate(&mut self, path: &str) {
        let prefix = format!("{path}:");
        self.seen.retain(|k, _| !k.starts_with(&prefix));
    }
}

fn grep(a: &Value, root: &Path) -> String {
    let re = match Regex::new(arg(a, "pattern")) {
        Ok(r) => r,
        Err(e) => return format!("error: bad regex: {e}"),
    };
    let sub = a["path"].as_str().unwrap_or(".");
    let base = if sub == "." { root.to_path_buf() } else { root.join(sub) };
    let filter = a["glob"].as_str().and_then(|g| glob::Pattern::new(g).ok());
    let (mut out, mut hits, mut files) = (String::new(), 0usize, 0usize);

    for e in walk(&base.to_string_lossy()) {
        let p = e.path();
        if let Some(f) = &filter {
            let name = e.file_name().to_string_lossy();
            if !f.matches_path(p) && !f.matches(&name) {
                continue;
            }
        }
        let Ok(text) = fs::read_to_string(p) else { continue }; // skips binary
        let mut counted = false;
        for (i, l) in text.lines().enumerate() {
            if re.is_match(l) {
                hits += 1;
                if !counted {
                    files += 1;
                    counted = true;
                }
                if hits <= GREP_MAX {
                    let _ = writeln!(out, "{}:{}: {}", display(p, root), i + 1, clip(l.trim(), 200));
                }
            }
        }
    }
    match hits {
        0 => "no matches".into(),
        h if h > GREP_MAX => {
            let _ = writeln!(out, "[{} more matches across {files} files; narrow pattern/path]", h - GREP_MAX);
            out
        }
        _ => out,
    }
}

fn glob_files(a: &Value, root: &Path) -> String {
    let pattern = root.join(arg(a, "pattern"));
    let paths = match glob::glob(&pattern.to_string_lossy()) {
        Ok(p) => p,
        Err(e) => return format!("error: {e}"),
    };
    let mut found: Vec<String> = paths
        .filter_map(Result::ok)
        .filter(|p| p.is_file() && !p.strip_prefix(root).unwrap_or(p).components().any(|c| is_skipped(&c.as_os_str().to_string_lossy())))
        .map(|p| display(&p, root))
        .collect();
    if found.is_empty() {
        return "no files".into();
    }
    let extra = found.len().saturating_sub(GLOB_MAX);
    found.truncate(GLOB_MAX);
    let mut out = found.join("\n");
    if extra > 0 {
        let _ = write!(out, "\n[{extra} more; narrow the pattern]");
    }
    out
}

// ---------- helpers ----------

/// Run a shell command. `timeout` in seconds, 0 = no limit. On timeout the
/// child is killed so a hung command can't wedge an (unattended) agent.
fn shell(cmd: &str, timeout: u64, cwd: &Path) -> Result<(i32, String), String> {
    let mut command = if cfg!(windows) { Command::new("cmd") } else { Command::new("sh") };
    command.args(if cfg!(windows) { ["/C", cmd] } else { ["-c", cmd] });
    command.current_dir(cwd);

    if timeout == 0 {
        let o = command.output().map_err(|e| e.to_string())?;
        return Ok((o.status.code().unwrap_or(-1), combine(&o.stdout, &o.stderr)));
    }

    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;
    // Drain both pipes on their own threads so large output can't fill the
    // pipe buffer and deadlock us while we wait on the timeout.
    let mut so = child.stdout.take().unwrap();
    let mut se = child.stderr.take().unwrap();
    let t_out = std::thread::spawn(move || { let mut b = Vec::new(); let _ = so.read_to_end(&mut b); b });
    let t_err = std::thread::spawn(move || { let mut b = Vec::new(); let _ = se.read_to_end(&mut b); b });

    let status = match child.wait_timeout(Duration::from_secs(timeout)).map_err(|e| e.to_string())? {
        Some(s) => s,
        None => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("command timed out after {timeout}s and was killed"));
        }
    };
    let out = t_out.join().unwrap_or_default();
    let err = t_err.join().unwrap_or_default();
    Ok((status.code().unwrap_or(-1), combine(&out, &err)))
}

fn combine(stdout: &[u8], stderr: &[u8]) -> String {
    let mut text = String::from_utf8_lossy(stdout).into_owned();
    let err = String::from_utf8_lossy(stderr);
    if !err.trim().is_empty() {
        text.push_str("\n[stderr]\n");
        text.push_str(&err);
    }
    strip_ansi(&text)
}

/// Line-based match that ignores leading/trailing whitespace on each line and
/// CRLF vs LF. Re-indents the replacement to the indentation actually found.
/// Err(n) = number of matches when not exactly one.
fn fuzzy_replace(text: &str, old: &str, new: &str) -> Result<(String, usize), usize> {
    let eol = if text.contains("\r\n") { "\r\n" } else { "\n" };
    let lines: Vec<&str> = text.lines().collect();
    let old_lines: Vec<&str> = {
        let v: Vec<&str> = old.lines().collect();
        let start = v.iter().position(|l| !l.trim().is_empty()).unwrap_or(v.len());
        let end = v.iter().rposition(|l| !l.trim().is_empty()).map_or(start, |e| e + 1);
        v[start..end].to_vec()
    };
    let key: Vec<&str> = old_lines.iter().map(|l| l.trim()).collect();
    if key.is_empty() || key.len() > lines.len() {
        return Err(0);
    }
    let hits: Vec<usize> = (0..=lines.len() - key.len())
        .filter(|&i| key.iter().enumerate().all(|(j, k)| lines[i + j].trim() == *k))
        .collect();
    if hits.len() != 1 {
        return Err(hits.len());
    }
    let start = hits[0];
    let indent = |l: &str| l[..l.len() - l.trim_start().len()].to_string();
    let (found, given) = (indent(lines[start]), indent(old_lines[0]));
    let replacement = new.lines().map(|l| match l.strip_prefix(given.as_str()) {
        Some(rest) if !l.trim().is_empty() => format!("{found}{rest}"),
        _ => l.to_string(),
    });

    let mut out: Vec<String> = lines[..start].iter().map(|s| s.to_string()).collect();
    out.extend(replacement);
    out.extend(lines[start + key.len()..].iter().map(|s| s.to_string()));
    let mut s = out.join(eol);
    if text.ends_with('\n') {
        s.push_str(eol);
    }
    Ok((s, start + 1))
}

pub(crate) fn walk(root: &str) -> impl Iterator<Item = walkdir::DirEntry> {
    WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| e.depth() == 0 || !is_skipped(&e.file_name().to_string_lossy()))
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
}

pub(crate) fn is_skipped(name: &str) -> bool {
    (name.starts_with('.') && name.len() > 1 && name != "..") || SKIP_DIRS.contains(&name)
}

/// Format a path for the model: relative to the working root, forward slashes.
/// Paths are shown as the model would supply them, never the sandbox prefix.
fn display(p: &Path, root: &Path) -> String {
    let p = p.strip_prefix(root).unwrap_or(p);
    let s = p.to_string_lossy();
    s.strip_prefix("./").unwrap_or(&s).replace('\\', "/")
}

fn arg<'a>(a: &'a Value, k: &str) -> &'a str {
    a[k].as_str().unwrap_or("")
}

pub fn clip(s: &str, max: usize) -> String {
    let n = s.chars().count();
    if n <= max {
        s.to_string()
    } else {
        let head: String = s.chars().take(max).collect();
        format!("{head}…[+{} chars]", n - max)
    }
}

/// Keep the start (first error) and the end (summary/exit status) of long output.
pub fn truncate_middle(s: &str, head: usize, tail: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    let clipped = |ls: &[&str]| ls.iter().map(|l| clip(l, LINE_MAX)).collect::<Vec<_>>().join("\n");
    if lines.len() <= head + tail {
        return clipped(&lines);
    }
    format!(
        "{}\n[… {} lines omitted …]\n{}",
        clipped(&lines[..head]),
        lines.len() - head - tail,
        clipped(&lines[lines.len() - tail..])
    )
}

/// When `truncate_middle` would drop content (lines omitted, or a line clipped
/// past LINE_MAX), save the full output to a temp file and return a one-line
/// pointer so the model can `read` it if it needs the detail. Costs ~15 tokens,
/// and only when something was actually lost. Returns "" otherwise, or on a
/// write failure (the truncated view is still useful on its own). The temp file
/// lives outside the working root, so a sandboxed subagent never merges it back.
fn spill(full: &str, head: usize, tail: usize) -> String {
    let lost = full.lines().count() > head + tail || full.lines().any(|l| l.chars().count() > LINE_MAX);
    if !lost {
        return String::new();
    }
    let path = std::env::temp_dir().join(format!("tern-bash-{:016x}.txt", hash(full)));
    match fs::write(&path, full) {
        Ok(_) => format!("\n[full output ({} lines) saved to {} — read it for the omitted detail]", full.lines().count(), path.display()),
        Err(_) => String::new(),
    }
}

fn strip_ansi(s: &str) -> String {
    Regex::new(r"\x1b\[[0-9;?]*[A-Za-z]").unwrap().replace_all(s, "").into_owned()
}

fn normalize(s: &str) -> String {
    s.lines().map(str::trim).collect::<Vec<_>>().join("\n")
}

pub(crate) fn hash(s: &str) -> u64 {
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn confirm(q: &str) -> bool {
    eprint!("  ? {q} [y/N] ");
    let _ = io::stderr().flush();
    let mut line = String::new();
    io::stdin().lock().read_line(&mut line).is_ok() && line.trim().eq_ignore_ascii_case("y")
}

pub fn summarize_args(a: &Value) -> String {
    let s = ["pattern", "command", "path", "description", "symbol"]
        .iter()
        .find_map(|k| a[*k].as_str())
        .map(str::to_string)
        .unwrap_or_else(|| a.to_string());
    clip(&s, 70)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str, body: &str) -> String {
        let p = std::env::temp_dir().join(format!("tern_test_{name}"));
        fs::write(&p, body).unwrap();
        p.to_string_lossy().into_owned()
    }

    #[test]
    fn middle_truncation_keeps_ends() {
        let s: String = (1..=200).map(|i| format!("line{i}\n")).collect();
        let t = truncate_middle(&s, 3, 2);
        assert!(t.starts_with("line1\nline2\nline3\n[… 195 lines omitted …]"));
        assert!(t.ends_with("line199\nline200"));
    }

    #[test]
    fn spill_saves_full_output_only_when_truncated() {
        // Short output: nothing lost, no spill.
        assert_eq!(spill("one\ntwo\nthree", 3, 2), "");

        // More lines than head+tail: full output saved to a readable file.
        let s: String = (1..=50).map(|i| format!("line{i}\n")).collect();
        let note = spill(&s, 3, 2);
        assert!(note.contains("full output (50 lines) saved to"));
        let path = note.rsplit("saved to ").next().unwrap().split(" —").next().unwrap();
        assert_eq!(fs::read_to_string(path).unwrap(), s);

        // A single over-long line also loses content and spills.
        let long = "x".repeat(LINE_MAX + 1);
        assert!(spill(&long, 3, 2).contains("full output"));
    }

    #[test]
    fn edit_requires_unique_match_and_returns_status_only() {
        let p = tmp("edit", "a\nfoo\nb\nfoo\n");
        let mut t = Tools::new(true, &Profile::default());
        let r = t.run("edit", &json!({"path": p, "old": "foo", "new": "bar"}));
        assert!(r.contains("matches 2 places"), "{r}");
        let r = t.run("edit", &json!({"path": p, "old": "b\nfoo", "new": "b\nbaz"}));
        assert_eq!(r, format!("ok: 1 replacement(s) in {p}, first at line 3"));
        assert_eq!(fs::read_to_string(&p).unwrap(), "a\nfoo\nb\nbaz\n");
    }

    #[test]
    fn edit_hints_on_whitespace_mismatch() {
        let p = tmp("ws", "fn x() {\n    let a = 1;\n}\n");
        let r = Tools::new(true, &Profile::default()).run("edit", &json!({"path": p, "old": "let a = 1;\n  }", "new": "x"}));
        assert!(r.contains("ignoring indentation"), "{r}");
    }

    #[test]
    fn repeat_read_is_stubbed_until_file_changes() {
        let p = tmp("read", "one\ntwo\n");
        let mut t = Tools::new(true, &Profile::default());
        assert!(t.run("read", &json!({"path": p})).contains("one"));
        assert!(t.run("read", &json!({"path": p})).starts_with("[unchanged"));
        t.run("edit", &json!({"path": p, "old": "two", "new": "three"}));
        assert!(t.run("read", &json!({"path": p})).contains("three"));
    }

    #[test]
    fn read_pages_long_files() {
        let body: String = (1..=300).map(|i| format!("{i}\n")).collect();
        let p = tmp("long", &body);
        let r = Tools::new(true, &Profile::default()).run("read", &json!({"path": p, "limit": 10}));
        assert!(r.contains("[290 more lines; offset=11 to continue]"), "{r}");
    }
    fn fuzzy() -> Profile {
        Profile { edit_format: EditFormat::Fuzzy, ..Profile::default() }
    }

    #[test]
    fn fuzzy_edit_reindents_and_keeps_crlf() {
        let p = tmp("fuzzy", "fn x() {\r\n    if a {\r\n        go();\r\n    }\r\n}\r\n");
        let r = Tools::new(true, &fuzzy()).run("edit", &json!({"path": p, "old": "if a {\n    go();\n}", "new": "if b {\n    stop();\n}"}));
        assert!(r.contains("at line 2 (matched ignoring indentation)"), "{r}");
        assert_eq!(fs::read_to_string(&p).unwrap(), "fn x() {\r\n    if b {\r\n        stop();\r\n    }\r\n}\r\n");
    }

    #[test]
    fn fuzzy_edit_refuses_ambiguous_matches() {
        let p = tmp("fuzzy2", "  a\n  b\na\nb\n");
        let r = Tools::new(true, &fuzzy()).run("edit", &json!({"path": p, "old": "a\nb\n", "new": "c"}));
        // exact match exists once ("a\nb\n" at end), so exact wins
        assert!(r.starts_with("ok: 1 replacement(s)"), "{r}");
        let p = tmp("fuzzy3", "  a\n  b\n\ta\n\tb\n");
        let r = Tools::new(true, &fuzzy()).run("edit", &json!({"path": p, "old": "a\nb", "new": "c"}));
        assert!(r.contains("matches 2 places ignoring indentation"), "{r}");
    }

    #[test]
    #[cfg(unix)]
    fn bash_spills_long_output_to_a_readable_file() {
        let mut t = Tools::new(true, &Profile::default());
        // 300 lines of output: truncated in context, full copy spilled to a file.
        let r = t.run("bash", &json!({"command": "seq 1 300"}));
        assert!(r.contains("[… "), "should truncate in context: {r}");
        assert!(r.contains("full output (300 lines) saved to"), "{r}");
        // The model can read exactly what the note points at.
        let path = r.rsplit("saved to ").next().unwrap().split(" —").next().unwrap();
        let full = fs::read_to_string(path).unwrap();
        assert_eq!(full.lines().count(), 300);
        assert!(full.starts_with("1\n") && full.trim_end().ends_with("\n300"));

        // Short output stays inline with no spill note.
        let r = t.run("bash", &json!({"command": "echo hi"}));
        assert!(!r.contains("full output"), "{r}");
    }

    #[test]
    #[cfg(unix)]
    fn bash_times_out_and_is_killed() {
        let start = std::time::Instant::now();
        let r = shell("sleep 10", 1, Path::new("."));
        assert!(r.is_err() && r.unwrap_err().contains("timed out"), "should time out");
        assert!(start.elapsed() < Duration::from_secs(5), "should return promptly after kill");
        // 0 = no limit still works for quick commands.
        assert_eq!(shell("exit 3", 0, Path::new(".")).unwrap().0, 3);
    }

    #[test]
    fn rooted_tools_redirect_to_the_root_with_relative_display() {
        // A sandbox dir the model knows nothing about.
        let root = std::env::temp_dir().join(format!("tern_root_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/a.rs"), "fn needle() {}\n").unwrap();

        let mut t = Tools::new(true, &Profile::default()).in_dir(root.clone());

        // write/read resolve under the root, not the process CWD.
        assert!(t.run("write", &json!({"path": "src/b.rs", "content": "let x = 1;\n"})).starts_with("ok"));
        assert_eq!(fs::read_to_string(root.join("src/b.rs")).unwrap(), "let x = 1;\n");
        assert!(t.run("read", &json!({"path": "src/a.rs"})).contains("needle"));

        // grep/glob search the root but report paths as the model gave them.
        let g = t.run("grep", &json!({"pattern": "needle"}));
        assert!(g.contains("src/a.rs:1:") && !g.contains("tern_root"), "{g}");
        let gl = t.run("glob", &json!({"pattern": "src/*.rs"}));
        assert!(gl.contains("src/a.rs") && gl.contains("src/b.rs") && !gl.contains("tern_root"), "{gl}");

        // bash runs with the root as its working directory.
        #[cfg(unix)]
        assert!(t.run("bash", &json!({"command": "cat src/a.rs"})).contains("needle"));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn task_tool_only_offered_with_subagents() {
        let plain = schemas(&Profile::default());
        assert!(!plain.as_array().unwrap().iter().any(|t| t["function"]["name"] == "task"));
        let orch = Profile { subagents: vec!["small".into()], ..Profile::default() };
        let t = schemas(&orch);
        let task = t.as_array().unwrap().iter().find(|t| t["function"]["name"] == "task").unwrap().clone();
        assert_eq!(task["function"]["parameters"]["properties"]["role"]["enum"][0], "small");
    }

    #[test]
    fn lsp_tools_only_offered_when_configured() {
        let plain = schemas(&Profile::default());
        let names: Vec<&str> =
            plain.as_array().unwrap().iter().filter_map(|t| t["function"]["name"].as_str()).collect();
        assert!(!names.contains(&"def") && !names.contains(&"refs"));

        let mut lsp = std::collections::HashMap::new();
        lsp.insert("rs".to_string(), "rust-analyzer".to_string());
        let p = Profile { lsp, ..Profile::default() };
        let sch = schemas(&p);
        let names: Vec<&str> =
            sch.as_array().unwrap().iter().filter_map(|t| t["function"]["name"].as_str()).collect();
        assert!(names.contains(&"def") && names.contains(&"refs"));
        // No server running, so a query degrades to a grep hint, never panics.
        let r = Tools::new(true, &p).run("def", &json!({"symbol": "nope", "lang": "py"}));
        assert!(r.contains("no language server configured for 'py'"), "{r}");

        // A server command that can't be spawned degrades to a grep hint too.
        let mut bad = std::collections::HashMap::new();
        bad.insert("rs".to_string(), "tern-no-such-lsp-binary".to_string());
        let p = Profile { lsp: bad, bash_timeout: 2, ..Profile::default() };
        let r = Tools::new(true, &p).run("def", &json!({"symbol": "x"}));
        assert!(r.contains("lsp unavailable") && r.contains("use grep"), "{r}");
    }

    #[test]
    fn whole_mode_disables_edit() {
        let prof = Profile { edit_format: EditFormat::Whole, ..Profile::default() };
        let r = Tools::new(true, &prof).run("edit", &json!({"path": "x", "old": "a", "new": "b"}));
        assert!(r.contains("write its full new content"), "{r}");
        let names: Vec<String> = schemas(&prof).as_array().unwrap().iter().map(|t| t["function"]["name"].as_str().unwrap().to_string()).collect();
        assert!(!names.contains(&"edit".to_string()) && names.contains(&"write".to_string()));
    }
}
