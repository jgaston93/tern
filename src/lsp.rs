//! Minimal LSP client over stdio JSON-RPC. No external crates: serde_json for
//! messages, std::process for the server, and a reader thread + channel for a
//! bounded read (the same pattern as tools::shell's timeout handling).
//!
//! Only what `def`/`refs` need: initialize, workspace/symbol, didOpen, and
//! textDocument/references. We advertise empty capabilities, so the server
//! won't send client-side requests that expect a reply — unmatched messages
//! are simply ignored.

use crate::tools::clip;
use serde_json::{json, Value};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

pub struct Lsp {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<Value>,
    next_id: i64,
    timeout: Duration,
}

impl Drop for Lsp {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Lsp {
    /// Start a server (command may include arguments), initialize it against
    /// `root`, and return a ready client. Any failure is reported so the caller
    /// can fall back to grep.
    pub fn start(cmd: &str, root: &Path, timeout: Duration) -> Result<Lsp, String> {
        let mut parts = cmd.split_whitespace();
        let prog = parts.next().ok_or("empty lsp command")?;
        let mut child = Command::new(prog)
            .args(parts)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null()) // rust-analyzer logs heavily; we don't need it
            .spawn()
            .map_err(|e| format!("could not start '{cmd}': {e}"))?;
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || reader(stdout, tx));

        let mut lsp = Lsp { child, stdin, rx, next_id: 0, timeout };
        let uri = path_to_uri(root);
        lsp.request(
            "initialize",
            json!({
                "processId": std::process::id(),
                "rootUri": uri,
                "capabilities": {},
                "workspaceFolders": [{"uri": uri, "name": "root"}],
            }),
        )?;
        lsp.notify("initialized", json!({}));
        Ok(lsp)
    }

    /// Definition sites for a symbol. workspace/symbol already returns the
    /// definition location, so this needs no second round-trip.
    pub fn def(&mut self, symbol: &str, cap: usize) -> Result<String, String> {
        let syms = self.workspace_symbol(symbol)?;
        let locs: Vec<Value> = exact_or_all(&syms, symbol)
            .iter()
            .filter_map(|s| s.get("location").cloned())
            .collect();
        Ok(format_locations(&locs, cap))
    }

    /// All references to a symbol, including its declaration.
    pub fn refs(&mut self, symbol: &str, cap: usize) -> Result<String, String> {
        let syms = self.workspace_symbol(symbol)?;
        let chosen = exact_or_all(&syms, symbol);
        let loc = chosen
            .first()
            .and_then(|s| s.get("location"))
            .ok_or_else(|| format!("no symbol named '{symbol}'"))?;
        let uri = loc["uri"].as_str().ok_or("symbol location missing uri")?.to_string();
        let pos = loc["range"]["start"].clone();
        self.did_open(&uri)?;
        let result = self.request(
            "textDocument/references",
            json!({
                "textDocument": {"uri": uri},
                "position": pos,
                "context": {"includeDeclaration": true},
            }),
        )?;
        let locs = result.as_array().cloned().unwrap_or_default();
        Ok(format_locations(&locs, cap))
    }

    /// Query workspace symbols, retrying briefly while the server finishes
    /// indexing (it answers quickly with an empty list until then).
    fn workspace_symbol(&mut self, query: &str) -> Result<Vec<Value>, String> {
        let deadline = Instant::now() + self.timeout;
        loop {
            let r = self.request("workspace/symbol", json!({"query": query}))?;
            let arr = r.as_array().cloned().unwrap_or_default();
            if !arr.is_empty() || Instant::now() >= deadline {
                return Ok(arr);
            }
            std::thread::sleep(Duration::from_millis(300));
        }
    }

    fn did_open(&mut self, uri: &str) -> Result<(), String> {
        let path = uri_to_path(uri);
        let text = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
        self.notify(
            "textDocument/didOpen",
            json!({"textDocument": {"uri": uri, "languageId": lang_id(&path), "version": 1, "text": text}}),
        );
        Ok(())
    }

    fn request(&mut self, method: &str, params: Value) -> Result<Value, String> {
        self.next_id += 1;
        let id = self.next_id;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))?;
        let deadline = Instant::now() + self.timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(format!("timed out waiting for {method}"));
            }
            match self.rx.recv_timeout(remaining) {
                // A response carries our id and no `method`. Server-initiated
                // requests (e.g. window/workDoneProgress/create) reuse the same
                // integer id space, so match on both or we'd mistake one for our
                // reply and return its (absent) result as Null.
                Ok(v) if v.get("method").is_none() && v.get("id").and_then(Value::as_i64) == Some(id) => {
                    if let Some(e) = v.get("error") {
                        return Err(format!("{method}: {e}"));
                    }
                    return Ok(v.get("result").cloned().unwrap_or(Value::Null));
                }
                Ok(_) => {} // notification, server request, or unrelated id; ignore
                Err(_) => return Err(format!("timed out waiting for {method}")),
            }
        }
    }

    fn notify(&mut self, method: &str, params: Value) {
        let _ = self.send(&json!({"jsonrpc": "2.0", "method": method, "params": params}));
    }

    fn send(&mut self, v: &Value) -> Result<(), String> {
        self.stdin.write_all(encode(v).as_bytes()).map_err(|e| e.to_string())?;
        self.stdin.flush().map_err(|e| e.to_string())
    }
}

// ---------- framing (pure, unit-tested) ----------

fn encode(v: &Value) -> String {
    let s = v.to_string();
    format!("Content-Length: {}\r\n\r\n{s}", s.len())
}

/// Read one `Content-Length`-framed message. `Ok(None)` at EOF.
fn read_frame<R: BufRead>(r: &mut R) -> io::Result<Option<Value>> {
    let mut len = 0usize;
    loop {
        let mut line = String::new();
        if r.read_line(&mut line)? == 0 {
            return Ok(None);
        }
        let line = line.trim_end();
        if line.is_empty() {
            break; // blank line ends the headers
        }
        if let Some(rest) = line.strip_prefix("Content-Length:") {
            len = rest.trim().parse().unwrap_or(0);
        }
    }
    if len == 0 {
        return Ok(Some(Value::Null));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(Some(serde_json::from_slice(&buf).unwrap_or(Value::Null)))
}

fn reader(stdout: ChildStdout, tx: Sender<Value>) {
    let mut r = BufReader::new(stdout);
    while let Ok(Some(v)) = read_frame(&mut r) {
        if tx.send(v).is_err() {
            break; // client dropped
        }
    }
}

// ---------- helpers ----------

/// Symbols whose name matches exactly, or all of them if none match (the
/// server's fuzzy match is the best we have).
fn exact_or_all<'a>(syms: &'a [Value], name: &str) -> Vec<&'a Value> {
    let exact: Vec<&Value> = syms.iter().filter(|s| s["name"].as_str() == Some(name)).collect();
    if exact.is_empty() {
        syms.iter().collect()
    } else {
        exact
    }
}

/// Render LSP locations as grep-style `path:line: source`, capped.
fn format_locations(locs: &[Value], cap: usize) -> String {
    if locs.is_empty() {
        return "no results".into();
    }
    let mut out = String::new();
    for loc in locs.iter().take(cap) {
        let path = uri_to_path(loc["uri"].as_str().unwrap_or(""));
        let line0 = loc["range"]["start"]["line"].as_u64().unwrap_or(0) as usize;
        let src = line_at(&path, line0);
        out.push_str(&format!("{}:{}: {}\n", rel(&path), line0 + 1, clip(src.trim(), 200)));
    }
    if locs.len() > cap {
        out.push_str(&format!("[{} more; narrow the symbol]\n", locs.len() - cap));
    }
    out
}

fn line_at(path: &Path, line0: usize) -> String {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|t| t.lines().nth(line0).map(str::to_string))
        .unwrap_or_default()
}

fn rel(path: &Path) -> String {
    let s = std::env::current_dir()
        .ok()
        .and_then(|c| path.strip_prefix(&c).ok().map(|r| r.to_path_buf()))
        .unwrap_or_else(|| path.to_path_buf());
    s.to_string_lossy().replace('\\', "/")
}

fn lang_id(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "rs" => "rust",
        "py" => "python",
        "ts" => "typescript",
        "tsx" => "typescriptreact",
        "js" => "javascript",
        "jsx" => "javascriptreact",
        "go" => "go",
        "c" => "c",
        "cc" | "cpp" | "cxx" | "h" | "hpp" => "cpp",
        "java" => "java",
        _ => "plaintext",
    }
}

fn path_to_uri(p: &Path) -> String {
    let abs = p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    let s = abs.to_string_lossy().replace('\\', "/");
    #[cfg(windows)]
    let s = format!("/{s}"); // file:///C:/...
    format!("file://{}", percent_encode(&s))
}

/// Percent-encode a path for a `file://` URI, leaving `/` and the drive-letter
/// `:` intact. Mirrors `percent_decode` so a path with spaces round-trips.
fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        let keep = b.is_ascii_alphanumeric() || matches!(b, b'/' | b':' | b'-' | b'.' | b'_' | b'~');
        if keep {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn uri_to_path(uri: &str) -> PathBuf {
    let p = percent_decode(uri.strip_prefix("file://").unwrap_or(uri));
    #[cfg(windows)]
    let p = p.strip_prefix('/').map(str::to_string).unwrap_or(p);
    PathBuf::from(p)
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 3 <= b.len() {
            if let Ok(n) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(n);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn frame_round_trip() {
        let v = json!({"jsonrpc": "2.0", "id": 1, "method": "x"});
        let bytes = encode(&v);
        assert!(bytes.starts_with("Content-Length: "));
        let mut cur = Cursor::new(bytes.into_bytes());
        assert_eq!(read_frame(&mut cur).unwrap().unwrap(), v);
        assert!(read_frame(&mut cur).unwrap().is_none()); // EOF
    }

    #[test]
    fn formats_locations_as_path_line_snippet() {
        let p = std::env::temp_dir().join("tern_lsp_fmt_test.rs");
        std::fs::write(&p, "fn a() {}\nfn target() {}\n").unwrap();
        let loc = json!({"uri": path_to_uri(&p), "range": {"start": {"line": 1, "character": 3}}});
        let out = format_locations(&[loc], 50);
        assert!(out.contains(":2: fn target() {}"), "{out}");
        assert_eq!(format_locations(&[], 50), "no results");
    }

    #[test]
    #[cfg(unix)]
    fn decodes_percent_encoded_uris() {
        assert_eq!(percent_decode("a%20b"), "a b");
        assert_eq!(uri_to_path("file:///tmp/a%20b.rs"), PathBuf::from("/tmp/a b.rs"));
    }

    #[test]
    #[cfg(unix)]
    fn uri_round_trips_a_path_with_spaces() {
        let dir = std::env::temp_dir().join("tern lsp dir");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a b.rs");
        std::fs::write(&p, "fn x() {}\n").unwrap();
        let uri = path_to_uri(&p);
        assert!(uri.contains("%20"), "{uri}");
        assert_eq!(uri_to_path(&uri), p.canonicalize().unwrap());
    }
}
