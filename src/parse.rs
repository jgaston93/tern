//! Lenient handling of tool calls from open-weight models.
//!
//! Small models (and servers whose chat template doesn't match the model)
//! often write tool calls as plain text instead of structured `tool_calls`,
//! or produce almost-valid JSON. Recovering these is far cheaper than a
//! retry round-trip, which resends the whole context.

use regex::Regex;
use serde_json::{json, Value};

/// Parse tool-call arguments, repairing common mistakes when `lenient`.
pub fn parse_args(raw: &str, lenient: bool) -> Result<Value, String> {
    let err = match serde_json::from_str::<Value>(raw) {
        Ok(v) => return Ok(unwrap_double_encoded(v)),
        Err(e) => e.to_string(),
    };
    if !lenient {
        return Err(err);
    }
    let mut s = raw.trim().to_string();
    if s.is_empty() {
        return Ok(json!({}));
    }
    if let Some(rest) = s.strip_prefix("```") {
        // drop the ```json line and the closing fence
        let body = rest.split_once('\n').map_or("", |(_, b)| b);
        s = body.trim_end().trim_end_matches("```").to_string();
    }
    let trailing_comma = Regex::new(r",(\s*[}\]])").unwrap();
    s = trailing_comma.replace_all(&s, "$1").into_owned();
    serde_json::from_str::<Value>(&s).map(unwrap_double_encoded).map_err(|_| err)
}

/// Some models JSON-encode the arguments twice: "{\"path\": ...}".
fn unwrap_double_encoded(v: Value) -> Value {
    match v.as_str().and_then(|s| serde_json::from_str::<Value>(s).ok()) {
        Some(inner) if inner.is_object() => inner,
        _ => v,
    }
}

/// If the message has no structured tool calls but its text contains some,
/// move them into `tool_calls`. Returns how many were recovered.
pub fn recover_text_calls(msg: &mut Value, known: &[&str]) -> usize {
    if msg["tool_calls"].as_array().is_some_and(|c| !c.is_empty()) {
        return 0;
    }
    let content = msg["content"].as_str().unwrap_or("").to_string();
    let (calls, rest) = extract(&content, known);
    if calls.is_empty() {
        return 0;
    }
    let calls: Vec<Value> = calls
        .into_iter()
        .enumerate()
        .map(|(i, (name, args))| {
            json!({"id": format!("txt_{i}"), "type": "function",
                   "function": {"name": name, "arguments": args.to_string()}})
        })
        .collect();
    let n = calls.len();
    msg["content"] = json!(rest);
    msg["tool_calls"] = json!(calls);
    n
}

fn extract(text: &str, known: &[&str]) -> (Vec<(String, Value)>, String) {
    let mut calls = vec![];
    let mut rest = text.to_string();

    // Qwen3-Coder style: <function=read><parameter=path>src/a.rs</parameter></function>
    let func = Regex::new(r"(?s)<function=([\w.-]+)>(.*?)</function>").unwrap();
    let param = Regex::new(r"(?s)<parameter=([\w.-]+)>(.*?)</parameter>").unwrap();
    for c in func.captures_iter(text) {
        let mut args = serde_json::Map::new();
        for p in param.captures_iter(&c[2]) {
            args.insert(p[1].to_string(), coerce(&p[1], &p[2]));
        }
        calls.push((c[1].to_string(), Value::Object(args)));
    }
    rest = func.replace_all(&rest, "").into_owned();

    // Hermes style: <tool_call>{"name": ..., "arguments": {...}}</tool_call>
    let hermes = Regex::new(r"(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>").unwrap();
    for c in hermes.captures_iter(&rest.clone()) {
        if let Some(call) = as_call(&c[1], known) {
            calls.push(call);
        }
    }
    rest = hermes.replace_all(&rest, "").into_owned();

    // A fenced JSON block shaped like a call. Only accepted for known tool
    // names, so a model showing example JSON isn't mistaken for a call.
    if calls.is_empty() {
        let fenced = Regex::new(r"(?s)```(?:json)?\s*(\{.*?\})\s*```").unwrap();
        for c in fenced.captures_iter(&rest.clone()) {
            if let Some(call) = as_call(&c[1], known) {
                calls.push(call);
                rest = rest.replace(&c[0], "");
            }
        }
    }

    let rest = rest.replace("<tool_call>", "").replace("</tool_call>", "");
    (calls, rest.trim().to_string())
}

fn as_call(json_text: &str, known: &[&str]) -> Option<(String, Value)> {
    let v = parse_args(json_text, true).ok()?;
    let name = v["name"].as_str()?;
    if !known.contains(&name) {
        return None;
    }
    let args = v.get("arguments").or_else(|| v.get("parameters")).cloned().unwrap_or(json!({}));
    let args = match args {
        Value::String(s) => parse_args(&s, true).ok()?,
        other => other,
    };
    Some((name.to_string(), args))
}

/// XML-style parameters arrive as text. Integer/boolean fields are converted;
/// everything else stays a string (so "123" in file content isn't mangled).
fn coerce(key: &str, raw: &str) -> Value {
    let v = raw.strip_prefix('\n').unwrap_or(raw);
    let v = v.strip_suffix('\n').unwrap_or(v);
    match key {
        "offset" | "limit" => v.trim().parse::<u64>().map(Value::from).unwrap_or(json!(v)),
        "all" => json!(v.trim() == "true"),
        _ => json!(v),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const KNOWN: &[&str] = &["read", "edit", "write", "grep", "glob", "bash"];

    #[test]
    fn repairs_sloppy_json() {
        assert_eq!(parse_args("{\"path\": \"a\",}", true).unwrap()["path"], "a");
        assert_eq!(parse_args("```json\n{\"path\": \"a\"}\n```", true).unwrap()["path"], "a");
        assert_eq!(parse_args("\"{\\\"path\\\": \\\"a\\\"}\"", true).unwrap()["path"], "a");
        assert_eq!(parse_args("", true).unwrap(), json!({}));
        assert!(parse_args("{\"path\": \"a\",}", false).is_err());
    }

    #[test]
    fn recovers_hermes_calls() {
        let mut m = json!({"role": "assistant",
            "content": "Let me look.\n<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"x.rs\"}}\n</tool_call>"});
        assert_eq!(recover_text_calls(&mut m, KNOWN), 1);
        assert_eq!(m["content"], "Let me look.");
        let args: Value = serde_json::from_str(m["tool_calls"][0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["path"], "x.rs");
    }

    #[test]
    fn recovers_qwen_xml_calls_preserving_indentation() {
        let mut m = json!({"role": "assistant", "content":
            "<tool_call>\n<function=edit>\n<parameter=path>\na.py\n</parameter>\n<parameter=old>\n    x = 1\n</parameter>\n<parameter=new>\n    x = 2\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=read>\n<parameter=path>\nb.py\n</parameter>\n<parameter=limit>\n50\n</parameter>\n</function>\n</tool_call>"});
        assert_eq!(recover_text_calls(&mut m, KNOWN), 2);
        let a: Value = serde_json::from_str(m["tool_calls"][0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(a["old"], "    x = 1");
        let b: Value = serde_json::from_str(m["tool_calls"][1]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(b["limit"], 50);
        assert_eq!(m["content"], "");
    }

    #[test]
    fn ignores_example_json_that_is_not_a_call() {
        let mut m = json!({"role": "assistant", "content": "Config looks like:\n```json\n{\"name\": \"server\", \"port\": 80}\n```"});
        assert_eq!(recover_text_calls(&mut m, KNOWN), 0);
    }
}
