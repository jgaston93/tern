//! Minimal OpenAI-compatible chat client. Works with llama.cpp server,
//! Ollama, vLLM, LM Studio, OpenRouter, OpenAI, etc.

use crate::profile::Profile;
use serde_json::{json, Map, Value};
use std::io::BufRead;
use std::time::Duration;

pub struct Config {
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub max_steps: usize,
    /// Skip confirmation prompts for shell commands.
    pub yolo: bool,
    pub profile: Profile,
}

impl Config {
    /// Env vars override the profile, so a one-off `TERN_CTX=8000` works
    /// without editing config.
    pub fn new(profile_name: Option<String>) -> Result<Self, String> {
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        let model = env("TERN_MODEL").unwrap_or_else(|| "local".into());
        let wanted = profile_name.or_else(|| env("TERN_PROFILE"));
        let profile = Self::resolve_profile(&model, wanted.as_deref())?;
        Ok(Config {
            base_url: env("TERN_BASE_URL").unwrap_or_else(|| "http://localhost:8080/v1".into()),
            model,
            api_key: env("TERN_API_KEY"),
            max_steps: env("TERN_MAX_STEPS").and_then(|v| v.parse().ok()).unwrap_or(40),
            yolo: env("TERN_YOLO").is_some(),
            profile,
        })
    }

    /// Select a profile by name (or model match) and apply the env overrides
    /// that let a one-off `TERN_CTX=8000` work without editing config. Shared by
    /// startup and mid-session mode switches, so the overrides stick either way.
    fn resolve_profile(model: &str, wanted: Option<&str>) -> Result<Profile, String> {
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        let mut profile = crate::profile::select(model, wanted)?;
        if let Some(c) = env("TERN_CTX").and_then(|v| v.parse().ok()) {
            profile.ctx = c;
        }
        if let Some(m) = env("TERN_MAX_TOKENS").and_then(|v| v.parse().ok()) {
            profile.max_tokens = m;
        }
        if let Some(c) = env("TERN_CHECK") {
            profile.check = Some(c);
        }
        if let Some(t) = env("TERN_BASH_TIMEOUT").and_then(|v| v.parse().ok()) {
            profile.bash_timeout = t;
        }
        if env("TERN_REQUIRE_CHECK").is_some() {
            profile.require_check_pass = true;
        }
        Ok(profile)
    }

    /// Switch to a named profile mid-session, keeping endpoint, model, and
    /// credentials. An explicit switch wins over the model-name match.
    pub fn switch_profile(&mut self, name: &str) -> Result<(), String> {
        self.profile = Self::resolve_profile(&self.model, Some(name))?;
        Ok(())
    }

    /// A child config for a subagent: same endpoint, model, credentials and
    /// limits, but a different profile (so subagents can run their own sampling,
    /// tool set, and edit format). The model is shared — it comes from the
    /// environment, not the profile.
    pub fn for_subagent(&self, mut profile: Profile) -> Config {
        // Inherit the parent's check unless the child sets its own, so a
        // child's `require_check_pass` actually gates — `TERN_CHECK` lands on
        // the selected profile, and without this it would never reach a child.
        if profile.check.is_none() {
            profile.check = self.profile.check.clone();
        }
        Config {
            base_url: self.base_url.clone(),
            model: self.model.clone(),
            api_key: self.api_key.clone(),
            max_steps: self.max_steps,
            yolo: self.yolo,
            profile,
        }
    }
}

#[derive(Default, Clone, Copy)]
pub struct Usage {
    pub prompt: u64,
    pub completion: u64,
    pub cached: u64,
}

pub struct Response {
    pub message: Value,
    pub usage: Usage,
}

/// A streamed fragment, handed to the caller's sink as it arrives. Content goes
/// to stdout; reasoning is shown live but still dropped from history by default.
pub enum Delta<'a> {
    Content(&'a str),
    Reasoning(&'a str),
}

/// Build the request body. Pure (no I/O) so it's unit-testable: a field is
/// present only when the profile sets it, because hosted APIs reject unknown or
/// unsupported keys while local servers fall back to their defaults.
fn build_body(cfg: &Config, messages: &[Value], tools: Option<&Value>) -> Value {
    let p = &cfg.profile;
    let mut body = json!({
        "model": cfg.model,
        "messages": messages,
        "max_tokens": p.max_tokens,
    });
    if let Some(t) = tools.filter(|t| t.as_array().is_some_and(|a| !a.is_empty())) {
        body["tools"] = t.clone();
    }
    if p.stream {
        body["stream"] = json!(true);
        // Ask for a final usage chunk; servers that don't support it just omit it.
        body["stream_options"] = json!({ "include_usage": true });
    }
    let mut set = |k: &str, v: Option<Value>| {
        if let Some(v) = v {
            body[k] = v;
        }
    };
    set("temperature", p.temperature.map(Value::from));
    set("top_p", p.top_p.map(Value::from));
    set("top_k", p.top_k.map(Value::from));
    set("min_p", p.min_p.map(Value::from));
    // llama.cpp calls it repeat_penalty, vLLM repetition_penalty.
    set("repeat_penalty", p.repeat_penalty.map(Value::from));
    set("repetition_penalty", p.repeat_penalty.map(Value::from));
    // Structured-output constraints: force well-formed tool calls at the server
    // so fewer come back as text for parse.rs to recover.
    set("tool_choice", p.tool_choice.clone().map(Value::from));
    set("grammar", p.grammar.clone().map(Value::from));
    if let Some(rf) = &p.response_format {
        if let Ok(v) = serde_json::to_value(rf) {
            body["response_format"] = v;
        }
    }
    body
}

pub fn chat(
    cfg: &Config,
    messages: &[Value],
    tools: Option<&Value>,
    sink: Option<&mut dyn FnMut(Delta)>,
) -> Result<Response, String> {
    let p = &cfg.profile;
    let body = build_body(cfg, messages, tools);

    let url = format!("{}/chat/completions", cfg.base_url.trim_end_matches('/'));
    let mut req = ureq::post(&url).timeout(Duration::from_secs(600));
    if let Some(k) = &cfg.api_key {
        req = req.set("Authorization", &format!("Bearer {k}"));
    }

    let resp = req.send_json(body).map_err(|e| match e {
        ureq::Error::Status(code, r) => format!("HTTP {code}: {}", r.into_string().unwrap_or_default()),
        e => e.to_string(),
    })?;

    let (message, u) = if p.stream {
        assemble_stream(std::io::BufReader::new(resp.into_reader()), sink)?
    } else {
        let v: Value = resp.into_json().map_err(|e| e.to_string())?;
        let message = v["choices"][0]["message"].clone();
        if message.is_null() {
            return Err(format!("unexpected response: {v}"));
        }
        (message, usage_from(&v["usage"]))
    };

    Ok(Response {
        message: clean(&message, p.keep_reasoning),
        usage: u,
    })
}

fn usage_from(u: &Value) -> Usage {
    Usage {
        prompt: u["prompt_tokens"].as_u64().unwrap_or(0),
        completion: u["completion_tokens"].as_u64().unwrap_or(0),
        cached: u["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0),
    }
}

/// Parse an SSE stream from an OpenAI-compatible `/chat/completions` into the
/// same `message` shape the non-streaming path returns, emitting each fragment
/// to `sink` as it arrives. Pure except for the sink callback, so it's unit
/// testable over a canned byte string. Delta tool-call arguments arrive in
/// fragments and are reassembled by index.
fn assemble_stream(
    reader: impl BufRead,
    mut sink: Option<&mut dyn FnMut(Delta)>,
) -> Result<(Value, Usage), String> {
    let mut content = String::new();
    let mut reasoning = String::new();
    // Tool calls accumulated by their streamed `index`.
    let mut calls: Vec<Map<String, Value>> = Vec::new();
    let mut usage = Usage::default();

    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let data = match line.strip_prefix("data:") {
            Some(d) => d.trim(),
            None => continue, // blank lines, `:` comments, event: fields
        };
        if data == "[DONE]" {
            break;
        }
        let chunk: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => continue, // tolerate a malformed keep-alive chunk
        };
        // Some servers stream a mid-generation failure as an error object over a
        // 200 response; surface it rather than returning an empty message.
        if let Some(err) = chunk.get("error").filter(|e| !e.is_null()) {
            return Err(format!("stream error: {err}"));
        }
        if let Some(u) = chunk.get("usage").filter(|u| !u.is_null()) {
            usage = usage_from(u);
        }
        let delta = &chunk["choices"][0]["delta"];
        if let Some(s) = delta["content"].as_str().filter(|s| !s.is_empty()) {
            content.push_str(s);
            if let Some(f) = sink.as_mut() {
                f(Delta::Content(s));
            }
        }
        if let Some(s) = delta["reasoning_content"]
            .as_str()
            .or(delta["reasoning"].as_str())
            .filter(|s| !s.is_empty())
        {
            reasoning.push_str(s);
            if let Some(f) = sink.as_mut() {
                f(Delta::Reasoning(s));
            }
        }
        if let Some(tcs) = delta["tool_calls"].as_array() {
            for tc in tcs {
                let i = tc["index"].as_u64().unwrap_or(0) as usize;
                if i >= calls.len() {
                    calls.resize(i + 1, Map::new());
                }
                let slot = &mut calls[i];
                if let Some(id) = tc["id"].as_str().filter(|s| !s.is_empty()) {
                    slot.insert("id".into(), json!(id));
                }
                if let Some(name) = tc["function"]["name"].as_str().filter(|s| !s.is_empty()) {
                    let f = slot.entry("function").or_insert_with(|| json!({}));
                    f["name"] = json!(name);
                }
                if let Some(args) = tc["function"]["arguments"].as_str() {
                    let f = slot.entry("function").or_insert_with(|| json!({}));
                    let cur = f["arguments"].as_str().unwrap_or("");
                    f["arguments"] = json!(format!("{cur}{args}"));
                }
            }
        }
    }

    // A stream that delivered nothing usable (only keep-alives/`[DONE]`, or a
    // non-streaming error body) mirrors the null-message case the buffered path
    // rejects. A real turn always carries content, a tool call, or reasoning.
    if content.is_empty() && calls.is_empty() && reasoning.is_empty() {
        return Err("unexpected response: empty stream".into());
    }

    let mut message = Map::new();
    message.insert("role".into(), json!("assistant"));
    message.insert("content".into(), json!(content));
    if !reasoning.is_empty() {
        message.insert("reasoning_content".into(), json!(reasoning));
    }
    if !calls.is_empty() {
        message.insert("tool_calls".into(), Value::Array(calls.into_iter().map(Value::Object).collect()));
    }
    Ok((Value::Object(message), usage))
}

/// Keep only what must be sent back next request. Reasoning text can be
/// thousands of tokens; by default it never enters history. With
/// `keep_reasoning` it's kept until the request finishes (see Context::push).
fn clean(msg: &Value, keep_reasoning: bool) -> Value {
    let mut out = Map::new();
    out.insert("role".into(), json!("assistant"));
    out.insert("content".into(), json!(msg["content"].as_str().unwrap_or("")));
    if keep_reasoning {
        if let Some(r) = msg["reasoning_content"].as_str().or(msg["reasoning"].as_str()).filter(|r| !r.is_empty()) {
            out.insert("reasoning_content".into(), json!(r));
        }
    }
    if let Some(calls) = msg["tool_calls"].as_array().filter(|c| !c.is_empty()) {
        // Normalize: some local servers return arguments as an object, not a string.
        let calls: Vec<Value> = calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let args = &c["function"]["arguments"];
                let args = args.as_str().map(str::to_string).unwrap_or_else(|| args.to_string());
                json!({
                    "id": c["id"].as_str().filter(|s| !s.is_empty()).map(str::to_string).unwrap_or_else(|| format!("call_{i}")),
                    "type": "function",
                    "function": { "name": c["function"]["name"], "arguments": args }
                })
            })
            .collect();
        out.insert("tool_calls".into(), json!(calls));
    }
    Value::Object(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::Profile;

    fn cfg(profile: Profile) -> Config {
        Config { base_url: "x".into(), model: "m".into(), api_key: None, max_steps: 1, yolo: true, profile }
    }

    #[test]
    fn build_body_sends_only_what_is_set() {
        // Default profile: no sampling or constraint keys leak into the body.
        let b = build_body(&cfg(Profile::default()), &[], None);
        assert_eq!(b["model"], "m");
        for k in ["temperature", "tool_choice", "grammar", "response_format", "tools"] {
            assert!(b.get(k).is_none(), "{k} should be absent by default");
        }
        // Streaming is on by default and asks for a trailing usage chunk.
        assert_eq!(b["stream"], true);
        assert_eq!(b["stream_options"]["include_usage"], true);
        // ...and is fully absent when the profile opts out.
        let off = build_body(&cfg(Profile { stream: false, ..Profile::default() }), &[], None);
        assert!(off.get("stream").is_none() && off.get("stream_options").is_none());

        let p = Profile {
            temperature: Some(0.3),
            tool_choice: Some("auto".into()),
            grammar: Some("root ::= \"x\"".into()),
            response_format: Some(toml::from_str::<toml::Value>("type = \"json_object\"").unwrap()),
            ..Profile::default()
        };
        let b = build_body(&cfg(p), &[], None);
        assert_eq!(b["temperature"], 0.3);
        assert_eq!(b["tool_choice"], "auto");
        assert_eq!(b["grammar"], "root ::= \"x\"");
        assert_eq!(b["response_format"]["type"], "json_object");
    }

    #[test]
    fn assemble_stream_reassembles_content_args_and_usage() {
        // Content split across chunks; a tool call whose name and arguments
        // arrive in fragments; a final usage-only chunk; then [DONE].
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n",
            ": keep-alive comment\n",
            "\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"read\",\"arguments\":\"{\\\"path\\\":\"}}]}}]}\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"x.rs\\\"}\"}}]}}]}\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":5,\"prompt_tokens_details\":{\"cached_tokens\":7}}}\n",
            "data: [DONE]\n",
        );
        let mut seen = String::new();
        let mut sink = |d: Delta| {
            if let Delta::Content(s) = d {
                seen.push_str(s);
            }
        };
        let (msg, usage) = assemble_stream(sse.as_bytes(), Some(&mut sink)).unwrap();
        assert_eq!(seen, "Hello"); // streamed live, in order
        assert_eq!(msg["content"], "Hello");
        let call = &msg["tool_calls"][0];
        assert_eq!(call["id"], "c1");
        assert_eq!(call["function"]["name"], "read");
        assert_eq!(call["function"]["arguments"], "{\"path\":\"x.rs\"}");
        assert_eq!((usage.prompt, usage.completion, usage.cached), (11, 5, 7));

        // The assembled message passes through clean() to the same shape the
        // non-streaming path produces: args stay a string, role is assistant.
        let cleaned = clean(&msg, false);
        assert_eq!(cleaned["role"], "assistant");
        assert!(cleaned["tool_calls"][0]["function"]["arguments"].is_string());
    }

    #[test]
    fn assemble_stream_drops_reasoning_from_history_but_streams_it() {
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking\"}}]}\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n",
            "data: [DONE]\n",
        );
        let mut thoughts = String::new();
        let mut sink = |d: Delta| {
            if let Delta::Reasoning(s) = d {
                thoughts.push_str(s);
            }
        };
        let (msg, _) = assemble_stream(sse.as_bytes(), Some(&mut sink)).unwrap();
        assert_eq!(thoughts, "thinking"); // shown live
        assert_eq!(msg["reasoning_content"], "thinking");
        // Default keep_reasoning=false: clean() strips it from what's resent.
        assert!(clean(&msg, false).get("reasoning_content").is_none());
        assert_eq!(clean(&msg, true)["reasoning_content"], "thinking");
    }

    #[test]
    fn assemble_stream_surfaces_mid_stream_error() {
        // A server that 200s then streams a failure object must not look like an
        // empty (successful) turn.
        let sse = concat!(
            "data: {\"error\":{\"message\":\"boom\"}}\n",
            "data: [DONE]\n",
        );
        let err = match assemble_stream(sse.as_bytes(), None) {
            Err(e) => e,
            Ok(_) => panic!("expected a stream error, got a message"),
        };
        assert!(err.contains("boom"), "error text should carry the server message: {err}");
    }

    #[test]
    fn assemble_stream_rejects_an_empty_stream() {
        // Only keep-alives and [DONE]: nothing usable arrived, so it's an error
        // rather than a silent empty message.
        let sse = concat!(
            ": keep-alive\n",
            "\n",
            "data: [DONE]\n",
        );
        assert!(assemble_stream(sse.as_bytes(), None).is_err());
    }

    #[test]
    fn subagent_inherits_parent_check_unless_it_sets_its_own() {
        let parent = cfg(Profile { check: Some("cargo check".into()), ..Profile::default() });
        // Child with no check of its own picks up the parent's.
        let child = parent.for_subagent(Profile::default());
        assert_eq!(child.profile.check.as_deref(), Some("cargo check"));
        // A child that sets its own keeps it.
        let child = parent.for_subagent(Profile { check: Some("make test".into()), ..Profile::default() });
        assert_eq!(child.profile.check.as_deref(), Some("make test"));
    }
}
