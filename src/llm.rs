//! Minimal OpenAI-compatible chat client. Works with llama.cpp server,
//! Ollama, vLLM, LM Studio, OpenRouter, OpenAI, etc.

use crate::profile::Profile;
use serde_json::{json, Map, Value};
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
        let mut profile = crate::profile::select(&model, wanted.as_deref())?;
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
        Ok(Config {
            base_url: env("TERN_BASE_URL").unwrap_or_else(|| "http://localhost:8080/v1".into()),
            model,
            api_key: env("TERN_API_KEY"),
            max_steps: env("TERN_MAX_STEPS").and_then(|v| v.parse().ok()).unwrap_or(40),
            yolo: env("TERN_YOLO").is_some(),
            profile,
        })
    }

    /// A child config for a subagent: same endpoint, model, credentials and
    /// limits, but a different profile (so subagents can run their own sampling,
    /// tool set, and edit format). The model is shared — it comes from the
    /// environment, not the profile.
    pub fn for_subagent(&self, profile: Profile) -> Config {
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

pub fn chat(cfg: &Config, messages: &[Value], tools: Option<&Value>) -> Result<Response, String> {
    let p = &cfg.profile;
    let body = build_body(cfg, messages, tools);

    let url = format!("{}/chat/completions", cfg.base_url.trim_end_matches('/'));
    let mut req = ureq::post(&url).timeout(Duration::from_secs(600));
    if let Some(k) = &cfg.api_key {
        req = req.set("Authorization", &format!("Bearer {k}"));
    }

    let v: Value = req
        .send_json(body)
        .map_err(|e| match e {
            ureq::Error::Status(code, r) => format!("HTTP {code}: {}", r.into_string().unwrap_or_default()),
            e => e.to_string(),
        })?
        .into_json()
        .map_err(|e| e.to_string())?;

    let message = &v["choices"][0]["message"];
    if message.is_null() {
        return Err(format!("unexpected response: {v}"));
    }
    let u = &v["usage"];
    Ok(Response {
        message: clean(message, p.keep_reasoning),
        usage: Usage {
            prompt: u["prompt_tokens"].as_u64().unwrap_or(0),
            completion: u["completion_tokens"].as_u64().unwrap_or(0),
            cached: u["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0),
        },
    })
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
}
