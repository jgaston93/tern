//! Conversation history with a token budget.
//!
//! Two levers, applied in batches rather than every turn. Rewriting earlier
//! messages invalidates the provider's prompt cache (and llama.cpp's KV
//! cache), so trimming a little each turn costs more than it saves. Instead:
//!   1. Elide: at ELIDE_AT of budget, replace old tool results with stubs.
//!   2. Compact: at COMPACT_AT, summarize earlier turns into one message.

use crate::llm::{self, Config, Usage};
use crate::tools::{clip, Tools};
use serde_json::{json, Value};

const ELIDE_AT: f64 = 0.5;
const COMPACT_AT: f64 = 0.8;
const KEEP_RECENT_RESULTS: usize = 4;

struct Entry {
    msg: Value,
    label: String, // for tool results: "read src/main.rs"
    elided: bool,
}

pub struct Context {
    entries: Vec<Entry>,
    budget: usize,
    /// Real tokens per estimated token, learned from the API's usage numbers.
    calib: f64,
    totals: Usage,
    requests: u64,
}

impl Context {
    pub fn new(system: &str, budget: usize) -> Self {
        let mut c = Context { entries: vec![], budget, calib: 1.0, totals: Usage::default(), requests: 0 };
        c.push(json!({"role": "system", "content": system}));
        c
    }

    pub fn push(&mut self, msg: Value) {
        // A new user message means earlier requests are finished; their
        // reasoning (only present with keep_reasoning) is no longer useful.
        // This rewrites history once per request, not once per step.
        if msg["role"] == "user" {
            for e in &mut self.entries {
                if let Some(m) = e.msg.as_object_mut() {
                    m.remove("reasoning_content");
                }
            }
        }
        self.entries.push(Entry { msg, label: String::new(), elided: false });
    }

    /// Swap the system prompt and token budget without touching history, for a
    /// mid-session mode switch. This necessarily invalidates the prompt cache
    /// from the first message, but a mode change is an explicit, rare act.
    pub fn set_system(&mut self, system: &str, budget: usize) {
        self.budget = budget;
        match self.entries.first_mut() {
            Some(e) if e.msg["role"] == "system" => e.msg = json!({"role": "system", "content": system}),
            _ => self.entries.insert(0, Entry { msg: json!({"role": "system", "content": system}), label: String::new(), elided: false }),
        }
    }

    /// Append to the most recent tool result (used for check output).
    pub fn append_to_last_tool_result(&mut self, extra: &str) {
        if let Some(e) = self.entries.iter_mut().rev().find(|e| e.msg["role"] == "tool") {
            let cur = e.msg["content"].as_str().unwrap_or("").to_string();
            e.msg["content"] = json!(format!("{cur}\n{extra}"));
        }
    }

    pub fn totals(&self) -> (u64, Usage) {
        (self.requests, self.totals)
    }

    pub fn push_tool_result(&mut self, id: &str, label: String, content: String) {
        self.entries.push(Entry {
            msg: json!({"role": "tool", "tool_call_id": id, "content": content}),
            label,
            elided: false,
        });
    }

    pub fn messages(&self) -> Vec<Value> {
        self.entries.iter().map(|e| e.msg.clone()).collect()
    }

    /// Cheap estimate: ~4 chars per token, corrected by calibration.
    pub fn raw_estimate(&self) -> usize {
        self.entries.iter().map(|e| e.msg.to_string().len() / 4).sum()
    }

    pub fn estimate(&self) -> usize {
        (self.raw_estimate() as f64 * self.calib) as usize
    }

    pub fn record_usage(&mut self, u: Usage, sent_estimate: usize) {
        self.requests += 1;
        self.totals.prompt += u.prompt;
        self.totals.completion += u.completion;
        self.totals.cached += u.cached;
        if u.prompt > 0 && sent_estimate > 0 {
            // Includes tool schemas, so it slightly overestimates. That's the safe direction.
            self.calib = u.prompt as f64 / sent_estimate as f64;
        }
    }

    pub fn maintain(&mut self, cfg: &Config, tools: &mut Tools) {
        let budget = self.budget as f64;
        let limit = |f: f64| (budget * f) as usize;
        if self.estimate() > limit(ELIDE_AT) && self.elide(KEEP_RECENT_RESULTS) > 0 {
            tools.forget_reads();
        }
        if self.estimate() > limit(COMPACT_AT) {
            match self.compact(cfg) {
                Ok(()) => tools.forget_reads(),
                Err(e) => {
                    eprintln!("  [compaction failed: {e}; eliding aggressively]");
                    self.elide(1);
                    tools.forget_reads();
                }
            }
        }
    }

    /// Replace all but the most recent `keep` tool results with one-line stubs.
    /// The tool call itself stays, so the model still knows what it did.
    fn elide(&mut self, keep: usize) -> usize {
        let tool_idx: Vec<usize> =
            (0..self.entries.len()).filter(|&i| self.entries[i].msg["role"] == "tool").collect();
        let cut = tool_idx.len().saturating_sub(keep);
        let mut n = 0;
        for &i in &tool_idx[..cut] {
            let e = &mut self.entries[i];
            let len = e.msg["content"].as_str().map_or(0, str::len);
            if e.elided || len < 200 {
                continue;
            }
            e.msg["content"] = json!(format!("[elided: {} (~{} tokens). Re-run if needed.]", e.label, len / 4));
            e.elided = true;
            n += 1;
        }
        if n > 0 {
            eprintln!("  [elided {n} old tool results → ~{} tokens]", self.estimate());
        }
        n
    }

    /// Summarize everything before the current user request.
    fn compact(&mut self, cfg: &Config) -> Result<(), String> {
        let cur = self
            .entries
            .iter()
            .rposition(|e| e.msg["role"] == "user")
            .ok_or("no user message")?;
        if cur <= 1 {
            return Err("single turn exceeds budget".into());
        }

        let mut transcript = String::new();
        for e in &self.entries[1..cur] {
            let role = e.msg["role"].as_str().unwrap_or("?");
            let text = e.msg["content"].as_str().unwrap_or("");
            transcript.push_str(&format!("{role}: {}\n", clip(text, 600)));
            if let Some(calls) = e.msg["tool_calls"].as_array() {
                for c in calls {
                    transcript.push_str(&format!("  call {} {}\n", c["function"]["name"], clip(c["function"]["arguments"].as_str().unwrap_or(""), 150)));
                }
            }
        }

        let req = [
            json!({"role": "system", "content": "Summarize this coding session so work can continue without it. Include: goals, decisions, files changed and how, current state, open problems. Under 250 words. No preamble."}),
            json!({"role": "user", "content": transcript}),
        ];
        let summary = llm::chat(cfg, &req, None, None)?.message["content"].as_str().unwrap_or("").to_string();

        // Fold into the current user message rather than adding a second user
        // message, since some local chat templates require strict alternation.
        let current = self.entries[cur].msg["content"].as_str().unwrap_or("").to_string();
        self.entries[cur].msg["content"] =
            json!(format!("[Summary of earlier session]\n{summary}\n\n[Current request]\n{current}"));
        let before = self.estimate();
        self.entries.drain(1..cur);
        eprintln!("  [compacted: ~{before} → ~{} tokens]", self.estimate());
        Ok(())
    }

    pub fn print_stats(&self) {
        let t = &self.totals;
        let hit = if t.prompt > 0 { 100 * t.cached / t.prompt } else { 0 };
        println!(
            "requests {} | prompt {} ({}% cached) | completion {} | context ~{}/{}",
            self.requests, t.prompt, hit, t.completion, self.estimate(), self.budget
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_system_swaps_prompt_and_budget_keeping_history() {
        let mut c = Context::new("old system", 1000);
        c.push(json!({"role": "user", "content": "hello"}));
        c.set_system("new system", 2000);
        let msgs = c.messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["content"], "new system");
        assert_eq!(msgs[1]["content"], "hello");
        assert_eq!(c.budget, 2000);
    }
}
