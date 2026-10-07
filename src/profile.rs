//! Per-model profiles: sampling, edit format, tool set, prompt style, and
//! parsing leniency. See profiles.toml for the built-ins and field docs.

use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

const BUILTIN: &str = include_str!("../profiles.toml");

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[serde(rename_all = "lowercase")]
pub enum EditFormat {
    /// Exact find-and-replace. Cheapest; strong models handle it well.
    #[default]
    Exact,
    /// Exact first, then retry ignoring indentation and line endings.
    Fuzzy,
    /// No edit tool; the model writes whole files. Costs more output tokens
    /// on big files, but small models get it right far more often.
    Whole,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[serde(rename_all = "lowercase")]
pub enum PromptStyle {
    #[default]
    Terse,
    Guided,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct Profile {
    pub name: String,
    #[serde(rename = "match")]
    pub matches: Vec<String>,
    pub ctx: usize,
    pub max_tokens: u32,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub min_p: Option<f64>,
    pub repeat_penalty: Option<f64>,
    pub edit_format: EditFormat,
    pub tools: Vec<String>,
    pub prompt: PromptStyle,
    pub prompt_extra: String,
    pub lenient_parsing: bool,
    pub keep_reasoning: bool,
    pub check: Option<String>,
    /// Kill bash/check commands after this many seconds (0 = no limit).
    pub bash_timeout: u64,
    /// Refuse to finish while `check` is failing (re-prompt instead).
    pub require_check_pass: bool,
    /// Profile names this agent may spawn via the `task` tool.
    /// Empty = no `task` tool, i.e. an ordinary single agent.
    pub subagents: Vec<String>,
    /// Constrain tool-call output at the server. Sent verbatim only when set;
    /// unsupported servers ignore what they don't know. `tool_choice`
    /// "required" forces a call every turn (breaks the text-only finish), so
    /// prefer "auto". `grammar` is llama.cpp GBNF; `response_format` is the
    /// vLLM guided / OpenAI structured-output object.
    pub tool_choice: Option<String>,
    pub grammar: Option<String>,
    pub response_format: Option<toml::Value>,
    /// Language-server commands by file extension, e.g. { rs = "rust-analyzer" }.
    /// Non-empty enables the `def`/`refs` tools.
    pub lsp: HashMap<String, String>,
    /// Run multiple `task` calls from one step concurrently. Only speeds things
    /// up on a backend that decodes requests in parallel (vLLM, llama.cpp
    /// --parallel); give subagents non-overlapping work (shared working tree).
    pub parallel_subagents: bool,
    /// Cap on how many parallel subagents run at once: a step emitting many
    /// `task` calls spawns them in batches of this size rather than all at once.
    /// 0 = no clamp. Ignored unless `parallel_subagents` is set.
    pub max_parallel: usize,
}

impl Default for Profile {
    fn default() -> Self {
        Profile {
            name: "default".into(),
            matches: vec![],
            ctx: 32_000,
            max_tokens: 4096,
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            repeat_penalty: None,
            edit_format: EditFormat::Exact,
            tools: vec![],
            prompt: PromptStyle::Terse,
            prompt_extra: String::new(),
            lenient_parsing: true,
            keep_reasoning: false,
            check: None,
            bash_timeout: 120,
            require_check_pass: false,
            subagents: vec![],
            tool_choice: None,
            grammar: None,
            response_format: None,
            lsp: HashMap::new(),
            parallel_subagents: false,
            max_parallel: 0,
        }
    }
}

impl Profile {
    pub fn tool_enabled(&self, name: &str) -> bool {
        if name == "edit" && self.edit_format == EditFormat::Whole {
            return false;
        }
        self.tools.is_empty() || self.tools.iter().any(|t| t == name)
    }
}

#[derive(Deserialize, Default)]
struct ProfileFile {
    #[serde(default)]
    profile: Vec<Profile>,
}

fn parse(src: &str, origin: &str) -> Result<Vec<Profile>, String> {
    toml::from_str::<ProfileFile>(src)
        .map(|f| f.profile)
        .map_err(|e| format!("{origin}: {e}"))
}

fn user_files() -> Vec<PathBuf> {
    let mut v = vec![];
    if let Ok(p) = std::env::var("TERN_CONFIG") {
        v.push(PathBuf::from(p));
    }
    v.push(PathBuf::from("tern.toml"));
    if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        v.push(PathBuf::from(home).join(".config/tern/profiles.toml"));
    }
    v
}

/// All profiles, user files first so they take priority over built-ins.
pub fn all() -> Result<Vec<Profile>, String> {
    let mut out = vec![];
    for path in user_files() {
        if let Ok(src) = std::fs::read_to_string(&path) {
            out.extend(parse(&src, &path.display().to_string())?);
        }
    }
    out.extend(parse(BUILTIN, "built-in profiles")?);
    Ok(out)
}

pub fn select(model: &str, wanted: Option<&str>) -> Result<Profile, String> {
    let profiles = all()?;
    if let Some(name) = wanted {
        return profiles.into_iter().find(|p| p.name == name).ok_or_else(|| {
            format!("no profile named '{name}'; see --list-profiles")
        });
    }
    let m = model.to_lowercase();
    let hit = profiles
        .iter()
        .find(|p| p.matches.iter().any(|s| m.contains(&s.to_lowercase())))
        .or_else(|| profiles.iter().find(|p| p.name == "default"));
    Ok(hit.cloned().unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_parse_and_match() {
        let p = select("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M", None).unwrap();
        assert_eq!(p.name, "qwen3-coder");
        assert_eq!(p.edit_format, EditFormat::Fuzzy);
        assert_eq!(p.top_k, Some(20));
        assert_eq!(select("some-unknown-model", None).unwrap().name, "default");
        let s = select("x", Some("small")).unwrap();
        assert!(!s.tool_enabled("edit") && s.tool_enabled("write") && !s.tool_enabled("glob"));
        assert!(select("x", Some("nope")).is_err());
        // New fields: defaults, and the orchestrator built-in enables subagents.
        assert_eq!(p.bash_timeout, 120);
        assert!(!p.require_check_pass && p.subagents.is_empty());
        let o = select("x", Some("orchestrator")).unwrap();
        assert_eq!(o.subagents, vec!["plan".to_string(), "build".to_string()]);
        assert!(p.tool_choice.is_none() && p.grammar.is_none() && p.response_format.is_none());
        // plan is read-only; build is a full-access executor that must pass its check.
        let plan = select("x", Some("plan")).unwrap();
        assert!(plan.tool_enabled("read") && plan.tool_enabled("grep") && plan.tool_enabled("glob"));
        assert!(!plan.tool_enabled("edit") && !plan.tool_enabled("write") && !plan.tool_enabled("bash"));
        let build = select("x", Some("build")).unwrap();
        assert!(build.require_check_pass && build.tool_enabled("edit") && build.tool_enabled("bash"));
    }

    #[test]
    fn parses_structured_output_knobs() {
        let src = "[[profile]]\nname='g'\ntool_choice='auto'\ngrammar='root ::= \"x\"'\n\
                   [profile.response_format]\ntype='json_object'";
        let p = &parse(src, "t").unwrap()[0];
        assert_eq!(p.tool_choice.as_deref(), Some("auto"));
        assert_eq!(p.grammar.as_deref(), Some("root ::= \"x\""));
        assert_eq!(p.response_format.as_ref().unwrap()["type"].as_str(), Some("json_object"));
    }

    #[test]
    fn max_parallel_defaults_off_and_parses() {
        assert_eq!(Profile::default().max_parallel, 0);
        let p = &parse("[[profile]]\nname='p'\nparallel_subagents=true\nmax_parallel=3", "t").unwrap()[0];
        assert_eq!(p.max_parallel, 3);
    }

    #[test]
    fn unknown_fields_are_rejected() {
        assert!(parse("[[profile]]\nname='x'\ntemprature=0.1", "t").is_err());
    }
}
