# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

tern is a minimal agentic coding CLI in Rust (~2,000 lines across `src/`, incl. inline tests). Its organizing goal is to **spend as few tokens as possible per task** and to **make open-weight models reliable**. It talks to any OpenAI-compatible `/chat/completions` endpoint (llama.cpp server, Ollama, vLLM, LM Studio, OpenRouter, OpenAI). Every design decision trades against the token budget — keep that lens when changing anything.

## Commands

```sh
cargo build --release
cargo test                              # unit tests live inline in each module (#[cfg(test)])
cargo test profile::                    # run one module's tests
cargo test builtins_parse_and_match     # run one test by name
cargo check --message-format short      # the same check the agent runs after edits

# run it
TERN_BASE_URL=http://localhost:8080/v1 TERN_MODEL=qwen3-coder ./target/release/tern
./target/release/tern -p "fix the failing test"   # one request, then exit (0 done, 2 step limit, 1 error)
./target/release/tern --list-profiles
```

Evals (compare profiles/models on pass rate and tokens-per-passing-task):
```sh
TERN_MODEL=qwen3-coder REPEAT=3 evals/run.sh qwen3-coder small default
evals/summary.sh
```
Each eval task is a folder under `evals/tasks/<name>/` with `prompt.txt`, a starting `repo/`, and `check.sh` (exits 0 when correct). `run.sh` copies `repo/` to a temp dir per run; failed runs keep their temp dir for inspection. Results append to `evals/results.csv`.

## Architecture

The agent is a straight loop; the cleverness is all in what gets sent and what gets kept. Data flows: `main` builds a static system prompt + tool schemas → `Context` holds history under a token budget → each step calls `llm::chat` → the response's tool calls run through `tools::Tools` → results go back into `Context`.

- **`main.rs`** — CLI/env parsing, the static `system_prompt`, and `run_turn` (the step loop). Key invariant: the system prompt and tool schemas contain **nothing that changes between requests** (no timestamps, no directory listings) so the whole prefix stays prompt-cacheable. After any step that changed files, the profile's check runs **once** and its output is appended to the last tool result (`append_to_last_tool_result`), so the model sees compiler errors without asking.

- **`context.rs`** — conversation history + token budgeting. Two levers, applied in **batches not every turn** (rewriting old messages breaks the prompt/KV cache, so trimming a little each turn costs more than it saves): at `ELIDE_AT` (50%) old tool results become one-line stubs but the tool *call* stays, so the model still knows what it did; at `COMPACT_AT` (80%) everything before the current user message is summarized by the model and folded *into* the current user message (one user message, because some local chat templates require strict role alternation). Token counts are a cheap ~4-chars/token estimate **calibrated** against the real `usage.prompt_tokens` the server returns.

- **`llm.rs`** — the OpenAI-compatible client plus `Config` (env → profile resolution). Sampling fields are sent **only when the profile sets them** (hosted APIs reject unknown/unsupported ones). `repeat_penalty` is sent under both the llama.cpp and vLLM names. `clean()` strips the response down to what must be resent: reasoning text is dropped by default (can be thousands of tokens) unless `keep_reasoning`; tool-call arguments are normalized to strings since some servers return objects.

- **`profile.rs`** — per-model `Profile` (sampling, `edit_format`, tool subset, prompt style, parsing leniency, check command). Built-ins are compiled in from `profiles.toml` (`include_str!`). User profiles from `$TERN_CONFIG`, `./tern.toml`, `~/.config/tern/profiles.toml` are loaded **first so they win**. Selection: explicit `--profile`/`TERN_PROFILE` name, else first profile whose `match` substring is in the model name, else `default`. `deny_unknown_fields` is deliberate — a typo'd profile key is an error, not silently ignored.

- **`parse.rs`** — leniency for open-weight models (gated by `lenient_parsing`). Recovers tool calls written as plain text (Qwen3-Coder `<function=…>` XML, Hermes `<tool_call>{…}`, fenced JSON naming a known tool) and repairs sloppy JSON (trailing commas, code fences, double-encoding). Every recovery here saves a full round-trip, which would resend the entire context. Fenced-JSON recovery only fires for **known tool names** so example JSON in a reply isn't mistaken for a call.

- **`lsp.rs`** — a minimal JSON-RPC-over-stdio LSP client (no extra crates) backing the optional `def`/`refs` tools. Started lazily per file extension from the profile's `lsp` map, kept for the session (killed on drop). `def`/`refs` resolve a symbol via `workspace/symbol` (+ `textDocument/references`) and return grep-style `path:line: source`; any failure degrades to a "use grep" hint so the agent never stalls. Offered only when `lsp` is configured, so the cacheable prefix is unchanged otherwise.

- **`tools.rs`** — the six core tools (`read`, `edit`, `write`, `grep`, `glob`, `bash`), each written to return the **fewest tokens that still let the model decide its next move**. `edit` returns a status line, never the file; on a failed exact match it hints that it would match ignoring indentation. Fuzzy edit (`fuzzy_replace`) matches line-by-line ignoring indentation/CRLF, re-indents the replacement, and refuses if it would match more than once. `whole` edit format removes the edit tool entirely (small models rewrite whole files more reliably). `read` pages at 200 lines, clips long lines, and returns `[unchanged]` for a re-read of the same range (the `seen` hash map, cleared by `forget_reads` whenever context is trimmed so the stub can't lie). `grep`/`glob` cap results and report how many were omitted. `bash` truncates the middle (keeps first error + final summary), strips ANSI. Every path (and `bash`/`check`/`lsp`) resolves against a per-`Tools` `root` — `.` normally, a sandbox copy for an isolated parallel subagent — while displayed paths stay relative, so the model never sees the sandbox prefix.

- **`sandbox.rs`** — filesystem isolation for parallel subagents (no git needed). Each parallel subagent gets a copy of the working tree (same skip rules as search) rooted via `Tools`, and after it joins only the files whose content hash changed are merged back; files two subagents in a batch both touched are withheld and both are told. Snapshot/diff per batch, so a later batch sees the earlier one's merged result.

## Conventions when editing

- **Token cost is the review criterion.** Before adding anything to the system prompt, tool schemas, or a tool's output, ask what it costs *per request* — the whole history is resent every step. Prefer making a tool's output smaller over adding prose to the prompt.
- Tests are inline `#[cfg(test)]` modules at the bottom of each file, not a separate `tests/` dir. Add tests next to the code they cover.
- Tool schema descriptions are kept terse by design; the `guided` prompt style adds examples. If you add a tool, register it in `ALL_TOOLS`, `schemas()`, and `Tools::run`, and respect `Profile::tool_enabled`.
- Keep the cacheable-prefix invariant: never put per-request-varying data into the system prompt or schemas.
