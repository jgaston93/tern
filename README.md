# tern

A minimal agentic coding CLI in Rust (~2,000 lines), built around spending as few tokens as possible per task and around making open-weight models reliable. Talks to any OpenAI-compatible endpoint: llama.cpp server, Ollama, vLLM, LM Studio, OpenRouter, OpenAI.

```sh
cargo build --release
TERN_BASE_URL=http://localhost:8080/v1 TERN_MODEL=qwen3-coder ./target/release/tern
./target/release/tern -p "fix the failing test in parser.rs"   # one request, then exit
```

| Env var | Default | |
|---|---|---|
| `TERN_BASE_URL` | `http://localhost:8080/v1` | Ollama: `http://localhost:11434/v1` |
| `TERN_MODEL` | `local` | also used to pick a profile |
| `TERN_API_KEY` | none | sent as Bearer token |
| `TERN_PROFILE` | matched on model name | same as `--profile` |
| `TERN_CTX`, `TERN_MAX_TOKENS`, `TERN_CHECK` | from profile | override the profile for one run |
| `TERN_MAX_STEPS` | `40` | tool-call iterations per request |
| `TERN_YOLO` | off | skip shell-command confirmation |

Commands: `/stats` (tokens used, cache hit rate, current context size), `/exit`. Options: `--profile NAME`, `--list-profiles`, `-p TASK`, `--stats-file PATH`.

## Model profiles

Open-weight models mostly fail on mechanics, not reasoning: malformed tool calls, find-and-replace edits that don't match, the wrong sampling settings. A profile tunes those per model. Built-ins live in `profiles.toml` (compiled in, and documented field by field). Add your own in `./tern.toml`, `~/.config/tern/profiles.toml`, or `$TERN_CONFIG`; yours take priority.

```toml
[[profile]]
name = "my-qwen"
match = ["qwen3-coder"]          # picked when the model name contains this
ctx = 65536                      # match the server's actual context size
temperature = 0.7
top_p = 0.8
top_k = 20
edit_format = "fuzzy"            # exact | fuzzy | whole
tools = []                       # empty = all; e.g. ["read", "write", "grep", "bash"]
prompt = "terse"                 # or "guided": adds an example to each tool description
check = "cargo check --message-format short"
```

What each knob does:

- **edit_format.** `exact` is cheapest and fine for strong models. `fuzzy` retries a failed match line by line, ignoring indentation and CRLF/LF, and re-indents the replacement to fit; it refuses if that matches more than one place. `whole` removes the edit tool so the model rewrites files, which costs output tokens but is what 7B–14B models get right.
- **lenient_parsing** (on by default). Recovers tool calls written as text: Qwen3-Coder's `<function=...>` XML, Hermes `<tool_call>{...}</tool_call>`, and fenced JSON naming a real tool. Repairs trailing commas, code fences, and double-encoded arguments. Each recovery saves a full round-trip.
- **check.** Runs after any step that changed files. A pass costs the model one line (`[check: passed]`); a failure includes the truncated output, so compiler errors arrive without the model having to ask.
- **keep_reasoning.** Keeps reasoning text across tool calls within a request (gpt-oss expects this), then drops it at the next user message.
- **Structured output** (`tool_choice`, `grammar`, `response_format`). Passed through to the server to constrain tool calls at the sampler, so fewer come back as text. `grammar` is llama.cpp GBNF; `response_format` is the vLLM guided / OpenAI object; `tool_choice` stays `"auto"` (`"required"` forces a call every turn and breaks the text-only finish). With a server that reliably emits well-formed calls you can then set `lenient_parsing = false`.
- **lsp.** Enables `def`/`refs` tools backed by a language server (`lsp = { rs = "rust-analyzer" }`, keyed by file extension). They resolve a symbol to `path:line: source` through the server — precise where grep is ambiguous (overloads, same-named methods) and fewer tokens than reading around. A missing or failed server degrades to a grep hint. Offered only when configured.
- **parallel_subagents.** Runs multiple `task` delegations from one step concurrently, each in its own context. Only speeds things up on a backend that decodes requests in parallel (vLLM, `llama.cpp --parallel`); a single llama.cpp slot serializes them. Subagents share the working tree, so give them non-overlapping work.
- **Sampling** fields are sent only when set, because hosted APIs reject some of them. `repeat_penalty` is sent under both the llama.cpp and vLLM names.

Server-side settings matter as much: run llama.cpp with `--jinja` so it uses the model's own chat template, and set Ollama's `num_ctx` explicitly (its default is small and it silently truncates).

## Measuring: evals/

Pick tasks representative of your work, then compare profiles and models on pass rate and **tokens per passing task**, not raw tokens. Cheap failures aren't savings.

```sh
TERN_MODEL=qwen3-coder REPEAT=3 evals/run.sh qwen3-coder small default
evals/summary.sh
```

Each task is a folder with `prompt.txt`, a starting `repo/`, and a `check.sh` that exits 0 when the work is correct. Two Python examples are included; C++ tasks with a `check.sh` that builds and runs tests work the same way. Failed runs keep their temp dir with the full log. On Windows, run the scripts from Git Bash (set `PYTHON=python` if needed).

## Where the tokens go, and what tern does about it

Every request resends the whole history, so a token added early is paid for on every later step. A 30-step task pays for the first file read 30 times.

**Fixed prefix (paid every request)**
- Tool schemas are one-line descriptions, ~400 tokens for all six.
- System prompt is short and static. Nothing in it changes between requests (no timestamps or directory listings), so providers and llama.cpp can reuse the cached prefix.

**Tool outputs (the bulk of growth)**
- `edit` returns a status line, never the file. Fails loudly on 0 or multiple matches, and hints when a match would succeed ignoring indentation, which saves a retry loop.
- `read` defaults to 200 numbered lines with paging, clips minified lines to 300 chars, and returns a one-line stub when the same range is read again unchanged.
- `grep` caps at 50 hits and tells the model how many it missed so it narrows the search instead of reading files.
- `bash` keeps the first 40 and last 60 lines (first compiler error and final summary), strips ANSI color codes.
- `.git`, `target`, `node_modules` etc. are skipped by search.

**History management**
- `reasoning_content` from thinking models is dropped and never resent.
- At 50% of budget, all but the last 4 tool results become stubs like `[elided: read src/main.rs (~1800 tokens). Re-run if needed.]`. The tool call stays in history, so the model knows what it did.
- At 80%, earlier turns are summarized by the model into ~250 words and folded into the current request.
- Both run in batches, not every turn. Every rewrite of old messages breaks the prompt cache, so trimming a little each step costs more than it saves.
- Token estimates are calibrated against the real `usage.prompt_tokens` the server reports.

## Next steps worth trying
1. **Streaming** so long responses don't look frozen.
2. **Save full shell output to a file** when truncating, and tell the model the path.
3. **Git-worktree isolation** for parallel subagents, so overlapping edits can't clobber each other.
4. **Plan/build profiles** — a read-only `plan` profile (no edit/write/bash) that produces an implementation plan, and a `build` profile that carries it out, so an orchestrator can plan first and delegate the build. Keeps planning cheap and building focused.
