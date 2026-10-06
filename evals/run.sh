#!/usr/bin/env bash
# Run every task under evals/tasks with each given profile and append results
# to evals/results.csv.
#
#   TERN_MODEL=qwen3-coder TERN_BASE_URL=http://localhost:8080/v1 evals/run.sh qwen3-coder small
#   REPEAT=3 evals/run.sh default      # models are nondeterministic; repeat for a real signal
#
# A task is a folder with:
#   prompt.txt   the request given to tern
#   repo/        starting files (copied to a temp dir for each run)
#   check.sh     exits 0 if the task was done correctly (run inside the temp dir)
set -u
here=$(cd "$(dirname "$0")" && pwd)
tern=${TERN_BIN:-$here/../target/release/tern}
out=${OUT:-$here/results.csv}
repeat=${REPEAT:-1}
[ $# -gt 0 ] || { echo "usage: $0 PROFILE [PROFILE...]"; exit 1; }
[ -x "$tern" ] || { echo "build first: cargo build --release"; exit 1; }
[ -f "$out" ] || echo "task,profile,model,outcome,passed,steps,requests,prompt_tokens,cached_tokens,completion_tokens,seconds" > "$out"

field() { sed -n "s/.*\"$1\":\"\{0,1\}\([^,\"}]*\).*/\1/p" "$2" 2>/dev/null; }

for profile in "$@"; do
  for task in "$here"/tasks/*/; do
    name=$(basename "$task")
    for i in $(seq "$repeat"); do
      work=$(mktemp -d)
      cp -R "$task/repo/." "$work/"
      stats="$work/.tern-stats.json"
      start=$(date +%s)
      (cd "$work" && TERN_YOLO=1 "$tern" --profile "$profile" -p "$(cat "$task/prompt.txt")" \
        --stats-file "$stats" > "$work/.tern-log.txt" 2>&1)
      secs=$(( $(date +%s) - start ))
      if (cd "$work" && bash "$task/check.sh" > "$work/.check-log.txt" 2>&1); then passed=1; else passed=0; fi

      echo "$name,$profile,$(field model "$stats"),$(field outcome "$stats"),$passed,$(field steps "$stats"),$(field requests "$stats"),$(field prompt_tokens "$stats"),$(field cached_tokens "$stats"),$(field completion_tokens "$stats"),$secs" >> "$out"
      if [ $passed = 1 ]; then
        echo "PASS $name [$profile] $(field prompt_tokens "$stats") prompt tokens, ${secs}s"
        rm -rf "$work"
      else
        echo "FAIL $name [$profile] ($(field outcome "$stats")) — kept $work for inspection (.tern-log.txt, .check-log.txt)"
      fi
    done
  done
done
echo; "$here/summary.sh" "$out"
