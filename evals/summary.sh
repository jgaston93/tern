#!/usr/bin/env bash
# Pass rate and average tokens per profile+model from results.csv.
# Tokens per *passing* task is the number to optimize: cheap failures aren't savings.
f=${1:-$(dirname "$0")/results.csv}
awk -F, 'NR > 1 {
  k = $2 " / " $3; runs[k]++; pass[k] += $5; tok[k] += $8; cached[k] += $9; out[k] += $10
} END {
  printf "%-36s %6s %10s %12s %10s %8s\n", "profile / model", "pass", "avg prompt", "tok per pass", "cached", "avg out"
  for (k in runs) {
    per = pass[k] ? int(tok[k] / pass[k]) : "-"
    hit = tok[k] ? int(100 * cached[k] / tok[k]) "%" : "-"
    printf "%-36s %3d/%-3d %10d %12s %10s %8d\n", k, pass[k], runs[k], tok[k] / runs[k], per, hit, out[k] / runs[k]
  }
}' "$f"
