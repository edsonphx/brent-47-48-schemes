#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
workspace=$(mktemp -d)
trap 'rm -rf "$workspace"' EXIT

if [[ "${1-}" != "--collect" ]]; then
    python3 - <<'PY'
from pathlib import Path
from hashlib import sha256

for line in Path("SHA256SUMS.txt").read_text().splitlines():
    digest, filename = line.split("  ", 1)
    if sha256(Path(filename).read_bytes()).hexdigest() != digest:
        raise SystemExit("SHA-256 mismatch for " + filename)
PY
fi

checks=(hensel audit equivalence boundary formal_dps48 scan families curve_grid flip_walk dps48_support dps48_superposition binary_crosscheck dps48_local dps48_catalogue)
pids=()
for name in "${checks[@]}"; do
    python3 "${name}.py" > "$workspace/$name.jsonl" &
    pids+=("$!")
done
failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[index]}"; then
        echo "${checks[index]} verification failed" >&2
        failed=1
    fi
done
[[ "$failed" == 0 ]]
for name in "${checks[@]}"; do
    cat "$workspace/$name.jsonl"
done > "$workspace/replayed.jsonl"
if [[ "${1-}" != "--collect" ]]; then
    cmp results.jsonl "$workspace/replayed.jsonl"
fi
cat "$workspace/replayed.jsonl"
