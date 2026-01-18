#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <policy_zip|cache_dir|run_id>" >&2
  echo "Examples:" >&2
  echo "  $0 .opponent_cache/<RUN_ID>/unified_latest.zip" >&2
  echo "  $0 .opponent_cache/<RUN_ID>" >&2
  echo "  $0 <RUN_ID>" >&2
  exit 1
fi

input="$1"
zip_path=""

if [[ -d "$input" ]]; then
  zip_path="$(ls -t "$input"/*.zip 2>/dev/null | head -n 1 || true)"
elif [[ -d ".opponent_cache/$input" ]]; then
  zip_path="$(ls -t ".opponent_cache/$input"/*.zip 2>/dev/null | head -n 1 || true)"
else
  zip_path="$input"
  if [[ "$zip_path" != *.zip && -f "${zip_path}.zip" ]]; then
    zip_path="${zip_path}.zip"
  fi
fi

if [[ -z "$zip_path" || ! -f "$zip_path" ]]; then
  echo "Could not find .zip for input: $input" >&2
  exit 1
fi

python - "$zip_path" <<'PY'
import sys
from stable_baselines3.common.save_util import load_from_zip_file

path = sys.argv[1]
result = load_from_zip_file(path, device="cpu")
data = result[0]
print(data.get("policy_class"))
PY
