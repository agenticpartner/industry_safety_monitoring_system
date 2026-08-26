#!/usr/bin/env bash
# On Blackwell (GB10 / DGX Spark and related), nvinfer must build a strongly-typed TensorRT
# network or detections come back empty. Harmless to skip on Ampere/Ada/Hopper/Orin.
#
# Called from docker/entrypoint.sh on every start and from scripts/build_engines.sh before
# trtexec, so the nvinfer YAML the pipeline reads matches the engine that was built.
set -euo pipefail
cd "$(dirname "$0")/.."

is_blackwell() {
  local cap name
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)"
  name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
  case "$cap" in
    12.*|13.*) return 0 ;;
  esac
  printf '%s\n' "$name" | grep -Eqi 'GB10|GB200|GB300|B200|B300|Blackwell|DGX Spark'
}

if ! is_blackwell; then
  exit 0
fi

echo "==> Blackwell GPU — ensuring strongly-typed: 1 in nvinfer configs"
python3 - <<'PY'
from pathlib import Path

for rel in ("configs/pgie_ppe.yml", "configs/pgie_fire.yml"):
    path = Path(rel)
    text = path.read_text()
    if any(line.strip().startswith("strongly-typed:") for line in text.splitlines()):
        print(f"    {rel} already has strongly-typed")
        continue
    lines = text.splitlines()
    out = []
    inserted = False
    for line in lines:
        out.append(line)
        if not inserted and line.startswith("  network-mode:"):
            out.append("  # Blackwell (GB10 / DGX Spark): TensorRT needs a strongly-typed network.")
            out.append("  strongly-typed: 1")
            inserted = True
    if not inserted:
        raise SystemExit(f"could not find network-mode in {rel}")
    path.write_text("\n".join(out) + "\n")
    print(f"    added strongly-typed: 1 to {rel}")
PY
