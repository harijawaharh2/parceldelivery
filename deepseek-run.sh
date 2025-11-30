#!/usr/bin/env bash
set -e

# deepseek-run.sh - helper to prepare and run DeepSeek
# Usage:
#   inside container: ./deepseek-run.sh prepare
#   or to run inference: ./deepseek-run.sh infer /path/to/image.jpg

DEEPSEEK_DIR=${DEEPSEEK_DIR:-/opt/deepseek}
HF_TOKEN=${HF_TOKEN:-}
DEEPSEEK_MODEL=${DEEPSEEK_MODEL:-""}  # optional HF model id if needed

if [ "$1" = "prepare" ]; then
  echo "[deepseek-run] prepare: deepseek dir = $DEEPSEEK_DIR"
  if [ -d "$DEEPSEEK_DIR" ]; then
    if [ -n "$HF_TOKEN" ] && [ -n "$DEEPSEEK_MODEL" ]; then
      echo "[deepseek-run] Attempting to download HF weights (if repo uses them)"
      python - <<PY
import os, sys
from huggingface_hub import hf_hub_download
m = os.getenv("DEEPSEEK_MODEL")
if m:
    print("Downloading model:", m)
    # This will error if model or file not found; customize filename per model card
    try:
        p = hf_hub_download(repo_id=m, filename="pytorch_model.bin", token=os.getenv("HF_TOKEN"))
        print("Downloaded model to", p)
    except Exception as e:
        print("Could not download automatically:", e)
PY
    fi
  else
    echo "[deepseek-run] No deepseek repo at $DEEPSEEK_DIR - clone or set DEEPSEEK_SCRIPT/DEEPSEEK_CMD"
  fi
  exit 0
fi

if [ "$1" = "infer" ]; then
  IMG="$2"
  if [ -z "$IMG" ]; then
    echo "Usage: $0 infer /path/to/image.jpg"
    exit 2
  fi
  # default try CLI (set DEEPSEEK_CMD env to actual binary)
  if [ -n "$DEEPSEEK_CMD" ]; then
    $DEEPSEEK_CMD --image "$IMG"
    exit $?
  fi
  # fallback: try python script in repo
  if [ -f "$DEEPSEEK_DIR/infer.py" ]; then
    python "$DEEPSEEK_DIR/infer.py" --image "$IMG"
    exit $?
  fi
  echo "No DEEPSEEK_CMD and no standard infer script found."
  exit 3
fi

echo "Usage: $0 prepare | infer /path/to/image.jpg"
exit 1

