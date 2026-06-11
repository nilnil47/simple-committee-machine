#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

gemini -p "Hi have a look at program.md and let's kick off a new experiment! let's do the setup first." \
  --approval-mode=yolo \
  -y
