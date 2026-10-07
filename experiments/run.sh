#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_ROOT"
PYTHON="${PYTHON:-python}"

# The client reads OPENAI_API_KEY. Extra arguments override the defaults below.
"$PYTHON" -m experiments.client_agent \
    --method recent \
    --persona_learning_type distill \
    --k 5 \
    "$@"

"$PYTHON" -m experiments.client_agent \
    --method relevance \
    --persona_learning_type distill \
    --k 5 \
    "$@"

"$PYTHON" -m experiments.client_agent \
    --method personax \
    --persona_learning_type distill \
    --distance_threshold 0.7 \
    --alpha 1.06 \
    --ratio 0.6 \
    "$@"
