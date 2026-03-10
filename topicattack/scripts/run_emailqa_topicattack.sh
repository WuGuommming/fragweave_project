#!/usr/bin/env bash
set -euo pipefail

python topicattack/run_emailqa_topicattack.py \
  --config topicattack/configs/emailqa_topicattack.yaml
