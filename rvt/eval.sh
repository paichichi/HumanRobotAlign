#!/usr/bin/env bash
set -euo pipefail

cd /home/xli990/paichichi/GitHub/HumanRobotAlign/rvt

#   --eval-datafolder "/home/xli990/paichichi/GitHub/X-ICM/data/unseen_tasks/test" \
/home/xli990/bin/hralign_xvfb -a python eval.py \
  --model-folder "runs/UnadaptedR3M2RLBench" \
  --eval-datafolder "/home/xli990/paichichi/GitHub/HumanRobotAlign/rvt/runs/data/test" \
  --tasks "push_buttons" \
  --eval-episodes 1 \
  --episode-length 25 \
  --log-name "debug_original_push_buttons" \
  --device 0 \
  --headless \
  --model-name "model_4.pth"