# Fuse + stage commands (run AFTER training completes - never overlap)

Run id: 20261031-235016-sft-2683, hash 2683
Adapters: ~/.mlx-lm-lora/runs/20261031-235016-sft-2683/adapters
Base: /Volumes/Sabrent/ornith-1.5-9b-mlx-vision

## 1. bf16 master (the named deliverable)
~/mlx-lm-lora/.venv/bin/python ~/mlx-lm-lora/tools/fuse_and_stage.py \
  --base /Volumes/Sabrent/ornith-1.5-9b-mlx-vision \
  --adapter ~/.mlx-lm-lora/runs/20261031-235016-sft-2683/adapters \
  --name ornith-1.5-9b-qwenlinux-2683

## 2. 4-bit serving build (gate + seat artifact, matched precision vs base)
sed -e 's#20261023-011815-sft-da7d#20261031-235016-sft-2683#' \
    -e 's#ornith-1.5-9b-duckbot-da7d-4bit#ornith-1.5-9b-qwenlinux-2683-4bit#' \
    ~/mlx-lm-lora/tools/fuse_and_stage_q4.py > /tmp/fuse_q4_2683.py
~/mlx-lm-lora/.venv/bin/python /tmp/fuse_q4_2683.py

## 3. Gate (~/ornith-eval/battery.py, on the Mac)
export PATH=$HOME/.lmstudio/bin:/opt/homebrew/bin:$PATH
lms load ornith-1.5-9b-mlx-vision --context-length 8192
python3 ~/ornith-eval/battery.py ornith-1.5-9b-mlx-vision base
lms unload ornith-1.5-9b-mlx-vision
lms load ornith-1.5-9b-qwenlinux-2683-4bit --context-length 8192
python3 ~/ornith-eval/battery.py ornith-1.5-9b-qwenlinux-2683-4bit tuned
lms unload ornith-1.5-9b-qwenlinux-2683-4bit

Note: if the app hasn't indexed the new folders, lms load fails -
then relaunch LM Studio app via GUI computer use only (never kill -9);
check the :1234 bind afterwards (must be *:1234).
