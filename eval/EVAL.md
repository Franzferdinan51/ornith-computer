# Gate results — ornith-qwenlinux tune (run 20261031-235016-sft-2683)

Keeper rule (RUN-PLAN): battery clearly beats base AND voice gate clean.
**Verdict: NOT a keeper. Tune discarded, base stays.**

## Numbers

| arm | battery (12 tasks) | voice empty (of 12) | voice leak |
|-----|--------------------|---------------------|------------|
| base ornith-1.5-9b-mlx-vision (LM Studio) | 1 | 8 | 0 |
| tuned ornith-1.5-9b-qwenlinux-2683-4bit | 0 | 10 | 0 |

Training: 1,000/1,000 iters in 26:57, val loss 2.323 -> 0.000 (final
logged val 0.012 at iter 800, 0.000 at 1000 - the set was memorized),
peak mem 38.9 GB, exit clean. Adapters kept at
~/.mlx-lm-lora/runs/20261031-235016-sft-2683/adapters.

## How the gate ran

Host-side simulator of agent.sh (eval/battery.py): identical SYS,
prompt formats, truncation, guards, nudges and 9-round cap; TOOL run
executed in a per-task sandbox; ground truth checked on sandbox files,
never on the model's word. Temp 0, max_tokens 220, single user message
- the same shape qwenbackend uses. Model identity verified from every
transcript (base arm: every call reported ornith-1.5-9b-mlx-vision;
tuned arm: every call reported the tuned folder path).

Serving-stack note: the tuned folders were staged after LM Studio's
last launch, so the app never indexed them, and the app absorbed two
proper GUI quits without exiting (the known behavior - not forced).
The tuned arm therefore served from mlx_lm.server (same MLX runtime,
same weights, same chat template) on :1235; the base arm served from
LM Studio. Both stacks split the think channel out of `content`, and
the real harness reads `content` only - so both arms measure what the
seat would actually receive.

## What the transcripts show

- The tune DID learn the protocol vocabulary: when it answered in the
  content channel, its lines were exact ("TOOL run echo quasar >
  /tmp/star.txt", "TOOL remember the favorite color is teal") -
  cleaner than base, which mangled commands and hit shell syntax
  errors.
- But round 1 was very often empty in the content channel (the reply
  landed in the think channel or nowhere), and after a correct first
  step it looped the same call instead of advancing, and almost never
  finished with SAY:. mul/list3/greet/dbl/menu died on 3 consecutive
  empty replies; the rest hit the 9-round cap mid-loop.
- Base is also weak (1/12, 8/12 voice empties - the family disease is
  real), but the tune is worse on both measures, so it fails the
  keeper rule outright.

## Disposition

- Fused artifacts (bf16 master + 4-bit) moved OUT of the LM Studio
  library to /Volumes/Sabrent/lmstudio-archive/ - documented failure,
  not servable, nothing hard-deleted.
- Dashboard seat untouched: still qwen3.5-4b-mlx, verified answering
  after restore. (The task body said restore to the 3d-coder seat;
  the seat was flipped to plain Qwen 3.5 4B later on 2026-10-31 at
  Duckets' order, so "back as it was" = qwen3.5-4b-mlx.)
- Full transcripts: eval/base.json, eval/tuned.json.
