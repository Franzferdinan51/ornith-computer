# ornith-computer

A separate project from Qwenputer-Linux: tune Ornith 1.5 9B on the
Qwen-Linux system contract so it can sit in the system's builtin seat,
then — later — retarget the Qwenputer compiler approach (Ornith is
Qwen-architecture) to build a computer on Ornith's own weights, with
exactness re-proven against Ornith's weights. This repo never touches
the Qwen work.

## The one-off tune (2026-10-31 → 2026-11-01)

Ordered by Duckets on 2026-10-31: "We will tune Ornith overnight" —
executed while he slept. A single run, NOT the RSI program (which
stays cancelled): no cadence, no lineage.

- **Plan:** RUN-PLAN.md
- **Dataset:** 1,748 train / 120 valid rows, synthesized from the
  Qwen-Linux harness itself (agent.sh SYS + result/nudge formats,
  ask.sh one-line contract, the bare circuit agent's store/recall/mul
  protocol). ~55% agent protocol traces + repair patterns, ~11% bare
  circuit agent, ~10% direct QWEN: answers, ~24% plain-chat replay.
  No private data. Eval battery tasks held out and decontaminated.
  See dataset/MANIFEST.md.
- **Recipe:** LoRA r16, LR 7e-5, 1,000 iters, batch 2, completions-only
  loss, base = ornith-1.5-9b MLX vision-complete (4-bit) on Sabrent.
- **Run:** 20261031-235016-sft-2683

## Verdict — NOT a keeper (2026-11-01)

- Dataset: 1,748 train / 120 valid rows.
- Run: 20261031-235016-sft-2683 — 1,000/1,000 iters in 27 min, val
  loss 2.323 → 0.000 (the small set was memorized).
- Gate (12 held-out agent tasks + voice gate, temp 0, identical
  prompts, model identity verified from transcripts):
  - battery: tuned 0/12 vs base 1/12
  - voice: tuned 10/12 empty replies vs base 8/12, no protocol leak
    either arm
- The tune learned the protocol's exact lines but answers too often
  in the think channel (the harness reads content only), loops the
  same call instead of advancing, and almost never finishes with
  SAY:. Base is weak too — the tune is just worse. Full detail:
  eval/EVAL.md.
- Disposition: fused artifacts moved out of the LM Studio library to
  /Volumes/Sabrent/lmstudio-archive/ (documented failure, nothing
  hard-deleted); adapters kept in the run folder. The Qwen-Linux seat
  is unchanged (qwen3.5-4b-mlx, verified answering after restore).
