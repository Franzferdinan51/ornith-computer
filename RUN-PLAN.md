# Run plan — Ornith 1.5 9B one-off tune for the Qwen-Linux system

Ordered by Duckets 2026-10-31 (~16:55 ET, voice call): "We will tune
Ornith overnight." ONE specific run. **Not RSI** — no cadence, no
lineage, no program. The RSI loop stays cancelled. This run exists to
make ornith-1.5-9b the brain of the Qwen-Linux system we built, as
the step before a separate rebuild-on-Ornith project.

**Separation rule (Duckets, 2026-10-31): the Ornith build gets its
OWN separate GitHub repo** so it can never interfere with the Qwen
work. qwenputer-linux stays untouched. Repo creation is Duckets'
one action (the connector cannot create repos — same as the Doom
repo); everything is prepared locally until the repo exists.
**Repo created by Duckets 2026-10-31: Franzferdinan51/ornith-computer.**
Execution instruction (same day): this run happens **while Duckets
sleeps**; his awake focus stays on the Qwen work.

## Purpose

Teach ornith-1.5-9b the Qwen-Linux system contract so it can sit in
the builtin seat and (a) drive the guest agent harness reliably and
(b) answer QWEN: prompts directly — one tool call per reply, never
fabricate a RESULT, exact work always through the computer, use the
notebook as memory, recover from nudges with a concrete step.

## Dataset (build first, target ~1,500–3,000 rows — small on purpose)

Lesson applied from the Gemma small-set pivot (2026-10-30): a gentle
tune on a little clean data added acting without damaging voice;
the big mixes broke models.

1. **Agent protocol traces (majority).** System prompt = the exact
   SYS text `agent.sh` uses. Multi-round `TOOL run ...` → `RESULT:`
   → ... → `SAY:` sequences across task families: create/read/edit
   files, notebook store/recall at `/root/nb`, arithmetic only via
   `$(( ))` or the tools (never in-head), exploring the guest,
   building a small memory system (write script, run it, test it),
   recursive memory (write the plan to the notebook, read it back
   next round), recursive thought (plan → step → check → step).
2. **Repair patterns.** After a glued/simulated reply or an empty
   one, the corrected behavior: exactly one call, a concrete first
   step, read the tool error and adjust. Never invent a RESULT line.
3. **Direct QWEN: answers.** Short factual answers, one line flat
   (the /dev/qwen contract), plus referrals where exact work belongs
   to the computer.
4. **Plain-chat replay (~25–35%).** Ordinary conversation so the
   tune keeps the base model's voice.

Sources: synthesized from the harness's own sources (agent.sh SYS,
ask.sh contract, the bare agent's task patterns) + hand-built task
families. No private data. Decontaminated against the eval battery
below (battery tasks are held out, never trained on). Rendered in
the model's chat template; a sample render gets eyeballed before
any training starts.

## Recipe (gentle)

MLX-LM-LoRA through the usual driver (`rsi_mcp.start_training`),
base = the Sabrent ornith-1.5-9b MLX (vision-complete) the da7d fuse
used. LoRA r16, LR 5e-5 to 1e-4, iterations sized to the dataset
(~600–1,200), completions-only loss. **Never overlap fuse/convert
with the live run** (the 2026-10-30 jetsam lesson).

## Machine prep

- The grown circuit run finishes first — its verdict lands, THEN
  the Mac frees up. Do not kill it for this.
- Training-mode baseline per the training-mode skill (lifelines
  kept: Muse, LM Studio, Terminal, SSH/Tailscale).
- Dashboard/terminal services stay up; they are light.

## Eval (held out, built before training)

- 12-task battery in the agenttest/bridgetest families (file work,
  notebook memory, exact arithmetic via tool, multi-step build),
  run against the tuned model AND base ornith-1.5-9b with identical
  prompts; model identity verified from transcripts.
- Voice gate: plain chat, temp 0, 6 prompts ×2 — no regression vs
  base, and empty-reply rate counted (this family's known disease).
- Keeper rule: battery clearly beats base AND voice gate clean.
  Otherwise the tune is discarded and base stays in the seat.

## Deliverable

Fused + MLX-staged `ornith-1.5-9b-qwenlinux-<runhash>` in the LM
Studio library (servable — we only ship what LM Studio runs), seat
switched in dashboard Settings, morning verdict in plain words.

## Timeline (tonight, 2026-10-31 → 11-01)

Plan + dataset this evening; launch once the Mac is free; fuse +
eval after the run; verdict in the morning report.
