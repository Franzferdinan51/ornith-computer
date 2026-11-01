# Dataset manifest — ornith-qwenlinux tune (2026-10-31)

Built by `../gen_dataset.py` (seed 20261031) from the Qwen-Linux harness
itself: `agent.sh` SYS text + result/nudge prompt formats (verbatim),
`ask.sh` one-line contract, the bare circuit agent's store/recall/mul
protocol (from circuit-builtin.log). No private data.

- train.jsonl: 1,748 rows — sha256 e504c0e01b157edcc51bad2eec8364217cb99a88d30528cf586ef00654f8beab
- valid.jsonl: 120 rows — sha256 89ac0e826e27f95d84929c041845fca169208c39873a781b8f3b2409e70f0bf6
- Copies: this folder (VM archive) + /Volumes/Sabrent/datasets/ornith-qwenlinux/2026-10-31/ (training copy, hash-verified after transfer)

Mix (train):
- Agent protocol traces w/ SYS (full sessions + singles): 954 (55%)
  file write/read, multi-line files, exact arithmetic via $(( )),
  remember/recall, schedule once/every + jobs + unschedule, notebook
  /root/nb, explore, error recovery, saveapp, saveskill + skill load,
  recursive memory (plan -> recall -> act), memory-system build,
  count/disk checks
- Repair rows (inside the 954): empty-reply nudges, task-echo nudge,
  repeat-call nudge, duplicate-schedule nudge, denylist-refusal
  recovery, SAY-must-copy-the-RESULT-value rows
- Bare circuit agent (store/recall/mul): 189 (11%)
- Direct QWEN: one-line answers + exact-work referrals: ~190
- Plain-chat replay + identity anchors: ~415 (24%)

Validation:
- All rows alternate user/assistant, start user, end assistant
- Arithmetic audit: every "a * b = c" in assistant text recomputed — 0 wrong
- Agent assistant turns: all exactly one line
- Render check: base tokenizer apply_chat_template eyeballed (ChatML,
  empty think scaffold as the template emits); token lengths max 2,593
  -> training max_seq_length set to 3072 (config change, dataset untouched)
- Decontamination: battery markers (eval/battery.py tasks: quasar/
  star.txt, teal, 847291x639284, list.txt, nb/ideas, dbl app, greet
  skill, banana files) asserted absent from every row
