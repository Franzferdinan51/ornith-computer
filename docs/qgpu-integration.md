# Working qwenGPU into Ornith-Computer — integration design

Status: **design + compatibility probe done 2026-11-02, no code moved yet.**
Source: https://github.com/Franzferdinan51/qwenGPU (fork of vixhal-baraiya/qwenGPU, MIT;
built on divagr18/qwenputer's transformer compiler, MIT — same provenance as this repo).

## What qwenGPU is

A SIMT GPU compiled into a stock Qwen3.5 checkpoint's weights: 45 transformer layers
(44 at Ornith's head geometry, see probe) in front of the model's own, none learned —
a compiler computes every weight. It runs real CUDA: the host compiles CUDA → PTX with
clang, lowers PTX to machine blocks (`qgpu/lower.py`), feeds program + inputs as tokens;
the model writes every step of the launch (`EXEC`/`VAL`/`MEM` records) and the host reads
results out of the DeltaNet recurrent state. Registers / program ROM / global memory live
in Gated DeltaNet state, exactly the trick this repo's RV32 CPU uses. Chat front-end has
two modes: *normal* (model writes an expression, ready-made kernels from `kernels/` run it)
and *advanced* (model writes the CUDA kernel itself). On Qwen3.5-2B: every launch exact
under `--check` replay, text KL vs base ≈ 7e-4.

## Why it fits here

- **Same compiler.** qgpu builds on `chipformer.q35` (`ir`, `prims`, `emit`, `merge`,
  `mute`, `overlay`, `dsl.graph`) — the same compiler this repo vendors and extends.
  qgpu pins upstream qwenputer rev `f07a0b2` as a git dependency; this repo's vendored
  chipformer is a descendant of that code.
- **Same architecture family.** Ornith 1.5 9B *is* Qwen3.5: hidden 4096, 32 layers
  (24 Gated DeltaNet + 8 full attention), `linear_num_key_heads` 16,
  `linear_num_value_heads` 32, intermediate 12288, vocab 248320. qgpu's merge is already
  parameterized over exactly these config fields — nothing in it is 2B-specific except
  defaults.
- **Same idea, second device.** Ornith-Computer today is a RISC-V CPU (host/Linux side
  proven; in-weights port still pending). A GPU is the natural second compiled device,
  and its math (exact integer kernels) covers what the CPU's bignum coprocessor only
  partly does.

## Compatibility probe (2026-11-02, this VM, numpy level)

Run with `PYTHONPATH = ornith-computer + ornith-computer/experimental/riscv + qwenGPU`:

- `qgpu.circuit.kernel.Kernel` builds **clean against this repo's vendored chipformer**
  (no upstream pin needed at graph level):
  - Qwen3.5-2B target (8/16 heads, inter 6144): 46 layers, 6,891 dims, CHIP_VOCAB 68,736
  - **Ornith 1.5 9B target (16/32 heads, inter 12288): 44 layers, same 6,891 dims / vocab**
  - Qwen3.5-4B target (16/32, inter 16384): 44 layers, same dims
- Every `chipformer` symbol qgpu imports at graph level exists here with the same shape
  (`Expr`, `Chan`, `Target`, `A_BIT/A_REF/REF_COLS/K`, `Circ`, `from_bits`, `overlay`).
- Torch-level signatures checked from source (this VM has no torch):
  `emit.allocate(g, clearable, keep, unit_cap, overlay=, overlay_cap=)`,
  `emit.Layout(hidden, off, m, slots, over)`, `merge.load/widen/pick_hidden/TAU_GAIN`,
  `mute.mute(sd, tc, n, dim, normed, prefix=, sinks=, first=)`, `tiny.code_gain` —
  all match qgpu's call sites. Note this repo's `q35.merge.load` already drops `mtp.*`
  tensors, which is what we want (see wrinkles).
- Not yet verified (needs torch + the Mac): `allocate` slot counts at Ornith dims,
  `emit_layers` exactness, the circuit test suite, overlay measurement. That is Phase 1's
  first gate, on the Mac where the qwenputer-linux uv env (`--extra hf`) already runs.

## The merge conflict (why this is phased, not a copy-paste)

Both merges do the same surgery — widen the residual, prepend circuit layers, append
chip tokens, mute the text layers on chip tokens, set final-norm gains for the code dims:

| | RV merge (this repo) | qgpu merge |
|---|---|---|
| Output layout | text-only `Qwen3_5ForCausalLM`, prefix `model.` ("rv1") | full multimodal checkpoint, prefix `model.language_model.` |
| Chip vocab | 230,324 RV tokens | 68,736 GPU tokens |
| Text rows carry | RV `Text` features (bytes, tool-call flags) | GPU `kernel.TEXT` features |
| Runtime | `RvChat` / liveterm / dashboard | `qgpu.runtime` + bridge/TUI |

A single checkpoint carrying **both** circuits must pick one layout, offset both token
families (combined vocab 248,320 + 230,324 + 68,736 → **547,392**), widen once for both
circuits' slots, mute text layers on both token sets, mute each circuit on the other's
tokens, and give text rows both feature sets. Doable — the compiler supports it in
principle — but it is a research step, not the first step.

## Phases

### Phase 0 — vendor (small, mechanical)
- Copy `qgpu/`, `kernels/`, qgpu's `tests/` into this repo under `experimental/gpu/`
  (mirroring `experimental/riscv/`), plus `docs/qgpu-cuda.md`; keep upstream README +
  LICENSE + credits (vixhal-baraiya, Qwenputer citation).
- Drop qgpu's git-pinned `chipformer` dependency; resolve `chipformer` from this repo
  (same dual-root PYTHONPATH pattern the RV code already uses).
- Gate: qgpu's numpy-level tests pass here; torch-level circuit tests pass on the Mac.

### Phase 1 — GPU-only merge into Ornith 1.5 9B (the headline artifact)
- `qgpu.merge` against the stock HF bf16 Ornith-1.5-9B checkpoint (safetensors — not the
  GGUF/MLX seat copy): 44 GPU layers + 32 Ornith layers, hidden widened from 4096 by
  `pick_hidden`, vocab +68,736, overlay on (re-measured at Ornith dims).
- Gates, same shape as qgpu's own: circuit tests exact at the Ornith target;
  `scripts/merged_check.py` KL vs base Ornith (target ≈ 7e-4 like the 2B);
  `runtime --check` launches replay-exact; `eval_chat` normal mode on the sampled set.
- Wrinkles to settle in this phase:
  - **Vision tower**: Ornith ships one; qgpu's merge already handles the merger bias
    (image tokens = text to the circuit) and this repo's `widen` sets
    `vision_config.out_hidden_size`. Keep the tower; verify with an image-token smoke.
  - **MTP head**: merged checkpoint drops it (both loads skip `mtp.*`). Accepted loss,
    recorded in `qgpu.json`; speculative decoding is a serving concern, not the circuit's.
  - **Serving**: the artifact is a transformers checkpoint driven by `qgpu.runtime`
    (like `RvChat`), *not* an LM Studio model — the LM Studio seat is untouched, same as
    the existing Qwenputer checkpoints. GGUF conversion is out of scope until the merged
    model is proven.
  - **Stage sets**: qgpu's `STAGES` map (mul/logic/atom stage placement) is keyed only by
    stage count; at 16/32 heads the circuit built 44 layers, but bf16 exactness of the
    multiply stages at 4096 hidden must be proven by the circuit tests, not assumed.

### Phase 2 — host integration (pick one, in order of value)
- **(a) Chat/bridge first**: qgpu's bridge + TUI pointed at the Ornith merged checkpoint,
  and a GPU panel/tab in this repo's dashboard next to Terminal/Chat — Ornith asks for a
  calculation, the in-weights GPU runs it, panel shows warps/registers live.
- **(b) Guest later**: a Linux-guest path to the GPU waits on the in-weights RV port
  (still pending for Linux generally) — do not couple the GPU work to it.

### Phase 3 — co-resident RV + GPU checkpoint (research)
- One widen, both circuits, combined vocab (547,392), dual mute, dual text features,
  final-norm gains for both code sets; drivers choreograph two token languages.
- Open questions: layer order (GPU-then-CPU vs CPU-then-GPU), overlay plans measured on a
  mixed CPU+GPU+text corpus, embedding cost of the combined vocab (~547k × widened
  hidden), whether the two circuits can share ballast/tau conventions or need separate
  markers. No timeline promised; Phase 1's artifact is useful standalone either way.

## Non-goals
- No changes to the Qwen seat, the LM Studio setup, or qwenputer-linux.
- No training: every GPU weight is compiler-computed, like the CPU's.
- Nothing pushed until Duckets approves the phase scope.
