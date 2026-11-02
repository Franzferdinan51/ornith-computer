# qgpu in Ornith-Computer (vendored 2026-11-02)

This directory is [qwenGPU](https://github.com/Franzferdinan51/qwenGPU) (fork of
vixhal-baraiya/qwenGPU, MIT — see `LICENSE-qgpu` and `README-UPSTREAM.md`), vendored
into Ornith-Computer per the design in [`docs/qgpu-integration.md`](../../docs/qgpu-integration.md).

## What changed vs upstream

- **No upstream chipformer pin.** Upstream `pyproject.toml` (kept as
  `pyproject.upstream.toml` for reference) depends on `chipformer` from
  divagr18/qwenputer rev `f07a0b2`. Here `chipformer` resolves from this repo —
  the vendored compiler the RV code already uses. Verified 2026-11-02: the GPU
  circuit builds clean against it, and every `chipformer.q35` symbol qgpu calls
  (`ir`, `prims`, `emit`, `merge`, `mute`, `overlay`) matches.
- Nothing else in `qgpu/`, `kernels/`, `tests/`, `scripts/` was modified.

## Layout

`qgpu/runtime.py` and `qgpu/cuda.py` resolve the kernel library as
`<parent of the qgpu package>/kernels`, and `tests/test_kernels.py` as
`<parent of tests>/kernels` — so `qgpu/`, `kernels/` and `tests/` must stay
siblings, as they are here.

## Running

From the repo root, with the same dual-root path the RV code uses:

```bash
export PYTHONPATH="$PWD:$PWD/experimental/riscv:$PWD/experimental/gpu"
python -m pytest experimental/gpu/tests -x -q        # circuit/kernel/runtime tests
python -m qgpu.merge SRC DST                          # merge into a Qwen3.5-family checkpoint
```

Torch-level work (circuit exactness, merges) runs in the repo's `uv sync --extra hf`
environment, on the Mac.

## Phase 1 target

`python -m qgpu.merge <Ornith-1.5-9B HF checkpoint> <out>` — 44 GPU layers in front of
Ornith's 32, hidden widened from 4096, vocab +68,736. Gates: circuit tests exact at the
Ornith target (16 key / 32 value heads, intermediate 12288), `scripts/merged_check.py`
text KL vs base ≈ 7e-4, `qgpu.runtime` launches exact under `--check`.
