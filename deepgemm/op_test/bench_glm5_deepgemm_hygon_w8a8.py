#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Hygon DeepGEMM I8 / W8A8 single-layer MoE benchmark.

Twin of `bench_glm5_deepgemm_fp8.py` (H20) but for the Hygon ("海光") variant
of DeepGEMM — package name is `deepgemm` (no underscore) and exposes int8 /
w8a8 entrypoints in addition to fp8.  Internals are Marlin-style ASM kernels
(`m_grouped_marlin_w8a8_gemm_nt_*`); the API is shape-compatible with H20's
`deep_gemm.m_grouped_fp8_gemm_nt_*`, only the dtypes differ.

Mapping to H20 deepgemm modes:

  H20 fp8 prefill   `deep_gemm.m_grouped_fp8_gemm_nt_contiguous`
    -> Hygon i8     `deepgemm.m_grouped_i8_gemm_nt_contiguous(
                       (act_i8, act_scale_fp32),
                       (w_i8, w_scale_fp32),
                       output_bf16, m_indices)`
    Per-token expert id `m_indices` (length M, int32, -1 for padding tail),
    each expert's tokens padded to multiples of 128.

  H20 fp8 decode    `deep_gemm.m_grouped_fp8_gemm_nt_masked`
    -> Hygon w8a8   `deepgemm.m_grouped_w8a8_gemm_nt_masked(
                       (act_i8, act_scale_fp32),
                       (w_i8, w_scale_fp32),
                       output_bf16, masked_m,
                       expected_m_per_group)`
    Per-expert layout `[E, M_padded, K]` with `masked_m[e]` = valid token rows
    for expert `e`. The last argument is the **padded** M cap (same as
    `M_padded` / buffer stride), not the average routed count. Between the two
    masked GEMMs, sglang uses ``lightop.fuse_silu_mul_quant_ep`` when available;
    this bench mirrors that (with a torch fallback). Optional
    ``m_grouped_w8a8_gemm_nt_masked_ll`` + w6 pack is enabled via
    ``--masked-low-latency`` when your DeepGEMM build provides it and (N,K,E)
    match its dispatch table.

Each timed iteration runs the FULL per-rank MoE layer (gemm1 + silu_and_mul +
re-quant + gemm2) but only the two GEMMs are timed (CUDA-event partial timing
matching `bench_glm5_three_kernels.py` and `bench_glm5_deepgemm_fp8.py`).

TP / EP semantics same as the fp8 version:
  --tp N : within-expert TP (slice moe_intermediate by N). E and topk unchanged.
  --ep N : expert parallel (each rank holds E/N experts). topk_local=1.
           M_local = ceil(M_global * original_topk / EP).

GLM5 default shape: hidden=6144, moe_intermediate=2048, experts=256, topk=8.

Notes specific to Hygon:
  - Activations follow sglang + lmslim ``per_token_quant_int8``: one fp32
    scale per token (tensor shape ``[..., 1]``), not K/128 block scales.
  - Weights stay Marlin int8 with per-output-channel fp32 scales ``[E, N, 1]``
    as in ``forward_groupgemm_w8a8_marlin_contiguous``.
  - Expert token padding still rounds to multiples of 128 for worst-case GEMM
    (same as before); this is layout padding, not activation group size.
  - `mode` defaults work; override via the kernel-level `config={'MODE': N}`
    if you want to force a specific ASM tiling.
"""
import argparse
import functools
import gc
import importlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# DeepGEMM availability check
# ---------------------------------------------------------------------------
try:
    deepgemm = importlib.import_module("deepgemm")
    _has_deepgemm = True
    _deepgemm_path = getattr(deepgemm, "__file__", "<unknown>")
except Exception as exc:  # noqa: BLE001
    deepgemm = None
    _has_deepgemm = False
    _deepgemm_import_err = repr(exc)


def _require_deepgemm():
    if not _has_deepgemm:
        raise RuntimeError(
            f"deepgemm (Hygon) not importable: {_deepgemm_import_err}. "
            "This bench needs the Hygon DeepGEMM wheel (package name `deepgemm`), "
            "not upstream `deep_gemm`. Run inside a Hygon container that ships it "
            "(e.g. sglang_daily_*_bch)."
        )


# ---------------------------------------------------------------------------
# Hygon marlin weight packers: prefer sglang (quiet import); else op_test helper.
# ---------------------------------------------------------------------------
def _marlin_packers_from_deepgemm_marlin_quant():
    path = Path(__file__).resolve().parent / "deepgemm_marlin_quant.py"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    spec = importlib.util.spec_from_file_location("_deepgemm_marlin_quant_bench", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"bad spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.weight8bit_nt_kpack2_marlin, mod.weight8bit_nt_kpack2_marlin1


def _get_marlin_packers():
    try:
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
                weight8bit_nt_kpack2_marlin,
                weight8bit_nt_kpack2_marlin1,
            )
        return weight8bit_nt_kpack2_marlin, weight8bit_nt_kpack2_marlin1
    except Exception as exc_sglang:
        try:
            return _marlin_packers_from_deepgemm_marlin_quant()
        except Exception as exc_local:
            raise RuntimeError(
                "Marlin INT8 packers unavailable: sglang import failed "
                f"({exc_sglang!r}); fallback deepgemm_marlin_quant.py failed ({exc_local!r})."
            ) from exc_local


# ---------------------------------------------------------------------------
# Pure-torch helpers (no triton dependency, all setup-time cost not timed)
# ---------------------------------------------------------------------------
def _per_token_quant_int8_torch(
    x: torch.Tensor,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token (per-row) INT8 dynamic quant on the last dim — matches sglang path.

    Aligns with ``lmslim.layers.gemm.int8_utils.per_token_quant_int8`` layout:
    one fp32 scale per row (full K), shapes ``scale == x.shape[:-1] + (1,)``.

    Memory-aware: amax in bf16, only the small scale tensor promoted to fp32.
    """
    s_bf16 = x.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    s_fp32 = s_bf16.to(torch.float32) / 127.0
    inv_s = (1.0 / s_fp32).to(x.dtype)
    y = (x * inv_s).round().clamp(-128, 127).to(torch.int8)
    return y.contiguous(), s_fp32.contiguous()


@functools.lru_cache(maxsize=2)
def _get_per_token_quant_int8_fn(prefer_torch: bool = False):
    """Return lmslim ``per_token_quant_int8`` when available (user bench style)."""
    if prefer_torch:
        return _per_token_quant_int8_torch
    try:
        from lmslim.layers.gemm.int8_utils import (  # noqa: PLC0415
            per_token_quant_int8 as ptq,
        )
    except Exception:
        return _per_token_quant_int8_torch
    return ptq


@functools.lru_cache(maxsize=1)
def _get_fuse_silu_mul_quant_ep():
    try:
        from lightop import fuse_silu_mul_quant_ep as fn  # noqa: PLC0415
    except Exception:
        return None
    return fn


def _try_import_masked_ll():
    try:
        from deepgemm.m_group_gemm import (  # noqa: PLC0415
            m_grouped_w8a8_gemm_nt_masked_ll,
            pack_int8_weight_enk_to_w6_low_latency,
        )
    except Exception:
        return None, None
    return m_grouped_w8a8_gemm_nt_masked_ll, pack_int8_weight_enk_to_w6_low_latency


def _make_masked_decode_weights(
    e: int,
    n1: int,
    hidden: int,
    moe_inter: int,
    device: torch.device,
    *,
    use_ll: bool,
):
    """Shared random INT8 bases -> Marlin packed (+ optional w6 for masked LL)."""
    _, pack_decode = _get_marlin_packers()

    w1_u = torch.randint(-128, 127, (e, n1, hidden), device=device, dtype=torch.int8)
    w2_u = torch.randint(-128, 127, (e, hidden, moe_inter), device=device, dtype=torch.int8)
    w1_scale = torch.rand((e, n1, 1), device=device, dtype=torch.float32) * 0.02 + 0.001
    w2_scale = torch.rand((e, hidden, 1), device=device, dtype=torch.float32) * 0.02 + 0.001

    w1_m = pack_decode(w1_u).contiguous()
    w2_m = pack_decode(w2_u).contiguous()

    w1_ll = w2_ll = None
    if use_ll:
        _, pack_w6 = _try_import_masked_ll()
        if pack_w6 is None:
            raise RuntimeError(
                "--masked-low-latency: cannot import "
                "deepgemm.m_group_gemm.pack_int8_weight_enk_to_w6_low_latency."
            )
        w1_ll = pack_w6(w1_u).contiguous()
        w2_ll = pack_w6(w2_u).contiguous()

    del w1_u, w2_u
    return w1_m, w1_scale, w2_m, w2_scale, w1_ll, w2_ll


def _make_marlin_int8_weight(
    e: int, n: int, k: int, device: torch.device, kind: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random per-expert INT8 weights, marlin-packed for Hygon deepgemm.

    Args:
        kind: "prefill" -> use weight8bit_nt_kpack2_marlin
              "decode"  -> use weight8bit_nt_kpack2_marlin1

    Returns:
        w_packed:  [E, N // 16, K * 16] int8  (marlin layout)
        scale:     [E, N, 1] float32        (per-out-channel)

    Packers come from sglang when installed, else ``op_test/deepgemm_marlin_quant.py``.
    Layout requires
    N divisible by k_tile=16 and K divisible by n_tile=16. GLM5 (N=4096/6144,
    K=6144/2048) is fine. After packing the row count is N // 16 and column
    count is K * 16; the numel stays the same.

    Activation/weight scale convention matches what sglang feeds:
      a_scale = per-token on K (one fp32 per row, shape [..., 1])
      w_scale = per-output-channel on N (this function returns [E, N, 1] fp32)
    """
    pack_prefill, pack_decode = _get_marlin_packers()
    packer = pack_prefill if kind == "prefill" else pack_decode

    w_unpacked = torch.randint(-128, 127, (e, n, k), device=device, dtype=torch.int8)
    w_packed = packer(w_unpacked).contiguous()
    assert w_packed.shape == (e, n // 16, k * 16), (
        f"unexpected packed shape {tuple(w_packed.shape)}, expected (E={e}, N//16={n//16}, K*16={k*16})"
    )
    del w_unpacked

    # Scale layout matches sglang's W8A8FP8 MoE create_weights: [E, N, 1] fp32
    # (per-channel scale, with a trailing singleton dim that the kernel reads).
    scale = torch.rand((e, n, 1), device=device, dtype=torch.float32) * 0.02 + 0.001
    return w_packed, scale.contiguous()


# ---------------------------------------------------------------------------
# Per-shape context builders
# ---------------------------------------------------------------------------
def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(a) * b


def _build_contiguous_ctx(shape, m_local: int, seed: int, pad_block_size: int = 128):
    """Mimic sglang's `forward_groupgemm_w8a8_*_contiguous` input layout under
    uniform routing: total routing entries = M_local * topk_local, distributed
    evenly across E_local experts, each expert padded to multiples of pad_block_size.

    `m_indices` (the 4th arg to `m_grouped_i8_gemm_nt_contiguous`) is a 1D INT32
    tensor of length M (= n_tokens_padded), each entry = the expert id of that
    token. (-1 marks padding; we use the per-expert label for the whole padded
    region to time worst-case GEMM cost, matching how sglang pads in practice.)
    """
    device = torch.device("cuda")
    e = shape["n_routed_experts"]
    topk = shape["num_experts_per_tok"]
    hidden = shape["hidden_size"]
    moe_inter = shape["moe_intermediate_size"]
    n1 = moe_inter * 2

    torch.manual_seed(seed)

    total_routed = m_local * topk
    per_expert = (total_routed + e - 1) // e
    per_expert_padded = ((per_expert + pad_block_size - 1) // pad_block_size) * pad_block_size
    n_tokens_padded = e * per_expert_padded

    m_indices = torch.repeat_interleave(
        torch.arange(e, device=device, dtype=torch.int32), per_expert_padded
    )
    assert m_indices.numel() == n_tokens_padded

    # BF16 activations -> per-token int8 + scale (same contract as DeepEP dispatch).
    x_bf16 = torch.randn((n_tokens_padded, hidden), device=device, dtype=torch.bfloat16) / 10.0
    a1_i8, a1_scale = _per_token_quant_int8_torch(x_bf16)
    del x_bf16

    # Marlin-packed weights for the prefill (contiguous) variant.
    w1, w1_scale = _make_marlin_int8_weight(e, n1, hidden, device, kind="prefill")
    w2, w2_scale = _make_marlin_int8_weight(e, hidden, moe_inter, device, kind="prefill")

    return {
        "device": device,
        "n_tokens_padded": n_tokens_padded,
        "per_expert_padded": per_expert_padded,
        "m_indices": m_indices,
        "a1_i8": a1_i8,
        "a1_scale": a1_scale,
        "w1": w1,
        "w1_scale": w1_scale,
        "w2": w2,
        "w2_scale": w2_scale,
        "hidden": hidden,
        "moe_inter": moe_inter,
        "n1": n1,
    }


def _build_masked_ctx(
    shape,
    m_local: int,
    seed: int,
    *,
    masked_vary_m: bool = False,
    masked_act_torch_only: bool = False,
    masked_silu_torch_only: bool = False,
    masked_low_latency: bool = False,
):
    """Mimic sglang's `forward_groupgemm_w8a8_masked` input layout.

    ``masked_m[e]`` is the valid token count for expert ``e`` (same name as user
    bench / ``m_grouped_w8a8_gemm_nt_masked``). ``m_padded`` rounds
    ``max_e masked_m[e]`` up to a multiple of 16 (>=16) and is the tensor layout
    M dimension **and** the kernel's ``expected_m_per_group`` argument.

    Activations default to ``lmslim.per_token_quant_int8`` when importable.
    """
    device = torch.device("cuda")
    e = shape["n_routed_experts"]
    topk = shape["num_experts_per_tok"]
    hidden = shape["hidden_size"]
    moe_inter = shape["moe_intermediate_size"]
    n1 = moe_inter * 2

    gemm_ll = None
    if masked_low_latency:
        gemm_ll, _ = _try_import_masked_ll()
        if gemm_ll is None:
            raise RuntimeError(
                "--masked-low-latency requires `deepgemm.m_group_gemm."
                "m_grouped_w8a8_gemm_nt_masked_ll` (not present in all wheels)."
            )
        if not (
            n1 in (3072, 4096, 6144, 7168)
            and hidden in (1536, 2048, 3072, 6144, 7168)
            and e in (1, 16, 32)
        ):
            raise ValueError(
                "masked-low-latency: (N1,K,E) must match the LL dispatch table in "
                "deepgemm; "
                f"got N1={n1}, K={hidden}, E={e}"
            )

    torch.manual_seed(seed)

    total_routed = m_local * topk
    expected_m = max(1, (total_routed + e - 1) // e)

    if masked_vary_m:
        g = torch.Generator()
        g.manual_seed(seed + 1337)
        masked_m = torch.empty((e,), device=device, dtype=torch.int32)
        for j in range(e):
            fac = 0.7 + 0.6 * torch.rand(1, generator=g).item()
            masked_m[j] = max(1, int(expected_m * fac))
    else:
        masked_m = torch.full((e,), expected_m, device=device, dtype=torch.int32)

    m_padded = int(masked_m.max().item())
    m_padded = ((max(m_padded, 16) + 15) // 16) * 16

    act_quant = _get_per_token_quant_int8_fn(prefer_torch=masked_act_torch_only)
    x_bf16 = torch.randn((e, m_padded, hidden), device=device, dtype=torch.bfloat16) / 10.0
    a1_i8, a1_scale = act_quant(x_bf16)
    del x_bf16

    w1_m, w1_scale, w2_m, w2_scale, w1_ll, w2_ll = _make_masked_decode_weights(
        e,
        n1,
        hidden,
        moe_inter,
        device,
        use_ll=masked_low_latency,
    )

    ctx: dict = {
        "device": device,
        "expected_m": expected_m,
        "m_padded": m_padded,
        "masked_m": masked_m.contiguous(),
        "a1_i8": a1_i8,
        "a1_scale": a1_scale,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "e": e,
        "hidden": hidden,
        "moe_inter": moe_inter,
        "n1": n1,
        "masked_act_torch_only": masked_act_torch_only,
        "masked_silu_torch_only": masked_silu_torch_only,
        "use_masked_ll": masked_low_latency,
    }

    if masked_low_latency:
        assert gemm_ll is not None and w1_ll is not None and w2_ll is not None
        ctx["w1"] = w1_ll
        ctx["w2"] = w2_ll
        ctx["w1_scale_squeezed"] = w1_scale.squeeze(-1).contiguous()
        ctx["w2_scale_squeezed"] = w2_scale.squeeze(-1).contiguous()
        ctx["a1_scale_squeezed"] = a1_scale.squeeze(-1).contiguous()
        ctx["_masked_ll_gemm_fn"] = gemm_ll
    else:
        ctx["w1"] = w1_m
        ctx["w2"] = w2_m

    return ctx

# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------
def _time_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return float(s.elapsed_time(e)) / iters


def _time_cuda_partial(fn_with_events, warmup: int, iters: int, n_segments: int) -> float:
    """Time only specific segments of `fn_with_events`.

    `fn_with_events(starts, ends)` runs ONE full pipeline iteration; it must
    `starts[i].record()` before each timed segment and `ends[i].record()` right
    after. Untimed work (e.g. silu/requant) just runs in between without any
    record calls. Returns the average per-iter SUM of all timed segments in ms.
    """
    for _ in range(warmup):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_segments)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_segments)]
        fn_with_events(starts, ends)
    torch.cuda.synchronize()

    all_starts = []
    all_ends = []
    for _ in range(iters):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_segments)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_segments)]
        fn_with_events(starts, ends)
        all_starts.append(starts)
        all_ends.append(ends)
    torch.cuda.synchronize()

    total = 0.0
    for starts, ends in zip(all_starts, all_ends):
        for s, e in zip(starts, ends):
            total += s.elapsed_time(e)
    return total / iters


# ---------------------------------------------------------------------------
# Bench: contiguous (prefill)
# ---------------------------------------------------------------------------
def _bench_deepgemm_contiguous(ctx, warmup: int, iters: int) -> dict:
    _require_deepgemm()
    device = ctx["device"]
    n_tokens_padded = ctx["n_tokens_padded"]
    n1 = ctx["n1"]
    hidden = ctx["hidden"]
    moe_inter = ctx["moe_inter"]

    cache1 = torch.empty((n_tokens_padded, n1), device=device, dtype=torch.bfloat16)
    cache3 = torch.empty((n_tokens_padded, hidden), device=device, dtype=torch.bfloat16)
    m_indices = ctx["m_indices"].contiguous()

    def run_once(starts, ends):
        # ---- stage 1: gate+up combined GEMM (TIMED) ----
        starts[0].record()
        deepgemm.m_grouped_i8_gemm_nt_contiguous(
            (ctx["a1_i8"], ctx["a1_scale"]),
            (ctx["w1"], ctx["w1_scale"]),
            cache1,
            m_indices,
        )
        ends[0].record()
        # ---- silu_and_mul + per-token INT8 re-quant (UNTIMED) ----
        cache2_bf16 = _silu_and_mul(cache1)  # [n_tokens_padded, moe_inter]
        cache2_i8, cache2_scale = _per_token_quant_int8_torch(cache2_bf16)
        # ---- stage 2: down GEMM (TIMED) ----
        starts[1].record()
        deepgemm.m_grouped_i8_gemm_nt_contiguous(
            (cache2_i8, cache2_scale),
            (ctx["w2"], ctx["w2_scale"]),
            cache3,
            m_indices,
        )
        ends[1].record()

    avg_ms = _time_cuda_partial(run_once, warmup, iters, n_segments=2)
    return {
        "kernel": "deepgemm_i8_contiguous (prefill)",
        "quant": "int8 per-token act + per-ch w",
        "avg_ms": avg_ms,
        "n_tokens_padded": n_tokens_padded,
        "per_expert_padded": ctx["per_expert_padded"],
    }


# ---------------------------------------------------------------------------
# Bench: vllm Triton fused_moe (block-fp8 path) -- KEPT for shape-compat with
# the H20 fp8 bench, but DOES NOT correspond to an int8/w8a8 path on Hygon.
# vllm's `fused_moe` block-fp8 path expects fp8 weights, not int8. If you want
# to compare against a triton MoE on Hygon, write one against
# chitu's `chitu/moe/experts/triton_fused_experts.py:fused_moe_kernel_int8`
# instead. For now we leave the vllm-fp8 helper here but DISABLE it from the
# default `--kernels` list, so unrelated downstream code paths in main() don't
# break if you re-enable it.
# ---------------------------------------------------------------------------
def _build_triton_fused_moe_ctx(shape, m_local: int, seed: int):
    """Inputs for vllm's fused_moe block-fp8 path.

    fused_moe takes BF16 hidden_states and pre-quantized FP8 weights; it
    quantizes activations internally. Layout:
      hidden_states: [M, K=hidden] bf16
      w1: [E, N=2*moe_inter, K=hidden] fp8
      w2: [E, N=hidden,      K=moe_inter] fp8
      w*_scale: [E, N/128, K/128] float32
      topk_ids: [M, topk] int32
      topk_weights: [M, topk] float32
    """
    device = torch.device("cuda")
    e = shape["n_routed_experts"]
    topk = shape["num_experts_per_tok"]
    hidden = shape["hidden_size"]
    moe_inter = shape["moe_intermediate_size"]
    n1 = moe_inter * 2
    block_size = 128

    torch.manual_seed(seed)

    # NOTE: this helper is kept for shape compat with the H20 fp8 bench but is
    # not active in the default --kernels list. Raise eagerly if someone wires
    # it back -- there is no fp8 weight maker in this Hygon-only file.
    raise NotImplementedError(
        "vllm fused_moe block-fp8 path is not applicable to Hygon int8/w8a8. "
        "If you want a triton MoE bench, port chitu's "
        "fused_moe_kernel_int8 / fused_moe_kernel_block_fp8 instead."
    )

    # Random uniform routing
    scores = torch.randn((m_local, e), device=device, dtype=torch.float32).abs()
    topk_weights, topk_ids = torch.topk(scores, topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = topk_ids.to(torch.int32)

    return {
        "device": device,
        "hidden_states": hidden_states,
        "w1": w1,
        "w1_scale": w1_scale,
        "w2": w2,
        "w2_scale": w2_scale,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "block_shape": [block_size, block_size],
        "topk": topk,
    }


def _bench_vllm_triton_fp8(ctx, warmup: int, iters: int) -> dict:
    """Time vllm's fused_moe end-to-end (gemm1+silu+gemm2 are all inside).

    Returns avg_ms = total kernel time across the whole MoE layer (no way to
    split per-stage without re-implementing fused_moe). This is the
    "comparable" number to chitu's deployed triton path.
    """
    try:
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"vllm.model_executor.layers.fused_moe not importable: {exc!r}. "
            "Install vllm to enable the triton fused_moe bench."
        )

    def call():
        return fused_moe(
            hidden_states=ctx["hidden_states"],
            w1=ctx["w1"],
            w2=ctx["w2"],
            topk_weights=ctx["topk_weights"],
            topk_ids=ctx["topk_ids"],
            inplace=False,
            use_fp8_w8a8=True,
            w1_scale=ctx["w1_scale"],
            w2_scale=ctx["w2_scale"],
            block_shape=ctx["block_shape"],
        )

    avg_ms = _time_cuda(call, warmup, iters)
    return {
        "kernel": "vllm_fused_moe (triton, block-fp8)",
        "quant": "blockfp8 128x128",
        "avg_ms": avg_ms,
        "note": "whole-layer (gemm1+silu+gemm2 inside fused_moe)",
    }


# ---------------------------------------------------------------------------
# Bench: masked (decode)
# ---------------------------------------------------------------------------
def _bench_deepgemm_masked(ctx, warmup: int, iters: int) -> dict:
    _require_deepgemm()
    device = ctx["device"]
    e = ctx["e"]
    m_padded = ctx["m_padded"]
    n1 = ctx["n1"]
    hidden = ctx["hidden"]
    expected_m = ctx["expected_m"]
    masked_m = ctx["masked_m"]
    m_cap = m_padded
    moe_inter = ctx["moe_inter"]
    # use_ll = ctx.get("use_masked_ll", False)
    use_ll = False
    gemm_ll = ctx.get("_masked_ll_gemm_fn") if use_ll else None
    quant_mid = _get_per_token_quant_int8_fn(
        prefer_torch=ctx["masked_act_torch_only"],
    )
    fuse_ep = (
        None
        if ctx["masked_silu_torch_only"]
        else _get_fuse_silu_mul_quant_ep()
    )

    cache1 = torch.empty((e, m_padded, n1), device=device, dtype=torch.bfloat16)
    cache3 = torch.empty((e, m_padded, hidden), device=device, dtype=torch.bfloat16)

    def masked_gemm1():
        deepgemm.m_grouped_w8a8_gemm_nt_masked(
            (ctx["a1_i8"], ctx["a1_scale"]),
            (ctx["w1"], ctx["w1_scale"]),
            cache1,
            masked_m,
            m_cap,
        )

    def masked_gemm2(cache2_i8: torch.Tensor, cache2_scale: torch.Tensor):
        deepgemm.m_grouped_w8a8_gemm_nt_masked(
            (cache2_i8, cache2_scale),
            (ctx["w2"], ctx["w2_scale"]),
            cache3,
            masked_m,
            m_cap,
        )

    def run_once(starts, ends):
        # ---- stage 1 (TIMED) ----
        starts[0].record()
        masked_gemm1()
        ends[0].record()
        # ---- silu+mul + per-token INT8 re-quant (UNTIMED; sglang: fuse when available) ----
        if fuse_ep is not None:
            cache2_i8, cache2_scale = fuse_ep(
                cache1,
                tokens_per_expert=masked_m,
                expect_m=m_cap,
            )
        else:
            cache2_bf16 = _silu_and_mul(cache1)
            cache2_i8, cache2_scale = quant_mid(cache2_bf16)
        # ---- stage 2 (TIMED) ----
        starts[1].record()
        masked_gemm2(cache2_i8, cache2_scale)
        ends[1].record()

    try:
        avg_ms = _time_cuda_partial(run_once, warmup, iters, n_segments=2)
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)
        err += (
            " | masked GEMM context: stage1 N1={} K={} E={} m_cap={}; "
            "stage2 N={} K={}".format(
                n1, hidden, e, m_cap, hidden, moe_inter,
            )
        )
        if "not found" in err or "symbol" in err.lower():
            err += (
                " | Hygon `m_grouped_w8a8_gemm_nt_masked` ships only a fixed set of "
                "HIP symbols; if stderr shows "
                "`hipModuleGetFunction`/`named symbol not found`, this wheel omits "
                "that variant (independent of i8 contiguous). Try another deepgemm "
                "build or adjust TP/shape only if your vendor documents supported "
                "(N,K) for masked Marlin."
            )
        return {
            "kernel": "deepgemm_w8a8_masked (decode)",
            "quant": "int8 per-token act + per-ch w",
            "status": "error",
            "error": err,
            "expected_m": expected_m,
            "m_padded": m_padded,
            "masked_total": int(masked_m.sum().item()),
        }

    mid_note = (
        "fuse_silu_mul_quant_ep"
        if fuse_ep is not None
        else "torch silu+requant"
    )
    act_note = "torch" if ctx["masked_act_torch_only"] else "lmslim(default)"
    note = f"mid={mid_note}"
    if use_ll:
        note += " | masked_ll"

    return {
        "kernel": "deepgemm_w8a8_masked (decode)",
        "quant": f"int8 per-token act ({act_note}) + per-ch w",
        "avg_ms": avg_ms,
        "expected_m": expected_m,
        "m_padded": m_padded,
        "masked_total": int(masked_m.sum().item()),
        "note": note,
    }


# ---------------------------------------------------------------------------
# TP / EP shape transforms (mirrors bench_glm5_three_kernels.py)
# ---------------------------------------------------------------------------
def _apply_tp(shape, tp: int):
    if tp == 1:
        return dict(shape)
    if shape["moe_intermediate_size"] % tp != 0:
        raise ValueError(
            f"moe_intermediate_size={shape['moe_intermediate_size']} not divisible by TP={tp}"
        )
    out = dict(shape)
    out["moe_intermediate_size"] = shape["moe_intermediate_size"] // tp
    return out


def _apply_ep(shape, ep: int):
    if ep == 1:
        return dict(shape)
    if shape["n_routed_experts"] % ep != 0:
        raise ValueError(
            f"n_routed_experts={shape['n_routed_experts']} not divisible by EP={ep}"
        )
    out = dict(shape)
    out["n_routed_experts"] = shape["n_routed_experts"] // ep
    out["num_experts_per_tok"] = 1
    return out


def _apply_ep_m(m_global: int, original_topk: int, ep: int) -> int:
    if ep == 1:
        return m_global
    return max(1, (m_global * original_topk + ep - 1) // ep)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
_HEADERS = ["bs", "kernel", "avg_ms", "tokens/s", "extra"]


def _format_row(row):
    return "{:>8}  {:<32}  {:>10}  {:>12}  {}".format(*row)


def _human_extra(result):
    parts = []
    if "quant" in result:
        parts.append(result["quant"])
    if "n_tokens_padded" in result:
        parts.append(
            f"n_tok_pad={result['n_tokens_padded']} "
            f"per_exp_pad={result['per_expert_padded']}"
        )
    if "expected_m" in result:
        extra = f"expected_m={result['expected_m']} m_pad={result['m_padded']}"
        if "masked_total" in result:
            extra += f" masked_sum={result['masked_total']}"
        parts.append(extra)
    if "note" in result:
        parts.append(result["note"])
    if result.get("status") == "oom":
        parts.append("OOM")
    if result.get("status") == "error":
        parts.append(f"ERROR: {result.get('error', '')[:100]}")
    return " | ".join(parts)


def _print_row(rows, bs_label, name, result):
    if "avg_ms" in result:
        avg = f"{result['avg_ms']:.3f} ms"
        # tokens/s uses M_local (per-rank input tokens). Same convention as
        # bench_glm5_three_kernels.py.
        m_for_throughput = result.get("m_local_for_throughput", bs_label)
        tps = f"{m_for_throughput / (result['avg_ms'] / 1000.0):.1f}"
    else:
        avg = "-"
        tps = "-"
    row = [bs_label, name, avg, tps, _human_extra(result)]
    print(_format_row(row), flush=True)
    rows.append(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _read_shape_from_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {
        "hidden_size": int(cfg["hidden_size"]),
        "moe_intermediate_size": int(cfg["moe_intermediate_size"]),
        "n_routed_experts": int(cfg["n_routed_experts"]),
        "num_experts_per_tok": int(cfg["num_experts_per_tok"]),
    }


_GLM5_DEFAULT_SHAPE = {
    "hidden_size": 6144,
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        help="Path to model config.json (reads hidden_size/moe_intermediate_size/"
             "n_routed_experts/num_experts_per_tok). If omitted, uses GLM5 defaults.",
    )
    parser.add_argument("--bs", default="1,16,32,128,32768",
                        help="Comma-separated GLOBAL batch sizes; supports k/m suffix.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument(
        "--kernels", default="contiguous,masked",
        help="Comma-separated subset of: "
             "contiguous (deepgemm i8 prefill), masked (deepgemm w8a8 decode). "
             "(`triton` slot is reserved for a vllm fused_moe path; left out by "
             "default because vllm fused_moe block-fp8 doesn't apply to int8/w8a8.)",
    )
    parser.add_argument(
        "--masked-vary-m",
        action="store_true",
        help="Decode (masked) path: per-expert masked_m ~ U(0.7,1.3)*expected_m "
             "(matches standalone w8a8 masked benches). Default: uniform masked_m.",
    )
    parser.add_argument(
        "--masked-act-torch",
        action="store_true",
        help="Decode path: force torch per-token INT8 quant for activations "
             "instead of lmslim.per_token_quant_int8.",
    )
    parser.add_argument(
        "--masked-silu-torch",
        action="store_true",
        help="Decode path: torch silu+mul + requant instead of "
             "lightop.fuse_silu_mul_quant_ep.",
    )
    parser.add_argument(
        "--masked-low-latency",
        action="store_true",
        help="Decode path: use deepgemm.m_group_gemm.m_grouped_w8a8_gemm_nt_masked_ll "
             "+ w6 weight pack when available; requires (N1,K,E) in the LL dispatch table "
             "(typically small E).",
    )
    parser.add_argument("--shape-override",
                        help="Override default shape, e.g. "
                             "'hidden=6144,moe_inter=2048,e=256,topk=8'.")
    args = parser.parse_args()

    if not _has_deepgemm:
        print(f"ERROR: deepgemm not importable: {_deepgemm_import_err}", flush=True)
        sys.exit(2)

    print(f"deepgemm path: {_deepgemm_path}", flush=True)
    print(
        f"torch={torch.__version__} | "
        f"device={torch.cuda.get_device_name(torch.cuda.current_device())} | "
        f"capability={torch.cuda.get_device_capability()}",
        flush=True,
    )

    if args.config:
        raw_shape = _read_shape_from_config(args.config)
    else:
        raw_shape = dict(_GLM5_DEFAULT_SHAPE)

    if args.shape_override:
        for kv in args.shape_override.split(","):
            k, v = kv.split("=")
            k = k.strip()
            v = int(v.strip())
            if k in ("hidden", "hidden_size"):
                raw_shape["hidden_size"] = v
            elif k in ("moe_inter", "moe_intermediate_size"):
                raw_shape["moe_intermediate_size"] = v
            elif k in ("e", "n_routed_experts"):
                raw_shape["n_routed_experts"] = v
            elif k in ("topk", "num_experts_per_tok"):
                raw_shape["num_experts_per_tok"] = v
            else:
                raise ValueError(f"unknown shape key: {k}")

    original_topk = raw_shape["num_experts_per_tok"]
    shape = _apply_tp(raw_shape, args.tp)
    shape = _apply_ep(shape, args.ep)

    print(
        "GLM5 single-layer MoE Hygon I8/W8A8 benchmark | "
        "hidden={} moe_inter={} experts={} topk={} | TP={} EP={}".format(
            raw_shape["hidden_size"],
            raw_shape["moe_intermediate_size"],
            raw_shape["n_routed_experts"],
            raw_shape["num_experts_per_tok"],
            args.tp,
            args.ep,
        ),
        flush=True,
    )
    if args.tp > 1 or args.ep > 1:
        print(
            "after TP/EP -> per-rank moe_inter={} E_local={} topk_local={} | "
            "gemm1: N={} K={} | gemm2: N={} K={}".format(
                shape["moe_intermediate_size"],
                shape["n_routed_experts"],
                shape["num_experts_per_tok"],
                shape["moe_intermediate_size"] * 2,
                shape["hidden_size"],
                shape["hidden_size"],
                shape["moe_intermediate_size"],
            ),
            flush=True,
        )
        if args.ep > 1:
            print(
                "EP routing model: M_local = ceil(M_global * original_topk={} / EP={})".format(
                    original_topk, args.ep,
                ),
                flush=True,
            )

    print(
        f"warmup={args.warmup} iters={args.iters} seed={args.seed} | "
        f"quantization: int8 activations per-token scale [...,1], weights per-out-channel [E,N,1]",
        flush=True,
    )
    print("", flush=True)
    print(_format_row(_HEADERS), flush=True)
    print("-" * 100, flush=True)

    selected = {k.strip() for k in args.kernels.split(",") if k.strip()}
    if "masked" in selected:
        _e_loc = shape["n_routed_experts"]
        _n1 = shape["moe_intermediate_size"] * 2
        _k1 = shape["hidden_size"]
        print(
            "提示 decode(`m_grouped_w8a8_gemm_nt_masked`): 本 rank 第一段 GEMM 使用 "
            f"N1={_n1}, K={_k1}, E={_e_loc}（与 Marlin 打包权重配套）。若 stderr 出现 "
            "`aiter_hip_common.h` + `named symbol not found`，说明当前 wheel 未编入该 "
            "(N,K,tiling/M) 下的 masked 设备函数；与 prefill 的 i8 contiguous 无关。"
            "可改用 `--kernels contiguous` 或换带齐对应内核的 DeepGEMM 构建。",
            flush=True,
        )
    batch_sizes = [
        int(x.strip().lower().replace("k", "000")) for x in args.bs.split(",") if x.strip()
    ]
    rows = []

    for m_global in batch_sizes:
        m_local = _apply_ep_m(m_global, original_topk, args.ep)
        if args.ep > 1 and m_local != m_global:
            print(
                f"[bs_global={m_global}] -> per-rank M_local={m_local} (EP={args.ep}, "
                f"original topk={original_topk})",
                flush=True,
            )
        bs_label = m_global

        # ---- contiguous (prefill) ----
        if "contiguous" in selected:
            ctx = None
            try:
                ctx = _build_contiguous_ctx(shape, m_local, args.seed)
                result = _bench_deepgemm_contiguous(ctx, args.warmup, args.iters)
                result["m_local_for_throughput"] = m_local
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                result = {"status": "oom", "error": str(exc).splitlines()[0]}
            except Exception as exc:  # noqa: BLE001
                result = {"status": "error", "error": repr(exc)}
            finally:
                if ctx is not None:
                    del ctx
                gc.collect()
                torch.cuda.empty_cache()
            _print_row(rows, bs_label, "deepgemm_i8_contiguous (prefill)", result)

        # ---- masked (decode) ----
        if "masked" in selected:
            ctx = None
            try:
                ctx = _build_masked_ctx(
                    shape,
                    m_local,
                    args.seed,
                    masked_vary_m=args.masked_vary_m,
                    masked_act_torch_only=args.masked_act_torch,
                    masked_silu_torch_only=args.masked_silu_torch,
                    masked_low_latency=args.masked_low_latency,
                )
                result = _bench_deepgemm_masked(ctx, args.warmup, args.iters)
                result["m_local_for_throughput"] = m_local
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                result = {"status": "oom", "error": str(exc).splitlines()[0]}
            except Exception as exc:  # noqa: BLE001
                result = {"status": "error", "error": repr(exc)}
            finally:
                if ctx is not None:
                    del ctx
                gc.collect()
                torch.cuda.empty_cache()
            _print_row(rows, bs_label, "deepgemm_w8a8_masked (decode)", result)

        # ---- vllm fused_moe (triton, block-fp8, autotuned) ----
        if "triton" in selected:
            ctx = None
            try:
                ctx = _build_triton_fused_moe_ctx(shape, m_local, args.seed)
                result = _bench_vllm_triton_fp8(ctx, args.warmup, args.iters)
                result["m_local_for_throughput"] = m_local
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                result = {"status": "oom", "error": str(exc).splitlines()[0]}
            except Exception as exc:  # noqa: BLE001
                result = {"status": "error", "error": repr(exc)}
            finally:
                if ctx is not None:
                    del ctx
                gc.collect()
                torch.cuda.empty_cache()
            _print_row(rows, bs_label, "vllm_fused_moe (triton)", result)

    print("", flush=True)
    print("Summary:", flush=True)
    print(_format_row(_HEADERS), flush=True)
    print("-" * 100, flush=True)
    for row in rows:
        print(_format_row(row), flush=True)


if __name__ == "__main__":
    main()

