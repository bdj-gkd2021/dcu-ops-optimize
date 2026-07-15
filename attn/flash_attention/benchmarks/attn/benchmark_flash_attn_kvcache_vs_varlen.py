#!/usr/bin/env python3
import argparse
import inspect
import math
import time

import torch


def parse_args():
    """Parse benchmark shape, backend, dtype, and chunked-prefill options."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark flash_attn_with_kvcache and flash_attn_varlen_func on the "
            "same logical prefix+current attention case."
        )
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--q-len", type=int, default=12800)
    parser.add_argument("--prefix-kv-len", type=int, default=38400)
    parser.add_argument("--new-kv-len", type=int, default=None)
    parser.add_argument("--q-head", type=int, default=6)
    parser.add_argument("--kv-head", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("both", "kvcache", "varlen", "vllm"),
        default="both",
        help="Which FlashAttention API to benchmark.",
    )
    parser.add_argument(
        "--cache-layout",
        choices=("both", "contiguous", "paged"),
        default="both",
        help="Cache layout for flash_attn_with_kvcache.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=256,
        help="Paged KV cache block size. flash_attn_with_kvcache requires a multiple of 256.",
    )
    parser.add_argument(
        "--chunked-prefill-size",
        type=int,
        default=8192,
        help=(
            "Split the current prefill tokens into chunks of this size for "
            "kvcache benchmarks. Use 0 to disable chunking."
        ),
    )
    parser.add_argument(
        "--vllm-page-size",
        type=int,
        default=64,
        help="vLLM paged KV cache block size. The local vLLM FlashAttention exports use 64-token blocks.",
    )
    parser.add_argument(
        "--sgl-page-size",
        type=int,
        default=64,
        help="SGLang paged KV cache block size. SGLang serving commonly uses 64-token pages.",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=0,
        help="flash_attn split-kv setting. 0 lets the kernel choose.",
    )
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=None,
        help="Optional hardware peak BF16/FP16 TFLOP/s. If set, prints percent of peak.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare outputs from varlen and contiguous kvcache once.",
    )
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        help="Do not try optional vLLM-specific FlashAttention exports.",
    )
    parser.add_argument(
        "--fp8-kvcache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Benchmark SGLang's FA3 flash_attn_with_kvcache path with "
            "torch.float8_e4m3fn Q/K/V and KV cache. This matches SGLang's "
            "--kv-cache-dtype fp8_e4m3 attention path. Enabled by default; "
            "use --no-fp8-kvcache to disable."
        ),
    )
    return parser.parse_args()


def synchronize():
    """Synchronize the active CUDA device before/after timed regions."""
    torch.cuda.synchronize()


def time_cuda(fn, warmup, iters):
    """Run a callable with CUDA events and return average event/wall time in ms."""
    if iters <= 0:
        raise ValueError("--iters must be positive")
    for _ in range(warmup):
        out = fn()
    synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iters):
        out = fn()
    end.record()
    synchronize()
    wall_end = time.perf_counter()

    return out, start.elapsed_time(end) / iters, (wall_end - wall_start) * 1000.0 / iters


def causal_attention_flops(batch_size, q_len, prefix_kv_len, new_kv_len, q_head, head_dim, causal):
    """Estimate causal attention FLOPs for prefix-cache + current-token prefill."""
    if causal:
        attended_keys = q_len * prefix_kv_len + q_len * (new_kv_len + 1) / 2.0
    else:
        attended_keys = q_len * (prefix_kv_len + new_kv_len)
    # QK matmul and PV matmul, each multiply-add counted as 2 FLOPs.
    return 4.0 * batch_size * q_head * head_dim * attended_keys


def dense_attention_flops(batch_size, q_len, kv_len, q_head, head_dim):
    """Estimate dense q_len x kv_len attention FLOPs for reference reporting."""
    return 4.0 * batch_size * q_len * kv_len * q_head * head_dim


def tensor_bytes(*tensors):
    """Return the total storage size of tensors, used as a rough bandwidth proxy."""
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def prefill_chunk_slices(seq_len, chunk_size):
    """Split the current prefill length into SGLang-style chunk ranges."""
    if chunk_size <= 0 or chunk_size >= seq_len:
        return [(0, seq_len)]
    return [(start, min(start + chunk_size, seq_len)) for start in range(0, seq_len, chunk_size)]


def round_up(value, multiple):
    """Round value up to a positive multiple for paged cache allocation."""
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    return ((value + multiple - 1) // multiple) * multiple


def get_flash_attn_export(name):
    """Look up optional flash_attn exports without making them hard dependencies."""
    import flash_attn

    return getattr(flash_attn, name, None)


def check_sgl_fa3_support(device):
    """Return whether SGLang's FA3 custom attention ops are available."""
    try:
        from sgl_kernel.flash_attn import is_fa3_supported
    except Exception as exc:
        return False, f"sgl_kernel FA3 import failed: {exc}"
    try:
        if not is_fa3_supported(device):
            capability = torch.cuda.get_device_capability(device)
            return False, f"FA3 is not supported on this device/CUDA stack, capability={capability}"
    except Exception as exc:
        return False, f"FA3 support check failed: {exc}"
    return True, "supported"


def check_fp8_kvcache_support(device):
    """Return whether the local platform can run SGLang's FP8 KV-cache test."""
    if not hasattr(torch, "float8_e4m3fn"):
        return False, "torch.float8_e4m3fn is not available"
    return check_sgl_fa3_support(device)


def call_with_supported_kwargs(fn, **kwargs):
    """Call optional extension functions while dropping unsupported kwargs."""
    signature = inspect.signature(fn)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return fn(**kwargs)
    supported = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return fn(**supported)


def print_result(name, out, event_ms, wall_ms, flops, dense_flops, bytes_moved, peak_tflops):
    """Print timing, throughput, and output sanity information for one benchmark."""
    effective_tflops = flops / (event_ms / 1000.0) / 1e12
    dense_tflops = dense_flops / (event_ms / 1000.0) / 1e12
    bandwidth_gbs = bytes_moved / (event_ms / 1000.0) / 1e9

    print()
    print(name)
    print("-" * len(name))
    print(f"out shape              : {tuple(out.shape)}")
    print(f"out dtype              : {out.dtype}")
    print(f"out sum                : {float(out.float().sum().item()):.6f}")
    print(f"event time             : {event_ms:.3f} ms")
    print(f"wall time              : {wall_ms:.3f} ms")
    print(f"effective causal compute: {effective_tflops:.2f} TFLOP/s")
    print(f"dense-equivalent compute: {dense_tflops:.2f} TFLOP/s")
    print(f"approx bandwidth       : {bandwidth_gbs:.2f} GB/s")
    if peak_tflops:
        print(f"compute efficiency     : {100.0 * effective_tflops / peak_tflops:.2f}% of peak")


def print_skipped(name, reason):
    """Print a benchmark section for a test that is unavailable locally."""
    print()
    print(name)
    print("-" * len(name))
    print(f"skipped: unsupported on this platform ({reason})")


def make_benchmark_result(name, fn):
    """Run one benchmark and return a compact record for comparison tables."""
    try:
        out, event_ms, wall_ms, bytes_moved = fn()
        return {
            "name": name,
            "out": out,
            "event_ms": event_ms,
            "wall_ms": wall_ms,
            "bytes_moved": bytes_moved,
            "skip_reason": None,
        }
    except Exception as exc:
        return skipped_benchmark_result(name, str(exc))


def skipped_benchmark_result(name, reason):
    """Create a skipped benchmark record for comparison tables."""
    return {
        "name": name,
        "out": None,
        "event_ms": None,
        "wall_ms": None,
        "bytes_moved": None,
        "skip_reason": reason,
    }


def print_comparison_group(title, results, flops, dense_flops, peak_tflops):
    """Print functionally equivalent attention backends in one aligned table."""
    headers = [
        "backend",
        "event_ms",
        "wall_ms",
        "eff_TF/s",
        "dense_TF/s",
        "GB/s",
        "out_dtype",
        "out_shape",
        "out_sum",
    ]
    rows = []
    skipped = []
    for result in results:
        if result["skip_reason"]:
            rows.append([result["name"], "SKIP", "-", "-", "-", "-", "-", "-", "-"])
            skipped.append((result["name"], result["skip_reason"]))
            continue

        out = result["out"]
        event_ms = result["event_ms"]
        wall_ms = result["wall_ms"]
        effective_tflops = flops / (event_ms / 1000.0) / 1e12
        dense_tflops = dense_flops / (event_ms / 1000.0) / 1e12
        bandwidth_gbs = result["bytes_moved"] / (event_ms / 1000.0) / 1e9
        rows.append(
            [
                result["name"],
                f"{event_ms:.3f}",
                f"{wall_ms:.3f}",
                f"{effective_tflops:.2f}",
                f"{dense_tflops:.2f}",
                f"{bandwidth_gbs:.2f}",
                str(out.dtype).replace("torch.", ""),
                "x".join(str(dim) for dim in out.shape),
                f"{float(out.float().sum().item()):.6f}",
            ]
        )

    widths = [
        max(len(str(row[idx])) for row in [headers] + rows)
        for idx in range(len(headers))
    ]

    print()
    print(title)
    print("-" * len(title))
    print("  ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print("  ".join("-" * widths[idx] for idx in range(len(headers))))
    for row in rows:
        print("  ".join(str(row[idx]).ljust(widths[idx]) for idx in range(len(headers))))
    if peak_tflops:
        print(f"peak reference        : {peak_tflops:.2f} TFLOP/s")
    for name, reason in skipped:
        print(f"skip reason [{name}]: {reason}")


def make_inputs(args, device, dtype):
    """Create random Q, prefix KV, and new KV tensors for the logical request."""
    new_kv_len = args.q_len if args.new_kv_len is None else args.new_kv_len
    total_kv_len = args.prefix_kv_len + new_kv_len

    q = torch.randn(
        args.batch_size,
        args.q_len,
        args.q_head,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    prefix_k = torch.randn(
        args.batch_size,
        args.prefix_kv_len,
        args.kv_head,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    prefix_v = torch.randn_like(prefix_k)
    new_k = torch.randn(
        args.batch_size,
        new_kv_len,
        args.kv_head,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    new_v = torch.randn_like(new_k)

    return q, prefix_k, prefix_v, new_k, new_v, new_kv_len, total_kv_len


def benchmark_varlen(args, q, prefix_k, prefix_v, new_k, new_v, new_kv_len, total_kv_len):
    """Benchmark flash_attn_varlen_func over concatenated prefix + new KV."""
    from flash_attn import flash_attn_varlen_func

    q_unpad = q.reshape(args.batch_size * args.q_len, args.q_head, args.head_dim)
    k_unpad = torch.cat((prefix_k, new_k), dim=1).reshape(
        args.batch_size * total_kv_len, args.kv_head, args.head_dim
    )
    v_unpad = torch.cat((prefix_v, new_v), dim=1).reshape(
        args.batch_size * total_kv_len, args.kv_head, args.head_dim
    )
    cu_q = torch.arange(
        0,
        (args.batch_size + 1) * args.q_len,
        args.q_len,
        device=q.device,
        dtype=torch.int32,
    )
    cu_k = torch.arange(
        0,
        (args.batch_size + 1) * total_kv_len,
        total_kv_len,
        device=q.device,
        dtype=torch.int32,
    )

    def run_once():
        return flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_q,
            cu_k,
            args.q_len,
            total_kv_len,
            dropout_p=0.0,
            softmax_scale=1.0 / math.sqrt(args.head_dim),
            causal=args.causal,
        )

    out, event_ms, wall_ms = time_cuda(run_once, args.warmup, args.iters)
    bytes_moved = tensor_bytes(q_unpad, k_unpad, v_unpad, out)
    return out, event_ms, wall_ms, bytes_moved


def benchmark_sgl_varlen(
    args, q, prefix_k, prefix_v, new_k, new_v, total_kv_len, use_fp8_attention
):
    """Benchmark SGLang's FA3 flash_attn_varlen_func over prefix + new KV."""
    from sgl_kernel.flash_attn import flash_attn_varlen_func

    attn_dtype = torch.float8_e4m3fn if use_fp8_attention else q.dtype
    q_unpad = q.reshape(args.batch_size * args.q_len, args.q_head, args.head_dim).to(
        attn_dtype
    )
    k_unpad = torch.cat((prefix_k, new_k), dim=1).reshape(
        args.batch_size * total_kv_len, args.kv_head, args.head_dim
    )
    v_unpad = torch.cat((prefix_v, new_v), dim=1).reshape(
        args.batch_size * total_kv_len, args.kv_head, args.head_dim
    )
    k_unpad = k_unpad.to(attn_dtype)
    v_unpad = v_unpad.to(attn_dtype)
    cu_q = torch.arange(
        0,
        (args.batch_size + 1) * args.q_len,
        args.q_len,
        device=q.device,
        dtype=torch.int32,
    )
    cu_k = torch.arange(
        0,
        (args.batch_size + 1) * total_kv_len,
        total_kv_len,
        device=q.device,
        dtype=torch.int32,
    )

    def run_once():
        return flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_q,
            cu_k,
            args.q_len,
            total_kv_len,
            softmax_scale=1.0 / math.sqrt(args.head_dim),
            causal=args.causal,
        )

    out, event_ms, wall_ms = time_cuda(run_once, args.warmup, args.iters)
    bytes_moved = tensor_bytes(q_unpad, k_unpad, v_unpad, out)
    return out, event_ms, wall_ms, bytes_moved


def benchmark_kvcache_contiguous(args, q, prefix_k, prefix_v, new_k, new_v, total_kv_len):
    """Benchmark upstream flash_attn_with_kvcache with a contiguous BF16/FP16 cache.

    When chunked prefill is enabled, each chunk updates the same cache with its
    new K/V slice and uses cache_seqlens=prefix_len+processed_tokens.
    """
    from flash_attn import flash_attn_with_kvcache

    chunks = prefill_chunk_slices(new_k.shape[1], args.chunked_prefill_size)
    k_cache = torch.empty(
        args.batch_size,
        total_kv_len,
        args.kv_head,
        args.head_dim,
        device=q.device,
        dtype=q.dtype,
    )
    v_cache = torch.empty_like(k_cache)
    k_cache[:, : args.prefix_kv_len].copy_(prefix_k)
    v_cache[:, : args.prefix_kv_len].copy_(prefix_v)
    cache_seqlens = torch.full(
        (args.batch_size,), args.prefix_kv_len, device=q.device, dtype=torch.int32
    )

    def run_once():
        outs = []
        for start, end in chunks:
            cache_seqlens.fill_(args.prefix_kv_len + start)
            outs.append(
                flash_attn_with_kvcache(
                    q[:, start:end],
                    k_cache,
                    v_cache,
                    k=new_k[:, start:end],
                    v=new_v[:, start:end],
                    cache_seqlens=cache_seqlens,
                    softmax_scale=1.0 / math.sqrt(args.head_dim),
                    causal=args.causal,
                    num_splits=args.num_splits,
                )
            )
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)

    out, event_ms, wall_ms = time_cuda(run_once, args.warmup, args.iters)
    bytes_moved = tensor_bytes(q, k_cache, v_cache, new_k, new_v, out)
    return out, event_ms, wall_ms, bytes_moved


def benchmark_sgl_kvcache_contiguous(
    args, q, prefix_k, prefix_v, new_k, new_v, total_kv_len, use_fp8_cache
):
    """Benchmark SGLang's FA3 kvcache path, including the FP8 KV-cache mode.

    SGLang converts Q, new K/V, and the KV cache to float8_e4m3fn for the
    --kv-cache-dtype fp8_e4m3 path, while the kernel output remains BF16.
    """
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    chunks = prefill_chunk_slices(new_k.shape[1], args.chunked_prefill_size)
    attn_dtype = torch.float8_e4m3fn if use_fp8_cache else q.dtype
    q_attn = q.to(attn_dtype)
    new_k_attn = new_k.to(attn_dtype)
    new_v_attn = new_v.to(attn_dtype)
    k_cache = torch.empty(
        args.batch_size,
        total_kv_len,
        args.kv_head,
        args.head_dim,
        device=q.device,
        dtype=attn_dtype,
    )
    v_cache = torch.empty_like(k_cache)
    k_cache[:, : args.prefix_kv_len].copy_(prefix_k.to(attn_dtype))
    v_cache[:, : args.prefix_kv_len].copy_(prefix_v.to(attn_dtype))
    cache_seqlens = torch.full(
        (args.batch_size,), args.prefix_kv_len, device=q.device, dtype=torch.int32
    )

    def run_once():
        outs = []
        for start, end in chunks:
            cache_seqlens.fill_(args.prefix_kv_len + start)
            outs.append(
                flash_attn_with_kvcache(
                    q_attn[:, start:end],
                    k_cache,
                    v_cache,
                    k=new_k_attn[:, start:end],
                    v=new_v_attn[:, start:end],
                    cache_seqlens=cache_seqlens,
                    softmax_scale=1.0 / math.sqrt(args.head_dim),
                    causal=args.causal,
                    num_splits=args.num_splits,
                )
            )
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)

    out, event_ms, wall_ms = time_cuda(run_once, args.warmup, args.iters)
    bytes_moved = tensor_bytes(q_attn, k_cache, v_cache, new_k_attn, new_v_attn, out)
    return out, event_ms, wall_ms, bytes_moved


def benchmark_sgl_kvcache_paged(
    args, q, prefix_k, prefix_v, new_k, new_v, total_kv_len, use_fp8_cache
):
    """Benchmark SGLang's FA3 kvcache path with a paged KV-cache layout.

    SGLang's wrapper takes page_table instead of upstream flash-attn's
    block_table. The cache allocation is padded to the SGLang page size, while
    cache_seqlens keeps the true prefix+processed length for each chunk.
    """
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    chunks = prefill_chunk_slices(new_k.shape[1], args.chunked_prefill_size)
    padded_total_kv_len = round_up(total_kv_len, args.sgl_page_size)
    blocks_per_seq = padded_total_kv_len // args.sgl_page_size
    num_blocks = args.batch_size * blocks_per_seq
    attn_dtype = torch.float8_e4m3fn if use_fp8_cache else q.dtype
    q_attn = q.to(attn_dtype)
    new_k_attn = new_k.to(attn_dtype)
    new_v_attn = new_v.to(attn_dtype)
    flat_k = torch.empty(
        args.batch_size,
        padded_total_kv_len,
        args.kv_head,
        args.head_dim,
        device=q.device,
        dtype=attn_dtype,
    )
    flat_v = torch.empty_like(flat_k)
    flat_k.zero_()
    flat_v.zero_()
    flat_k[:, : args.prefix_kv_len].copy_(prefix_k.to(attn_dtype))
    flat_v[:, : args.prefix_kv_len].copy_(prefix_v.to(attn_dtype))
    k_cache = flat_k.reshape(num_blocks, args.sgl_page_size, args.kv_head, args.head_dim).contiguous()
    v_cache = flat_v.reshape(num_blocks, args.sgl_page_size, args.kv_head, args.head_dim).contiguous()
    page_table = torch.arange(num_blocks, device=q.device, dtype=torch.int32).view(
        args.batch_size, blocks_per_seq
    )
    cache_seqlens = torch.full(
        (args.batch_size,), args.prefix_kv_len, device=q.device, dtype=torch.int32
    )

    def run_once():
        outs = []
        for start, end in chunks:
            cache_seqlens.fill_(args.prefix_kv_len + start)
            outs.append(
                flash_attn_with_kvcache(
                    q_attn[:, start:end],
                    k_cache,
                    v_cache,
                    k=new_k_attn[:, start:end],
                    v=new_v_attn[:, start:end],
                    cache_seqlens=cache_seqlens,
                    page_table=page_table,
                    softmax_scale=1.0 / math.sqrt(args.head_dim),
                    causal=args.causal,
                    num_splits=args.num_splits,
                )
            )
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)

    out, event_ms, wall_ms = time_cuda(run_once, args.warmup, args.iters)
    bytes_moved = tensor_bytes(q_attn, k_cache, v_cache, new_k_attn, new_v_attn, out, page_table)
    return out, event_ms, wall_ms, bytes_moved


def benchmark_kvcache_paged(args, q, prefix_k, prefix_v, new_k, new_v, total_kv_len):
    """Benchmark upstream flash_attn_with_kvcache with paged KV-cache layout."""
    from flash_attn import flash_attn_with_kvcache

    if args.page_size % 256 != 0:
        raise ValueError("flash_attn_with_kvcache paged cache requires --page-size to be a multiple of 256")
    if total_kv_len % args.page_size != 0:
        raise ValueError("total KV length must be divisible by --page-size for this benchmark")

    chunks = prefill_chunk_slices(new_k.shape[1], args.chunked_prefill_size)
    blocks_per_seq = total_kv_len // args.page_size
    num_blocks = args.batch_size * blocks_per_seq
    flat_k = torch.empty(
        args.batch_size,
        total_kv_len,
        args.kv_head,
        args.head_dim,
        device=q.device,
        dtype=q.dtype,
    )
    flat_v = torch.empty_like(flat_k)
    flat_k[:, : args.prefix_kv_len].copy_(prefix_k)
    flat_v[:, : args.prefix_kv_len].copy_(prefix_v)
    k_cache = flat_k.reshape(num_blocks, args.page_size, args.kv_head, args.head_dim).contiguous()
    v_cache = flat_v.reshape(num_blocks, args.page_size, args.kv_head, args.head_dim).contiguous()
    block_table = torch.arange(num_blocks, device=q.device, dtype=torch.int32).view(
        args.batch_size, blocks_per_seq
    )
    cache_seqlens = torch.full(
        (args.batch_size,), args.prefix_kv_len, device=q.device, dtype=torch.int32
    )

    def run_once():
        outs = []
        for start, end in chunks:
            cache_seqlens.fill_(args.prefix_kv_len + start)
            outs.append(
                flash_attn_with_kvcache(
                    q[:, start:end],
                    k_cache,
                    v_cache,
                    k=new_k[:, start:end],
                    v=new_v[:, start:end],
                    cache_seqlens=cache_seqlens,
                    block_table=block_table,
                    softmax_scale=1.0 / math.sqrt(args.head_dim),
                    causal=args.causal,
                    num_splits=args.num_splits,
                )
            )
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)

    out, event_ms, wall_ms = time_cuda(run_once, args.warmup, args.iters)
    bytes_moved = tensor_bytes(q, k_cache, v_cache, new_k, new_v, out, block_table)
    return out, event_ms, wall_ms, bytes_moved


def benchmark_vllm_varlen(args, q, prefix_k, prefix_v, new_k, new_v, new_kv_len, total_kv_len):
    """Benchmark optional vLLM varlen export when the local flash_attn exposes it."""
    vllm_flash_attn_varlen_func = get_flash_attn_export("vllm_flash_attn_varlen_func")
    if vllm_flash_attn_varlen_func is None:
        return None
    if args.vllm_page_size != 64:
        raise ValueError("local vLLM FlashAttention exports require --vllm-page-size 64")
    if total_kv_len % args.vllm_page_size != 0:
        raise ValueError("total KV length must be divisible by --vllm-page-size for vLLM varlen benchmark")

    blocks_per_seq = total_kv_len // args.vllm_page_size
    num_blocks = args.batch_size * blocks_per_seq
    q_unpad = q.reshape(args.batch_size * args.q_len, args.q_head, args.head_dim)
    flat_k = torch.cat((prefix_k, new_k), dim=1)
    flat_v = torch.cat((prefix_v, new_v), dim=1)
    k_cache = flat_k.reshape(num_blocks, args.vllm_page_size, args.kv_head, args.head_dim)
    k_cache = k_cache.permute(0, 2, 1, 3).contiguous()
    v_cache = flat_v.reshape(num_blocks, args.vllm_page_size, args.kv_head, args.head_dim)
    v_cache = v_cache.permute(0, 2, 3, 1).contiguous()
    cu_q = torch.arange(
        0,
        (args.batch_size + 1) * args.q_len,
        args.q_len,
        device=q.device,
        dtype=torch.int32,
    )
    seqused_k = torch.full(
        (args.batch_size,), total_kv_len, device=q.device, dtype=torch.int32
    )
    block_table = torch.arange(num_blocks, device=q.device, dtype=torch.int32).view(
        args.batch_size, blocks_per_seq
    )

    def run_once():
        return call_with_supported_kwargs(
            vllm_flash_attn_varlen_func,
            q=q_unpad,
            k=k_cache,
            v=v_cache,
            max_seqlen_q=args.q_len,
            cu_seqlens_q=cu_q,
            max_seqlen_k=total_kv_len,
            cu_seqlens_k=None,
            seqused_k=seqused_k,
            softmax_scale=1.0 / math.sqrt(args.head_dim),
            causal=args.causal,
            window_size=(-1, -1),
            softcap=0.0,
            block_table=block_table,
            return_softmax_lse=False,
            is_prefix_cache=True,
            fa_version=2,
        )

    out, event_ms, wall_ms = time_cuda(run_once, args.warmup, args.iters)
    bytes_moved = tensor_bytes(q_unpad, k_cache, v_cache, out, block_table, seqused_k)
    return out, event_ms, wall_ms, bytes_moved


def benchmark_vllm_kvcache(args, q, prefix_k, prefix_v, new_k, new_v, total_kv_len):
    """Benchmark optional vLLM kvcache export when available."""
    vllm_flash_attn_with_kvcache = get_flash_attn_export("vllm_flash_attn_with_kvcache")
    if vllm_flash_attn_with_kvcache is None:
        return None
    if args.vllm_page_size != 64:
        raise ValueError("local vLLM FlashAttention exports require --vllm-page-size 64")
    if total_kv_len % args.vllm_page_size != 0:
        raise ValueError("total KV length must be divisible by --vllm-page-size for vLLM kvcache benchmark")
    blocks_per_seq = total_kv_len // args.vllm_page_size
    num_blocks = args.batch_size * blocks_per_seq
    flat_k = torch.cat((prefix_k, new_k), dim=1)
    flat_v = torch.cat((prefix_v, new_v), dim=1)
    k_cache = flat_k.reshape(num_blocks, args.vllm_page_size, args.kv_head, args.head_dim)
    k_cache = k_cache.permute(0, 2, 1, 3).contiguous()
    v_cache = flat_v.reshape(num_blocks, args.vllm_page_size, args.kv_head, args.head_dim)
    v_cache = v_cache.permute(0, 2, 3, 1).contiguous()
    cache_seqlens = torch.full(
        (args.batch_size,), total_kv_len, device=q.device, dtype=torch.int32
    )
    block_table = torch.arange(num_blocks, device=q.device, dtype=torch.int32).view(
        args.batch_size, blocks_per_seq
    )
    q_scale = torch.tensor([1.0], device=q.device, dtype=torch.float32)
    k_scale = torch.tensor([1.0], device=q.device, dtype=torch.float32)
    v_scale = torch.tensor([1.0], device=q.device, dtype=torch.float32)

    def run_once():
        return call_with_supported_kwargs(
            vllm_flash_attn_with_kvcache,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            softmax_scale=1.0 / math.sqrt(args.head_dim),
            causal=args.causal,
            num_splits=args.num_splits,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            max_seqlen_k=total_kv_len,
            return_softmax_lse=False,
        )

    out, event_ms, wall_ms = time_cuda(run_once, args.warmup, args.iters)
    bytes_moved = tensor_bytes(q, k_cache, v_cache, out, block_table, cache_seqlens)
    return out, event_ms, wall_ms, bytes_moved


def main():
    """Prepare inputs, run selected benchmark backends, and print comparisons."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.q_head % args.kv_head != 0:
        raise ValueError("--q-head must be divisible by --kv-head for MQA/GQA")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.chunked_prefill_size < 0:
        raise ValueError("--chunked-prefill-size must be non-negative")
    if args.sgl_page_size <= 0:
        raise ValueError("--sgl-page-size must be positive")

    new_kv_len = args.q_len if args.new_kv_len is None else args.new_kv_len
    if args.chunked_prefill_size > 0 and args.q_len != new_kv_len:
        raise ValueError(
            "--chunked-prefill-size requires --q-len to equal --new-kv-len, "
            "which is the prefill/extend case this benchmark models."
        )
    total_kv_len = args.prefix_kv_len + new_kv_len
    chunks = prefill_chunk_slices(new_kv_len, args.chunked_prefill_size)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    sgl_fa3_supported, sgl_fa3_reason = check_sgl_fa3_support(device)
    fp8_supported, fp8_support_reason = check_fp8_kvcache_support(device)

    q, prefix_k, prefix_v, new_k, new_v, new_kv_len, total_kv_len = make_inputs(args, device, dtype)
    flops = causal_attention_flops(
        args.batch_size,
        args.q_len,
        args.prefix_kv_len,
        new_kv_len,
        args.q_head,
        args.head_dim,
        args.causal,
    )
    dense_flops = dense_attention_flops(
        args.batch_size, args.q_len, total_kv_len, args.q_head, args.head_dim
    )

    print("FlashAttention benchmark")
    print("========================")
    print(f"device                 : {torch.cuda.get_device_name(device)}")
    print(f"dtype                  : {args.dtype}")
    print(f"batch_size             : {args.batch_size}")
    print(f"q_len                  : {args.q_len}")
    print(f"prefix_kv_len          : {args.prefix_kv_len}")
    print(f"new_kv_len             : {new_kv_len}")
    print(f"total_kv_len           : {total_kv_len}")
    print(f"q_head                 : {args.q_head}")
    print(f"kv_head                : {args.kv_head}")
    print(f"head_dim               : {args.head_dim}")
    print(f"causal                 : {args.causal}")
    print(f"chunked_prefill_size   : {args.chunked_prefill_size}")
    print(f"sgl_page_size          : {args.sgl_page_size}")
    print(f"prefill chunks         : {len(chunks)} {chunks}")
    print(f"warmup / iters         : {args.warmup} / {args.iters}")
    print(
        "vLLM varlen export     : "
        f"{'yes' if get_flash_attn_export('vllm_flash_attn_varlen_func') else 'no'}"
    )
    print(
        "vLLM kvcache export    : "
        f"{'yes' if get_flash_attn_export('vllm_flash_attn_with_kvcache') else 'no'}"
    )
    print(
        "SGLang FA3 custom ops  : "
        f"{'enabled' if sgl_fa3_supported else 'unsupported'} ({sgl_fa3_reason})"
    )
    if args.fp8_kvcache:
        print(
            "SGLang FP8 KV case     : "
            f"{'enabled' if fp8_supported else 'unsupported'} ({fp8_support_reason})"
        )
    else:
        print("SGLang FP8 KV case     : disabled")

    varlen_out = None
    kvcache_contig_out = None

    if args.mode in ("both", "varlen"):
        varlen_results = []
        upstream_result = make_benchmark_result(
            "upstream flash_attn_varlen_func",
            lambda: benchmark_varlen(
                args, q, prefix_k, prefix_v, new_k, new_v, new_kv_len, total_kv_len
            ),
        )
        varlen_out = upstream_result["out"]
        varlen_results.append(upstream_result)

        if sgl_fa3_supported:
            varlen_results.append(
                make_benchmark_result(
                    f"sgl_kernel flash_attn_varlen_func {args.dtype}",
                    lambda: benchmark_sgl_varlen(
                        args,
                        q,
                        prefix_k,
                        prefix_v,
                        new_k,
                        new_v,
                        total_kv_len,
                        use_fp8_attention=False,
                    ),
                )
            )

            if args.fp8_kvcache and fp8_supported:
                varlen_results.append(
                    make_benchmark_result(
                        "sgl_kernel flash_attn_varlen_func fp8_e4m3",
                        lambda: benchmark_sgl_varlen(
                            args,
                            q,
                            prefix_k,
                            prefix_v,
                            new_k,
                            new_v,
                            total_kv_len,
                            use_fp8_attention=True,
                        ),
                    )
                )
            elif args.fp8_kvcache:
                varlen_results.append(
                    skipped_benchmark_result(
                        "sgl_kernel flash_attn_varlen_func fp8_e4m3",
                        fp8_support_reason,
                    )
                )
        else:
            varlen_results.append(
                skipped_benchmark_result(
                    f"sgl_kernel flash_attn_varlen_func {args.dtype}",
                    sgl_fa3_reason,
                )
            )
            if args.fp8_kvcache:
                varlen_results.append(
                    skipped_benchmark_result(
                        "sgl_kernel flash_attn_varlen_func fp8_e4m3",
                        sgl_fa3_reason,
                    )
                )

        print_comparison_group(
            "varlen attention comparison",
            varlen_results,
            flops,
            dense_flops,
            args.peak_tflops,
        )

    if args.mode in ("both", "kvcache") and args.cache_layout in ("both", "contiguous"):
        contig_results = []
        upstream_result = make_benchmark_result(
            f"upstream flash_attn_with_kvcache {args.dtype}-kv",
            lambda: benchmark_kvcache_contiguous(
                args, q, prefix_k, prefix_v, new_k, new_v, total_kv_len
            ),
        )
        kvcache_contig_out = upstream_result["out"]
        contig_results.append(upstream_result)

        if sgl_fa3_supported:
            contig_results.append(
                make_benchmark_result(
                    f"sgl_kernel flash_attn_with_kvcache contiguous {args.dtype}-kv",
                    lambda: benchmark_sgl_kvcache_contiguous(
                        args,
                        q,
                        prefix_k,
                        prefix_v,
                        new_k,
                        new_v,
                        total_kv_len,
                        use_fp8_cache=False,
                    ),
                )
            )

            if args.fp8_kvcache and fp8_supported:
                contig_results.append(
                    make_benchmark_result(
                        "sgl_kernel flash_attn_with_kvcache contiguous fp8_e4m3-kv",
                        lambda: benchmark_sgl_kvcache_contiguous(
                            args,
                            q,
                            prefix_k,
                            prefix_v,
                            new_k,
                            new_v,
                            total_kv_len,
                            use_fp8_cache=True,
                        ),
                    )
                )
            elif args.fp8_kvcache:
                contig_results.append(
                    skipped_benchmark_result(
                        "sgl_kernel flash_attn_with_kvcache contiguous fp8_e4m3-kv",
                        fp8_support_reason,
                    )
                )
        else:
            contig_results.append(
                skipped_benchmark_result(
                    f"sgl_kernel flash_attn_with_kvcache contiguous {args.dtype}-kv",
                    sgl_fa3_reason,
                )
            )
            if args.fp8_kvcache:
                contig_results.append(
                    skipped_benchmark_result(
                        "sgl_kernel flash_attn_with_kvcache contiguous fp8_e4m3-kv",
                        sgl_fa3_reason,
                    )
                )

        print_comparison_group(
            "kvcache contiguous attention comparison",
            contig_results,
            flops,
            dense_flops,
            args.peak_tflops,
        )

    if args.mode in ("both", "kvcache") and args.cache_layout in ("both", "paged"):
        paged_results = [
            make_benchmark_result(
                f"upstream flash_attn_with_kvcache paged {args.dtype}-kv",
                lambda: benchmark_kvcache_paged(
                    args, q, prefix_k, prefix_v, new_k, new_v, total_kv_len
                ),
            )
        ]

        if sgl_fa3_supported:
            paged_results.append(
                make_benchmark_result(
                    f"sgl_kernel flash_attn_with_kvcache paged {args.dtype}-kv",
                    lambda: benchmark_sgl_kvcache_paged(
                        args,
                        q,
                        prefix_k,
                        prefix_v,
                        new_k,
                        new_v,
                        total_kv_len,
                        use_fp8_cache=False,
                    ),
                )
            )

            if args.fp8_kvcache and fp8_supported:
                paged_results.append(
                    make_benchmark_result(
                        "sgl_kernel flash_attn_with_kvcache paged fp8_e4m3-kv",
                        lambda: benchmark_sgl_kvcache_paged(
                            args,
                            q,
                            prefix_k,
                            prefix_v,
                            new_k,
                            new_v,
                            total_kv_len,
                            use_fp8_cache=True,
                        ),
                    )
                )
            elif args.fp8_kvcache:
                paged_results.append(
                    skipped_benchmark_result(
                        "sgl_kernel flash_attn_with_kvcache paged fp8_e4m3-kv",
                        fp8_support_reason,
                    )
                )
        else:
            paged_results.append(
                skipped_benchmark_result(
                    f"sgl_kernel flash_attn_with_kvcache paged {args.dtype}-kv",
                    sgl_fa3_reason,
                )
            )
            if args.fp8_kvcache:
                paged_results.append(
                    skipped_benchmark_result(
                        "sgl_kernel flash_attn_with_kvcache paged fp8_e4m3-kv",
                        sgl_fa3_reason,
                    )
                )

        print_comparison_group(
            "kvcache paged attention comparison",
            paged_results,
            flops,
            dense_flops,
            args.peak_tflops,
        )

    if not args.no_vllm and args.mode in ("both", "vllm"):
        vllm_varlen_result = benchmark_vllm_varlen(
            args, q, prefix_k, prefix_v, new_k, new_v, new_kv_len, total_kv_len
        )
        if vllm_varlen_result is None:
            print()
            print("vllm_flash_attn_varlen_func")
            print("---------------------------")
            print("skipped: not exported by installed flash_attn")
        else:
            out, event_ms, wall_ms, bytes_moved = vllm_varlen_result
            print_result(
                "vllm_flash_attn_varlen_func",
                out,
                event_ms,
                wall_ms,
                flops,
                dense_flops,
                bytes_moved,
                args.peak_tflops,
            )

        vllm_kvcache_result = benchmark_vllm_kvcache(
            args, q, prefix_k, prefix_v, new_k, new_v, total_kv_len
        )
        if vllm_kvcache_result is None:
            print()
            print("vllm_flash_attn_with_kvcache")
            print("----------------------------")
            print("skipped: not exported by installed flash_attn")
        else:
            out, event_ms, wall_ms, bytes_moved = vllm_kvcache_result
            print_result(
                "vllm_flash_attn_with_kvcache",
                out,
                event_ms,
                wall_ms,
                flops,
                dense_flops,
                bytes_moved,
                args.peak_tflops,
            )

    if args.check and varlen_out is not None and kvcache_contig_out is not None:
        varlen_4d = varlen_out.reshape(args.batch_size, args.q_len, args.q_head, args.head_dim)
        max_abs = (varlen_4d - kvcache_contig_out).abs().max().item()
        mean_abs = (varlen_4d - kvcache_contig_out).abs().float().mean().item()
        print()
        print("Output check")
        print("------------")
        print(f"varlen vs contiguous kvcache max abs : {max_abs:.6f}")
        print(f"varlen vs contiguous kvcache mean abs: {mean_abs:.6f}")


if __name__ == "__main__":
    main()
