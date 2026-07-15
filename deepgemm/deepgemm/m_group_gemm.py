import os
from typing import Any, Dict, Optional, Tuple
import torch

def load_deep_gemm_library():
    lib_path = os.environ.get("DEEP_GEMM_LIB_PATH")

    if not lib_path:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        checked = [os.path.join(current_dir, "libdeep_gemm.so")]
        for p in checked:
            if os.path.exists(p):
                lib_path = p
                break
    else:
        checked = [lib_path]

    if not lib_path or not os.path.exists(lib_path):
        raise FileNotFoundError(
            f"Could not find libdeep_gemm.so. Checked paths: {checked}"
        )

    torch.ops.load_library(lib_path)


def _ensure_torch_ops_loaded() -> None:
    try:
        import importlib

        importlib.import_module("deepgemm.op")
    except Exception:
        load_deep_gemm_library()


try:
    _ensure_torch_ops_loaded()
except Exception as e:
    print(f"Warning: Failed to load DeepGEMM native library: {e}")


def m_grouped_gemm_fp8_fp8_bf16_nt_masked(
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    experts: int,
    *,
    block_wise: bool = True,
    cu: int = 128,
    enable_overlap: bool = False,
    signal: Optional[torch.Tensor] = None,
) -> Tuple[int, int]:
    matrix_a, matrix_a_scale = lhs
    matrix_b, matrix_b_scale = rhs
    grouped_gemm_fp8_fp8_bf16_nt_masked_entry(
        matrix_a,
        matrix_b,
        matrix_a_scale,
        matrix_b_scale,
        masked_m,
        out,
        expected_m,
        experts,
        cu=cu,
        block_wise=block_wise,
        over_lap=enable_overlap,
        signal=signal,
    )
    block_m = 16
    bn = 256
    return block_m, bn


def m_grouped_w8a8_gemm_nt_masked_ll(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int,
    config: Optional[Dict[str, Any]] = None,
    *,
    block_wise: bool = True,
    cu: int = 128,
) -> torch.Tensor:
    """W8A8 entry point for low-latency masked GEMM.

    b[0] must be 6-D packed weights from pack_int8_weight_enk_to_w6_low_latency.
    Use block_wise=False with 2-D per-token / per-channel scales. config is reserved.
    """
    experts = int(a[0].shape[0])
    if int(masked_m.numel()) != experts:
        raise ValueError(
            f"masked_m length {masked_m.numel()} != expert count from a[0] ({experts})"
        )
    if int(b[0].shape[0]) != experts:
        raise ValueError(
            f"b[0].shape[0] ({b[0].shape[0]}) != expert count ({experts})"
        )
    m_grouped_gemm_fp8_fp8_bf16_nt_masked(
        a,
        b,
        d,
        masked_m,
        expected_m_per_group,
        experts,
        block_wise=block_wise,
        cu=cu,
    )
    return d


def m_grouped_fp8_gemm_nt_masked_ll(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m_per_group: int,
    enable_overlap: bool = False,
    signal: Optional[torch.Tensor] = None,
    config: Optional[Dict[str, Any]] = None,
    *,
    block_wise: bool = False,
    cu: int = 128,
) -> torch.Tensor:
    
    del config

    experts = int(a[0].shape[0])
    if int(masked_m.numel()) != experts:
        raise ValueError(
            f"masked_m length {masked_m.numel()} != expert count from a[0] ({experts})"
        )
    if int(b[0].shape[0]) != experts:
        raise ValueError(
            f"b[0].shape[0] ({b[0].shape[0]}) != expert count ({experts})"
        )

    matrix_a, a_scale = a
    matrix_b, b_scale = b

    if not block_wise:
        if a_scale.dim() == 3:
            a_scale = a_scale.squeeze(-1)
        if b_scale.dim() == 3:
            b_scale = b_scale.squeeze(-1)
    a_scale = a_scale.contiguous()
    b_scale = b_scale.contiguous()

    masked_m_i32 = masked_m if masked_m.dtype == torch.int32 else masked_m.to(torch.int32)

    grouped_gemm_fp8_fp8_bf16_nt_masked_entry(
        matrix_a,
        matrix_b,
        a_scale,
        b_scale,
        masked_m_i32,
        d,
        int(expected_m_per_group),
        experts,
        cu=cu,
        block_wise=block_wise,
        over_lap=enable_overlap,
        signal=signal,
    )
    return d


def grouped_gemm_fp8_fp8_bf16_nt_masked_entry(
    matrix_a: torch.Tensor,
    matrix_b: torch.Tensor,
    matrix_a_scale: torch.Tensor,
    matrix_b_scale: torch.Tensor,
    actual_tokens: torch.Tensor,
    matrix_c: torch.Tensor,
    max_tokens: int,
    experts: int,
    cu: int = 128,
    block_wise: bool = True,
    over_lap: bool = False,
    signal: Optional[torch.Tensor] = None,
):
    assert matrix_a_scale.is_contiguous()
    return torch.ops.deep_gemm.low_latency_grouped_gemm(
        matrix_a,
        matrix_b,
        matrix_a_scale,
        matrix_b_scale,
        actual_tokens,
        matrix_c,
        max_tokens,
        experts,
        cu,
        block_wise,
        over_lap,
        signal,
    )


_CUSTOMER_N_TILE = 16
_CUSTOMER_K_TILE = 16
_CUSTOMER_K_PACK = _CUSTOMER_K_TILE * 4
_CUSTOMER_N_GFX936_LD_PERMUTE = (0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15)


def _logical_int8_weight_2d_to_w6(
    w2d: torch.Tensor, num_experts: int, n: int, k: int
) -> torch.Tensor:
    n_in = _CUSTOMER_N_TILE
    n_out_band = n // n_in
    k_in_tile = _CUSTOMER_K_TILE
    k_mid = 4
    k_out_band = k // (k_in_tile * k_mid)
    w6 = w2d.reshape(num_experts, n_out_band, n_in, k_out_band, k_mid, k_in_tile)
    w6 = w6.permute(0, 3, 1, 4, 2, 5)

    # gfx936 reads the 16-row N sub-tile in the order previously expressed by
    # STORE_PERMUTE in the kernel. Move that reorder to pack time so the hot
    # GEMM path can keep STORE_PERMUTE=false.
    n_perm = torch.tensor(
        _CUSTOMER_N_GFX936_LD_PERMUTE,
        dtype=torch.long,
        device=w6.device,
    )
    return w6.index_select(4, n_perm).contiguous()


def pack_int8_weight_enk_to_w6_low_latency(weight_enk: torch.Tensor) -> torch.Tensor:
    """Reshape [E, N, K] int8 weights into the 6-D layout expected by low_latency_grouped_gemm."""
    if weight_enk.dim() != 3:
        raise ValueError(
            f"weight_enk must be [E,N,K], got dim {weight_enk.dim()} shape {tuple(weight_enk.shape)}"
        )
    num_experts = int(weight_enk.size(0))
    n = int(weight_enk.size(1))
    k = int(weight_enk.size(2))
    if n % _CUSTOMER_N_TILE != 0 or k % _CUSTOMER_K_PACK != 0:
        raise ValueError(
            f"N={n}, K={k} must satisfy N%{_CUSTOMER_N_TILE}==0 and K%{_CUSTOMER_K_PACK}==0"
        )
    return _logical_int8_weight_2d_to_w6(
        weight_enk.contiguous(), num_experts, n, k
    )


