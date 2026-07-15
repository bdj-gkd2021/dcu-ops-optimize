import os
import subprocess
import importlib
from pathlib import Path
from torch.utils.cpp_extension import ROCM_HOME
ROOT_DIR = Path(__file__).parent.resolve()

def _run_cmd(cmd, shell=False):
    try:
        return subprocess.check_output(cmd, cwd=ROOT_DIR, stderr=subprocess.DEVNULL, shell=shell).decode("ascii").strip()
    except Exception:
        return None
    
def get_dtk_version(ROCM_HOME):
    dtk_version = _run_cmd(["cat", os.path.join(ROCM_HOME, '.info/rocm_version')])
    dtk_version = ''.join(dtk_version.split('.'))
    dtk_version = int(dtk_version)
    return dtk_version

dtk_version = get_dtk_version(ROCM_HOME)
op = importlib.import_module('.op', __name__)

from .utils import gfx, DEEPGEMM_ROOT_DIR, DEEPGEMM_ASM_DIR
from .m_group_gemm_nt_contiguous import m_grouped_i8_gemm_nt_contiguous, m_grouped_i8_gemm_nt_contiguous_nopad, \
      m_grouped_bf16_gemm_nt_contiguous, m_grouped_fp8_gemm_nt_contiguous
from .moe_marlin_prefill import (
    assert_moe_prefill_shapes,
    moe_prefill_prepare_scales_2d_3d,
    moe_prefill_routing_contiguous_expert_blocks,
    moe_w8a8_i8_marlin_prefill_down,
)
from .m_group_gemm_nt_masked import m_grouped_w4a8_gemm_nt_masked, m_grouped_fp8_gemm_nt_masked, \
      m_grouped_w8a8_gemm_nt_masked, m_grouped_bf16_gemm_nt_masked, m_grouped_w8a8_gemm_nt_masked_impl, m_grouped_fp8_gemm_nt_masked_impl
from .mqa_logits import mqa_logits, get_paged_mqa_logits_metadata, paged_mqa_logits
from .gemm import fp8_gemm
from .tf32_hc_pernorm_gemm import tf32_hc_pernorm_gemm
