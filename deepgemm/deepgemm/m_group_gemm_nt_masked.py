import torch
from typing import Optional, Dict, Any, List, NamedTuple
import functools
import os
import math
from . import op
from .config_deepgemm import gemm_deepgemm_masked_config
from .config import get_deepgemm_groupgemm_w8a8_config


def round_up_to_next_power_of_2(x):
    if x == 0:
        return 1  # 特殊情况：0 的向上取整是 1
    x -= 1  # 处理 x 已经是 2 的幂的情况
    # 填充所有低位为 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    return x + 1


def ceil_div(a, b):
    return (a + b - 1) // b


@functools.lru_cache(maxsize=128)
def deepgemm_masked_config(m, size_k=32):

    # # cuda选择 目前只支持7168和2048和gfx938
    if size_k == 7168 and m <= 12:
        return 41
    elif size_k == 2048 and m <= 12:
        return 33

    if m < 16:
        adapt_m = 16  # 向上取整到下一个16的倍数
    else:
        adapt_m = round_up_to_next_power_of_2(m)
    key = f"EXPECTED_M{adapt_m}"
    if key in gemm_deepgemm_masked_config.get_deepgemm_dict():
        MODE = gemm_deepgemm_masked_config.get_deepgemm_dict()[key]
    else:
        MODE = 1000
    return MODE

@functools.lru_cache(maxsize=128)
def deepgemm_masked_config_i8fp8(m, size_k=32):

    # # cuda选择 目前只支持7168和2048和gfx938
    # marlin layout 改变了
    # if size_k == 7168 and m <= 12:
    #     return 184 #248 41 
    # elif size_k == 2048 and m <= 17:
    #     return 170

    if m < 16:
        adapt_m = 16  # 向上取整到下一个16的倍数
    else:
        adapt_m = round_up_to_next_power_of_2(m)
    key = f"EXPECTED_M{adapt_m}"
    if key in gemm_deepgemm_masked_config.get_deepgemm_dict():
        MODE = gemm_deepgemm_masked_config.get_deepgemm_dict()[key]
    else:
        MODE = 1002
    return MODE

# 读取json文件的方式 待完善
@functools.lru_cache
def get_groupgemm_w8a8_mode(expected_m: int, n: int, k: int) -> int:
    """
    获取 groupgemm w8a8 kernel 的 mode
    
    优先从 JSON 配置文件读取，如果没有则使用默认配置
    """
    device_props = torch.cuda.get_device_properties(0)
    device_name = device_props.gcnArchName.split(':')[0] if hasattr(device_props, 'gcnArchName') else device_props.name
    num_cus = device_props.multi_processor_count
    E = 1  # E 不影响 config 选择
    
    config, status = get_deepgemm_groupgemm_w8a8_config(
        E=E, M=expected_m, N=n, K=k,
        device_name=device_name, num_cus=num_cus)
    
    if status and 'MODE' in config:
        return config['MODE']
    
    # 回退到旧的 config 方式
    return deepgemm_masked_config(expected_m)


def m_grouped_w4a8_gemm_nt_masked(a: tuple[torch.Tensor, torch.Tensor],
                                  b: tuple[torch.Tensor, torch.Tensor],
                                  d: torch.Tensor,
                                  masked_m: torch.Tensor,
                                  expected_m_per_group: int,
                                  config: Optional[Dict[str, Any]] = {"MODE": 1002,}) -> torch.Tensor:

    mode = config['MODE']
    input, a_scale = a
    b_qweight, b_scale = b
    output = d
    op.m_grouped_marlin_w4a8_gemm_nt_masked(input, b_qweight, output, a_scale, b_scale,
                                        masked_m, expected_m_per_group, mode)
    return output


def m_grouped_fp8_gemm_nt_masked(a: tuple[torch.Tensor, torch.Tensor],
                                  b: tuple[torch.Tensor, torch.Tensor],
                                  d: torch.Tensor,
                                  masked_m: torch.Tensor,
                                  expected_m_per_group: int,
                                  enable_overlap: bool = False,
                                  signal: Optional[torch.Tensor] = None,
                                  config: Optional[Dict[str, Any]] = {"MODE": 1000,}) -> torch.Tensor:

    size_k = a[0].shape[2]
    size_n = b[1].shape[1]
    mode = deepgemm_masked_config_i8fp8(expected_m_per_group, size_k)
    threshold = ceil_div(size_n, gemm_deepgemm_masked_config.get_deepgemm_block_n(mode))
    block_m = gemm_deepgemm_masked_config.get_deepgemm_block_m(mode)
    input, a_scale = a
    b_qweight, b_scale = b
    output = d
    op.m_grouped_marlin_fp8_gemm_nt_masked(input, b_qweight, output, a_scale, b_scale,
                                        masked_m, expected_m_per_group, enable_overlap, signal, mode)
    return block_m, threshold

def m_grouped_fp8_gemm_nt_masked_impl(a: tuple[torch.Tensor, torch.Tensor],
                                  b: tuple[torch.Tensor, torch.Tensor],
                                  d: torch.Tensor,
                                  masked_m: torch.Tensor,
                                  expected_m_per_group: int,
                                  mode) -> torch.Tensor:

    input, a_scale = a
    b_qweight, b_scale = b
    output = d
    op.m_grouped_marlin_fp8_gemm_nt_masked(input, b_qweight, output, a_scale, b_scale,
                                        masked_m, expected_m_per_group, mode)
    return output


def m_grouped_w8a8_gemm_nt_masked(a: tuple[torch.Tensor, torch.Tensor],
                                  b: tuple[torch.Tensor, torch.Tensor],
                                  d: torch.Tensor,
                                  masked_m: torch.Tensor,
                                  expected_m_per_group: int,
                                  config: Optional[Dict[str, Any]] = {"MODE": 1002,}) -> torch.Tensor:
    if config is not None and 'MODE' in config:
        mode = config['MODE']
    else:
        size_k = a[0].shape[2]
        mode = deepgemm_masked_config_i8fp8(expected_m_per_group, size_k)
    input, a_scale = a
    b_qweight, b_scale = b
    output = d
    op.m_grouped_marlin_w8a8_gemm_nt_masked(input, b_qweight, output, a_scale, b_scale,
                                        masked_m, expected_m_per_group, mode)
    return output

def m_grouped_w8a8_gemm_nt_masked_impl(a: tuple[torch.Tensor, torch.Tensor],
                                  b: tuple[torch.Tensor, torch.Tensor],
                                  d: torch.Tensor,
                                  masked_m: torch.Tensor,
                                  expected_m_per_group: int,
                                  mode) -> torch.Tensor:
    input, a_scale = a
    b_qweight, b_scale = b
    output = d
    op.m_grouped_marlin_w8a8_gemm_nt_masked(input, b_qweight, output, a_scale, b_scale,
                                        masked_m, expected_m_per_group, mode)
    return output

def m_grouped_bf16_gemm_nt_masked(a: torch.Tensor,
                                  b: torch.Tensor,
                                  d: torch.Tensor,
                                  masked_m: torch.Tensor,
                                  expected_m_per_group: int,
                                  config: Optional[Dict[str, Any]] = {"MODE": 1000,}) -> torch.Tensor:

    mode = config['MODE']
    op.m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group, mode)
    return d   

