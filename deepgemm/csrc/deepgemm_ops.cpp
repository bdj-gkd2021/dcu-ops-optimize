// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "asm_m_grouped_w8a8_gemm_nt_contiguous.h"
#include "asm_m_grouped_w16a16_gemm_nt_contiguous.h"
#include "m_grouped_gemm_nt_masked.h"
#include "mqa_logits.h"
#include "paged_mqa_logits.h"
#include "gemm.h"
#include "tf32_hc_pernorm_gemm.h"
#include <torch/extension.h>

#include "deepgemm_ops.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    M_GROUP_GEMM_NT_CONTIGUOUS_PYBIND;
    M_GROUP_GEMM_NT_MASKED_PYBIND;
    MQA_LOGITS_PYBIND;
    PAGED_MQA_LOGITS_PYBIND;
    GEMM_NT;
    TF32_HC_PERNORM_GEMM_PYBIND;
}
