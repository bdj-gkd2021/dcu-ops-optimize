#define M_GROUP_GEMM_NT_CONTIGUOUS_PYBIND \
    m.def("m_grouped_w8a8_gemm_nt_nopad_contiguous", &deepgemm::m_grouped_w8a8_gemm_nt_nopad_contiguous, "m_grouped_w8a8_gemm_nt_nopad_contiguous", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("token_per_expert")); \
    m.def("m_grouped_w8a8_gemm_nt_contiguous", &deepgemm::m_grouped_w8a8_gemm_nt_contiguous, "m_grouped_w8a8_gemm_nt_contiguous", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("mode")); \
    m.def("m_grouped_w16a16_gemm_nt_contiguous", &deepgemm::m_grouped_w16a16_gemm_nt_contiguous, "m_grouped_w16a16_gemm_nt_contiguous", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("mode"));  \
    m.def("m_grouped_fp8_gemm_nt_contiguous", &deepgemm::m_grouped_fp8_gemm_nt_contiguous, "m_grouped_fp8_gemm_nt_contiguous", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("m_indices"), \
        py::arg("mode"));  
    
#define M_GROUP_GEMM_NT_MASKED_PYBIND \
    m.def("m_grouped_marlin_w4a8_gemm_nt_masked", &deepgemm::m_grouped_marlin_w4a8_gemm_nt_masked, "m_grouped_marlin_w4a8_gemm_nt_masked", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("masked_m"), \
        py::arg("expected_m_per_group"), \
        py::arg("mode")); \
    m.def("m_grouped_marlin_w8a8_gemm_nt_masked", &deepgemm::m_grouped_marlin_w8a8_gemm_nt_masked, "m_grouped_marlin_w8a8_gemm_nt_masked", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("masked_m"), \
        py::arg("expected_m_per_group"), \
        py::arg("mode")); \
    m.def("m_grouped_marlin_fp8_gemm_nt_masked", &deepgemm::m_grouped_marlin_fp8_gemm_nt_masked, "m_grouped_marlin_fp8_gemm_nt_masked", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("a_scale"), \
        py::arg("b_scale"), \
        py::arg("masked_m"), \
        py::arg("expected_m_per_group"), \
        py::arg("enable_overlap"), \
        py::arg("signal"), \
        py::arg("mode")); \
    m.def("m_grouped_bf16_gemm_nt_masked", &deepgemm::m_grouped_bf16_gemm_nt_masked, "m_grouped_bf16_gemm_nt_masked", \
        py::arg("input"), \
        py::arg("b_qweight"), \
        py::arg("output"), \
        py::arg("masked_m"), \
        py::arg("expected_m_per_group"), \
        py::arg("mode"));   
#define MQA_LOGITS_PYBIND \
    m.def("mqa_logits", &deepgemm::mqa_logits, "mqa_logits", \
        py::arg("Q"), \
        py::arg("K"), \
        py::arg("Weights"), \
        py::arg("KS"), \
        py::arg("KE"), \
        py::arg("q_seq_len"), \
        py::arg("kv_seq_len"), \
        py::arg("num_heads"), \
        py::arg("head_dim"), \
        py::arg("KV_scale") = py::none(), \
        py::arg("clean_logits") = true, \
        py::arg("D_out") = py::none());
#define PAGED_MQA_LOGITS_PYBIND \
    m.def("get_paged_mqa_logits_metadata", &deepgemm::get_paged_mqa_logits_metadata, "get_paged_mqa_logits_metadata", \
        py::arg("context_lens"), \
        py::arg("block_kv"), \
        py::arg("num_sms")); \
    m.def("paged_mqa_logits", &deepgemm::paged_mqa_logits, "paged_mqa_logits", \
        py::arg("q"), \
        py::arg("fused_kv_cache"), \
        py::arg("weights"), \
        py::arg("context_lens"), \
        py::arg("block_table"), \
        py::arg("schedule_meta"), \
        py::arg("max_context_len"), \
        py::arg("clean_logits"));

#define GEMM_NT  \
    m.def("fp8_gemm", &deepgemm::fp8_gemm, "fp8_gemm", \
        py::arg("mat_a"), \
        py::arg("mat_b"), \
        py::arg("scale_a"), \
        py::arg("scale_b"), \
        py::arg("output"), \
        py::arg("m"),  \
        py::arg("n"),  \
        py::arg("k"),  \
        py::arg("batch"),  \
        py::arg("transpose_flag"), \
        py::arg("alpha"), \
        py::arg("beta"),  \
        py::arg("bias"));
#define TF32_HC_PERNORM_GEMM_PYBIND \
    m.def("tf32_hc_pernorm_gemm", &deepgemm::tf32_hc_pernorm_gemm, "tf32_hc_pernorm_gemm", \
        py::arg("A"), \
        py::arg("B"), \
        py::arg("D"), \
        py::arg("sqr_sum"), \
        py::arg("num_splits"));