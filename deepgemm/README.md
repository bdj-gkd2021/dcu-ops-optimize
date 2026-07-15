# deepgemm
## 简介
DeepGEMM是一个专为简洁高效的通用矩阵相乘（GEMM）而设计的库。它支持普通和混合专家（MoE）分组场景的FP8/INT8/BF16.

## 安装
环境依赖：
    pytorch2.0以上
    dtk24.04\dtk25.04\dtk26.04 


安装脚本：
    python3 setup.py bdist_wheel

## 算子介绍
### 核心算子

<table>
    <th>算子接口</th><th>算子介绍</th>
        <tr>
            <td> m_grouped_w4a8_gemm_nt_masked </td><td>input类型int8，权重为int4类型的group_gemm_mask，并且权重提前进行了marlin重排</td>
        </tr>
        <tr>
            <td> m_grouped_w8a8_gemm_nt_masked </td><td>input类型int8，权重为int8类型的group_gemm_mask，并且权重提前进行了marlin重排</td>
        </tr>
        <tr>
            <td> m_grouped_bf16_gemm_nt_masked </td><td>input类型bf16，权重为bf16类型的group_gemm_mask</td>
        </tr>
        <tr>
            <td> m_grouped_fp8_gemm_nt_masked </td><td>input类型fp8，权重为fp8类型的group_gemm_mask，并且权重提前进行了marlin重排, 仅支持ARCH>=GFX938</td>
        </tr>
        <tr>
            <td> m_grouped_i8_gemm_nt_contiguous </td><td>input类型int8，权重为int8类型的group_gemm_contiguous，并且权重提前进行了marlin重排</td>
        </tr>
        <tr>
            <td> m_grouped_i8_gemm_nt_contiguous_nopad </td><td>input类型int8，权重为int8类型的group_gemm_contiguous，并且权重提前进行了marlin重排, nopad是不对input进行填充，但是使用此接口需要对weight进行额外permute</td>
        </tr>
        <tr>
            <td> m_grouped_bf16_gemm_nt_contiguous </td><td>input类型bf16，权重为bf16类型的group_gemm_contiguous</td>
        </tr>
        <tr>
            <td> m_grouped_fp8_gemm_nt_contiguous </td><td>input类型fp8，权重为fp8类型的group_gemm_mask，并且权重提前进行了marlin重排, 仅支持ARCH>=GFX938</td>
        </tr>
        <tr>
            <td> mqa_logits </td><td>在不使用KV分页的情况下，为单个序列计算MQA logits</td>
        </tr>
        <tr>
            <td> paged_mqa_logits </td><td>使用分页键值缓存计算MQA logits</td>
        </tr>
        <tr>
            <td> get_paged_mqa_logits_metadata </td><td>为分页MQA logit构建调度元数据</td>
        </tr>
        <tr>
            <td> fp8_gemm_nt </td> <td> fp8 gemm nt布局， 参数(list[Tenosr(a), Tenosr(ascale)], list[Tenosr(b), Tenosr(bscale)], out)</td>
        </tr>
</table>

## 差异
与官方DeepGemm仓库中的相比：

1.增加int8类型支持

2.input与weight均使用perChannel量化

3.INT8/FP8的group类gemm,weight部分皆按照marlin layout进行重排 

4.contiguous接口input按照256的倍数对齐

5.k_group_contiguous kernel暂不支持

