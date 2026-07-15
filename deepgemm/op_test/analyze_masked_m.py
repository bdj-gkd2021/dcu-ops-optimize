#!/usr/bin/env python3
"""
分析日志文件中的 masked_m 数据分布
"""

import re
from collections import Counter
import numpy as np

def parse_masked_m_from_log(log_file, max_lines=100000):
    """
    从日志文件中解析 masked_m tensor 数据
    
    Args:
        log_file: 日志文件路径
        max_lines: 最大读取行数（避免读取过大文件）
    
    Returns:
        所有 masked_m 中的元素值列表
    """
    all_values = []
    
    # 匹配 tensor([...]) 的正则表达式
    # 支持跨行的tensor，如：
    # tensor([ 0,  0,  0, 29, 29,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    #        device='cuda:2', dtype=torch.int32)
    pattern = r'masked_m:.*?tensor\(\[([\d\s,\-]+)\]'
    
    buffer = ""
    line_count = 0
    
    with open(log_file, 'r') as f:
        for line in f:
            line_count += 1
            if line_count > max_lines:
                break
            
            buffer += line
            
            # 尝试匹配完整的tensor
            matches = re.findall(pattern, buffer, re.DOTALL)
            for match in matches:
                # 解析数值
                values = [int(v.strip()) for v in match.split(',') if v.strip()]
                all_values.extend(values)
            
            # 如果找到了匹配，清空已匹配的部分
            if matches:
                # 保留最后一部分以防跨行
                last_masked_idx = buffer.rfind('masked_m:')
                if last_masked_idx != -1 and 'dtype=torch.int32)' not in buffer[last_masked_idx:]:
                    buffer = buffer[last_masked_idx:]
                else:
                    buffer = ""
    
    return all_values

def analyze_distribution(values):
    """
    分析数据分布
    
    Args:
        values: 数值列表
    """
    if not values:
        print("没有找到 masked_m 数据！")
        return
    
    values = np.array(values)
    
    print("=" * 60)
    print("masked_m 数据分析报告")
    print("=" * 60)
    
    # 基本统计
    print(f"\n【基本统计】")
    print(f"  总元素数量: {len(values)}")
    print(f"  最小值: {values.min()}")
    print(f"  最大值: {values.max()}")
    print(f"  平均值: {values.mean():.4f}")
    print(f"  中位数: {np.median(values):.4f}")
    print(f"  标准差: {values.std():.4f}")
    
    # 零值统计
    zero_count = np.sum(values == 0)
    zero_ratio = zero_count / len(values) * 100
    print(f"\n【零值统计】")
    print(f"  零值数量: {zero_count}")
    print(f"  零值占比: {zero_ratio:.2f}%")
    
    # 非零值统计
    non_zero_values = values[values != 0]
    if len(non_zero_values) > 0:
        print(f"\n【非零值统计】")
        print(f"  非零值数量: {len(non_zero_values)}")
        print(f"  非零值占比: {100 - zero_ratio:.2f}%")
        print(f"  非零值最小: {non_zero_values.min()}")
        print(f"  非零值最大: {non_zero_values.max()}")
        print(f"  非零值平均: {non_zero_values.mean():.4f}")
    
    # 频率分布
    counter = Counter(values)
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n【元素值频率分布】(按频率降序排列)")
    print(f"  {'值':<10} {'出现次数':<15} {'占比':<10}")
    print(f"  {'-'*35}")
    
    for value, count in sorted_items[:30]:  # 显示前30个
        ratio = count / len(values) * 100
        print(f"  {value:<10} {count:<15} {ratio:.2f}%")
    
    if len(sorted_items) > 30:
        print(f"  ... (共 {len(sorted_items)} 种不同的值)")
    
    # 分段统计
    print(f"\n【区间分布统计】")
    bins = [0, 1, 5, 10, 20, 50, 100, 200, 500, 1000, float('inf')]
    bin_labels = ['0', '1-4', '5-9', '10-19', '20-49', '50-99', '100-199', '200-499', '500-999', '>=1000']
    
    for i in range(len(bins) - 1):
        if i == 0:
            count = np.sum(values == 0)
        else:
            count = np.sum((values >= bins[i]) & (values < bins[i+1]))
        if count > 0:
            ratio = count / len(values) * 100
            print(f"  {bin_labels[i]:<12}: {count:<10} ({ratio:.2f}%)")

def main():
    log_file = "/public/home/yangyn1/work_space/2025/NMZ/deepgemm/op_test/DeepSeek-R1-Channel-INT8-tp16-dp16-ep16-node122-model_len6000-mem0.9-0117-0346-.log"
    
    print(f"正在读取日志文件: {log_file}")
    print(f"(仅读取前 100000 行)")
    
    # 解析数据
    values = parse_masked_m_from_log(log_file, max_lines=6000000)
    
    # 分析数据
    analyze_distribution(values)

if __name__ == "__main__":
    main()
