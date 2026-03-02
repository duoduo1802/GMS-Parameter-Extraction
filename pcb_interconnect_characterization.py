"""
PCB 互连表征工具 (PCB Interconnect Characterization)
===================================================

简介:
    本脚本用于对印制电路板（PCB）互连线进行表征，主要包含两大核心功能：
    1. 夹具去嵌 (De-embedding)：调用 IEEE P370 标准的 AFR 算法，通过 2xThru 短线智能等分并切除测试夹具，提取纯 DUT 的 S 参数。
    2. GMS 参数提取：对去嵌后的迹线进行 S 矩阵和 T 矩阵分析，通过特征值分解计算通用模态 S 参数 (GMS)，获取纯走线的传输衰减和相位特性。

作者: duoduo1802
开源协议: MIT License
日期: 2026-03-02
"""

import argparse
import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import skrf as rf
from skrf.calibration.deembedding import IEEEP370_MM_NZC_2xThru, IEEEP370_SE_NZC_2xThru

# ==========================================
# 默认配置区域
# ==========================================
# 获取脚本所在目录，确保默认文件路径正确
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 如果未提供命令行参数，则使用以下默认值：

# 默认启用双线去嵌模式
DEFAULT_ENABLE_DEEMBBEDDING = True
# 默认开启 GMS 参数提取
DEFAULT_EXTRACT_GMS = True

# 默认文件路径
DEFAULT_SHORT = os.path.join(SCRIPT_DIR, 'PCIE_S_para_20inch_S.s4p')
DEFAULT_LONG = os.path.join(SCRIPT_DIR, 'PCIE_S_para_30inch_S.s4p')
DEFAULT_SINGLE = os.path.join(SCRIPT_DIR, 'PCIE_S_para_10inch_S.s4p')
# ==========================================


def setup_args():
    """配置并解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="PCB 互连表征与 GMS 提取工具"
    )
    
    # 模式选择
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--dual', action='store_true', help='启用双线去嵌模式 (默认)')
    mode_group.add_argument('--single', action='store_true', help='启用单线模式')

    # 文件路径配置
    parser.add_argument('--short', type=str, default=DEFAULT_SHORT, help='短线 (2xThru) .s4p 文件路径')
    parser.add_argument('--long', type=str, default=DEFAULT_LONG, help='长线 (Fixture-DUT-Fixture) .s4p 文件路径')
    parser.add_argument('--target', type=str, default=DEFAULT_SINGLE, help='单线 .s4p 文件路径 (仅用于单线模式)')
    
    parser.add_argument('--no-gms', action='store_true', help='禁用 GMS 参数提取 (仅执行去嵌操作)')
    parser.add_argument('--no-plot', action='store_true', help='禁用绘图')

    return parser.parse_args()


def load_network(file_path):
    """安全加载 Touchstone (.sNp) 文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    
    try:
        nw = rf.Network(file_path)
        print(f"  [已加载] {os.path.basename(file_path)} | 频点数: {len(nw.f)} | 端口数: {nw.nports}")
        return nw
    except Exception as e:
        raise ValueError(f"加载文件 {file_path} 失败。错误信息: {e}")


def convert_to_diff_sdd(nw):
    """
    将 4 端口网络转换为纯差分 Sdd 参数矩阵。
    处理端口映射：将可能存在的非标准映射 (输入[1,3] -> 输出[2,4]) 
    转换为 skrf 期望的标准混合模端口排列。
    """
    if nw.nports == 2:
        return nw.s
    elif nw.nports == 4:
        # P: 1(0)-2(1), N: 3(2)-4(3) -> 转换为 skrf 标准映射: In(0,2)->Diff1, Out(1,3)->Diff2
        # 利用置换矩阵 [0, 2, 1, 3] 完成用户映射到 skrf 的标准映射转换
        perm = [0, 2, 1, 3]
        s_orig = nw.s
        s_perm = s_orig[:, perm, :][:, :, perm]
        
        nw_perm = rf.Network(frequency=nw.frequency, s=s_perm)
        nw_perm.se2gmm(p=2) # 转换为混合模 S 参数
        # 提取第一个 2x2 块，即 Sdd (差分到差分)
        return nw_perm.s[:, 0:2, 0:2]
    else:
        raise ValueError(f"不支持的端口数: {nw.nports}。仅支持 2 和 4 端口。")


def calculate_eigenvalues(t_matrix_series):
    """
    计算 T 矩阵序列的特征值以提取传播常数
    """
    num_points = len(t_matrix_series)
    lambda_forward = np.zeros(num_points, dtype=complex)
    
    for i in range(num_points):
        T = t_matrix_series[i]
        try:
            eigenvalues = np.linalg.eigvals(T)
            # 对于无源传输线，选取幅值 <= 1 的特征值代表正向衰减传输特性 (abs(e^-gamma*L) <= 1)
            # 通过幅值排序通常可正确处理被动物理线路
            if np.abs(eigenvalues[0]) < np.abs(eigenvalues[1]):
                lambda_forward[i] = eigenvalues[0]
            else:
                lambda_forward[i] = eigenvalues[1]
        except np.linalg.LinAlgError:
            lambda_forward[i] = np.nan + 1j * np.nan
            
    return lambda_forward


def perform_afr_deembedding(file_short_path, file_long_path, script_dir, base_name):
    """
    使用 IEEE P370 AFR (自动夹具移除) 算法进行去嵌。
    算法通过切分短线 (2xThru) 得出左右两侧夹具模型，将这部分附加响应
    从长线两端去嵌剥离，得出纯被测走线 (DUT) 的网络参数。
    """
    print(f"--- 夹具去嵌 (AFR) ---")
    print(f"  短线(夹具): {file_short_path}")
    print(f"  长线(带夹具对象): {file_long_path}")
    
    nw_short = load_network(file_short_path)
    nw_long = load_network(file_long_path)
    
    if not np.array_equal(nw_short.f, nw_long.f):
        raise ValueError("短线和长线的频率范围不匹配。")
    
    if nw_short.nports == 4:
        dm_afr = IEEEP370_MM_NZC_2xThru(dummy_2xthru=nw_short, name='2xthru', port_order='first')
        
        left_fix_path = os.path.join(script_dir, f"AFR_{base_name}_short_half_left.s4p")
        right_fix_path = os.path.join(script_dir, f"AFR_{base_name}_short_half_right.s4p")
        dm_afr.se_side1.write_touchstone(left_fix_path)
        dm_afr.se_side2.write_touchstone(right_fix_path)
        print(f"  [AFR去嵌] 左侧一半夹具存至:\n  -> {left_fix_path}")
        print(f"  [AFR去嵌] 右侧一半夹具存至:\n  -> {right_fix_path}")
        
        nw_dut = dm_afr.deembed(nw_long)
        
    elif nw_short.nports == 2:
        dm_afr = IEEEP370_SE_NZC_2xThru(dummy_2xthru=nw_short, name='2xthru')
        
        left_fix_path = os.path.join(script_dir, f"AFR_{base_name}_short_half_left.s2p")
        right_fix_path = os.path.join(script_dir, f"AFR_{base_name}_short_half_right.s2p")
        dm_afr.s_side1.write_touchstone(left_fix_path)
        dm_afr.s_side2.write_touchstone(right_fix_path)
        print(f"  [AFR去嵌] 左侧一半单端夹具保存于:\n  -> {left_fix_path}")
        print(f"  [AFR去嵌] 右侧一半单端夹具保存于:\n  -> {right_fix_path}")
        
        nw_dut = dm_afr.deembed(nw_long)
        
    else:
        raise ValueError(f"AFR 计算遇到不支持端口总数: {nw_short.nports}")

    port_ext = f".s{nw_dut.nports}p"
    deembedded_path = os.path.join(script_dir, f"AFR_{base_name}_deembedded_DUT{port_ext}")
    nw_dut.write_touchstone(deembedded_path)
    print(f"  [AFR去嵌] 去嵌后纯 DUT S参数已落盘:\n  -> {deembedded_path}\n")
    
    return nw_dut


def extract_gms_from_network(nw):
    """
    通用 GMS 参数提取逻辑，输入为一个 rf.Network 对象 (纯DUT)
    """
    print(f"--- 开始 GMS 参数提取 ---")
    # 1. 提取差分 Sdd (S 参数矩阵)
    s_mat = convert_to_diff_sdd(nw)
    
    # 2. S 矩阵转化为 T 矩阵 (传递矩阵)
    t_mat = rf.s2t(s_mat)
    
    # 3. 求解特征值得到 GMS (通用模态 S 参数)
    lambda_forward = calculate_eigenvalues(t_mat)
    
    return nw.f, lambda_forward


def save_results(script_dir, freq, lambda_forward, base_name, suffix):
    """保存提取的 GMS 结果到 .s4p 文件"""
    # 构造 GMS S 矩阵 (对于理想差分网络: S21=S12=S34=S43=lambda, 其他=0)
    num_points = len(freq)
    s_gms = np.zeros((num_points, 4, 4), dtype=complex)
    
    # 填充正反向传输参数映射
    # 矩阵下标对应: S21(1,0), S12(0,1), S43(3,2), S34(2,3)
    s_gms[:, 1, 0] = lambda_forward
    s_gms[:, 0, 1] = lambda_forward
    s_gms[:, 3, 2] = lambda_forward
    s_gms[:, 2, 3] = lambda_forward
    
    nw_gms = rf.Network(frequency=rf.Frequency.from_f(freq, unit='hz'), s=s_gms)
    
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{timestamp_str}_{base_name}_{suffix}.s4p"
    output_path = os.path.join(script_dir, output_filename)
    
    nw_gms.write_touchstone(output_path)
    print(f"\n[成功] GMS 参数已保存至:\n  -> {output_path}")
    return output_path


def plot_results(freq, lambda_forward, title_suffix):
    """绘制 GMS 的幅度与相位响应曲线"""
    mag_db = 20 * np.log10(np.abs(lambda_forward) + 1e-15)
    # 相位解卷绕并控制在 0~360 度之间
    phase_rad = np.unwrap(np.angle(lambda_forward))
    phase_deg = np.degrees(phase_rad) % 360
    
    freq_ghz = freq / 1e9
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(freq_ghz, mag_db, label='GMS S21', color='#1f77b4')
    ax1.set_title(f'GMS S21 幅度响应 / Magnitude ({title_suffix})')
    ax1.set_ylabel('幅度 / Magnitude (dB)')
    ax1.set_xlabel('频率 / Frequency (GHz)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(freq_ghz, phase_deg, label='Phase', color='#d62728')
    ax2.set_title(f'GMS S21 相位响应 / Phase ({title_suffix})')
    ax2.set_ylabel('相位 / Phase (Degrees)')
    ax2.set_xlabel('频率 / Frequency (GHz)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    print("正在显示绘图... (关闭窗口以退出)")
    plt.show()


def main():
    args = setup_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 确定运行模式逻辑
    if args.dual:
        use_deembed_mode = True
    elif args.single:
        use_deembed_mode = False
    else:
        # 若命令行没有指定，回退至全局默认状态
        use_deembed_mode = DEFAULT_ENABLE_DEEMBBEDDING
        
    # 确定是否提取 GMS
    extract_gms_flag = not args.no_gms if args.no_gms else DEFAULT_EXTRACT_GMS

    try:
        lambda_val = None
        freq = None
        base_name = ""
        mode_label = ""
        
        if use_deembed_mode:
            file_short = os.path.abspath(args.short)
            file_long = os.path.abspath(args.long)
            base_name = os.path.splitext(os.path.basename(file_long))[0]
            
            # 第一步：执行 AFR 去嵌
            nw_dut = perform_afr_deembedding(file_short, file_long, script_dir, base_name)
            
            # 第二步：检查是否提取 GMS
            if extract_gms_flag:
                freq, lambda_val = extract_gms_from_network(nw_dut)
                mode_label = "afr_deembedded_gms"
            else:
                print("\n[提示] 夹具去嵌已顺利执行。由于配置禁用，跳过 GMS 参数提取阶段。")
                
        else:
            # 单线模式：直接提取 GMS
            file_single = os.path.abspath(args.target)
            print(f"--- 单线模式: {file_single} ---")
            base_name = os.path.splitext(os.path.basename(file_single))[0]
            nw_single = load_network(file_single)
            
            if extract_gms_flag:
                freq, lambda_val = extract_gms_from_network(nw_single)
                mode_label = "single_gms"
            else:
                print("\n[提示] 当前为单线模式，且禁用了 GMS 提取，脚本不执行任何操作。")

        # 第三步：保存和绘图 (如果有提取到 GMS 参数)
        if lambda_val is not None:
            save_results(script_dir, freq, lambda_val, base_name, mode_label)
            if not args.no_plot:
                plot_results(freq, lambda_val, mode_label)
            
    except Exception as e:
        # 为排错保留完整的追踪路径，同时输出显式的错误提示
        import traceback
        traceback.print_exc()
        print(f"\n[错误] 流程未能走通，执行失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
