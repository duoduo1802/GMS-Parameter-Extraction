"""
GMS Extraction Tool (Generalized Modal S-parameters)
====================================================

Description:
    This script implements the Generalized Modal S-parameters (GMS) extraction method.
    It supports two modes:
    1. Dual Line De-embedding: Extracts GMS from two transmission lines (Short & Long).
       Method: M = T_long * inv(T_short) -> Eigenvalue Decomposition
    2. Single Line Mode: Analyzes a single transmission line (mostly for verification).
       Method: Eigenvalue Decomposition of T-matrix

    本脚本实现了通用模态 S 参数 (GMS) 提取方法。
    支持两种模式：
    1. 双线去嵌模式：从两条传输线（短线和长线）提取 GMS。
       原理：M = T_long * inv(T_short) -> 特征值分解
    2. 单线模式：分析单条传输线（主要用于验证）。
       原理：T 矩阵的特征值分解

Author: [Your Name/GitHub Username]
License: MIT License
Date: 2026-02-26
"""

import argparse
import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import skrf as rf

# ==========================================
# 默认配置区域 / Default Configuration
# ==========================================
# 获取脚本所在目录，确保默认文件路径正确
# Get script directory to ensure default file paths are correct
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 如果不使用命令行参数，将使用以下默认值
# If command line arguments are not provided, these defaults are used.

# 默认启用双线去嵌模式 / Default to Dual Line De-embedding
DEFAULT_ENABLE_DEEMBBEDDING = False

# 默认文件名 / Default Filenames
DEFAULT_SHORT = os.path.join(SCRIPT_DIR, 'PCIE_S_para_10inch_S.s4p')
DEFAULT_LONG = os.path.join(SCRIPT_DIR, 'PCIE_S_para_30inch_S.s4p')
DEFAULT_SINGLE = os.path.join(SCRIPT_DIR, 'PCIE_S_para_10inch_S.s4p')
# ==========================================


def setup_args():
    """配置命令行参数 / Setup command line arguments"""
    parser = argparse.ArgumentParser(
        description="GMS Parameters Extraction Tool / GMS 参数提取工具"
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--dual', action='store_true', help='Enable Dual Line De-embedding mode (Default if files provided)')
    mode_group.add_argument('--single', action='store_true', help='Enable Single Line mode')

    # File paths
    parser.add_argument('--short', type=str, default=DEFAULT_SHORT, help='Path to Short Line .s4p file')
    parser.add_argument('--long', type=str, default=DEFAULT_LONG, help='Path to Long Line .s4p file')
    parser.add_argument('--target', type=str, default=DEFAULT_SINGLE, help='Path to Single Line .s4p file (for single mode)')
    
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting / 禁用绘图')

    return parser.parse_args()


def load_network(file_path):
    """安全加载 Touchstone 文件 / Safely load Touchstone file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        nw = rf.Network(file_path)
        print(f"  [LOADED] {os.path.basename(file_path)} | Points: {len(nw.f)} | Ports: {nw.nports}")
        return nw
    except Exception as e:
        raise ValueError(f"Failed to load {file_path}. Error: {e}")


def convert_to_diff_sdd(nw):
    """
    将 4 端口网络转换为差分 Sdd 参数。
    Handle Single-Ended vs. Differential logic.
    Assumed Port Mapping: P:1,3 (Input) -> P:2,4 (Output) which is non-standard.
    We convert it to standard skrf expectation.
    """
    if nw.nports == 2:
        return nw.s
    elif nw.nports == 4:
        # P: 1(0)-2(1), N: 3(2)-4(3) -> Standard: In(0,2)->Diff1, Out(1,3)->Diff2
        # Permutation to [0, 2, 1, 3] converts user mapping to skrf expected mapping
        perm = [0, 2, 1, 3]
        s_orig = nw.s
        s_perm = s_orig[:, perm, :][:, :, perm]
        
        nw_perm = rf.Network(frequency=nw.frequency, s=s_perm)
        nw_perm.se2gmm(p=2)
        # Extract Sdd (differential-to-differential) which is the first 2x2 block
        return nw_perm.s[:, 0:2, 0:2]
    else:
        raise ValueError(f"Unsupported number of ports: {nw.nports}. Only 2 and 4 supported.")


def calculate_eigenvalues(t_matrix_series):
    """
    计算 T 矩阵序列的特征值
    Compute eigenvalues for a series of T-matrices
    """
    num_points = len(t_matrix_series)
    lambda_forward = np.zeros(num_points, dtype=complex)
    
    for i in range(num_points):
        T = t_matrix_series[i]
        try:
            eigenvalues = np.linalg.eigvals(T)
            # Select eigenvalue with magnitude <= 1 (passive forward transmission)
            # sorting by magnitude usually works for passive lines
            # abs(e^-gamma*L) <= 1
            if np.abs(eigenvalues[0]) < np.abs(eigenvalues[1]):
                lambda_forward[i] = eigenvalues[0]
            else:
                lambda_forward[i] = eigenvalues[1]
        except np.linalg.LinAlgError:
            lambda_forward[i] = np.nan + 1j * np.nan
            
    return lambda_forward


def extract_gms_single(file_path):
    """Single Line Extraction Logic"""
    print(f"--- Single Line Mode: {file_path} ---")
    nw = load_network(file_path)
    freq = nw.f
    
    # 1. Get Sdd
    s_mat = convert_to_diff_sdd(nw)
    
    # 2. S -> T
    t_mat = rf.s2t(s_mat)
    
    # 3. Eigenvalues
    lambda_forward = calculate_eigenvalues(t_mat)
    
    return freq, lambda_forward


def extract_gms_dual(file_short_path, file_long_path):
    """Dual Line De-embedding Logic"""
    print(f"--- Dual Line Mode ---\n  Short: {file_short_path}\n  Long:  {file_long_path}")
    nw_short = load_network(file_short_path)
    nw_long = load_network(file_long_path)
    
    if not np.array_equal(nw_short.f, nw_long.f):
        raise ValueError("Frequency mismatch between Short and Long files.")
    
    freq = nw_short.f
    
    # 1. Get Sdd
    s_short = convert_to_diff_sdd(nw_short)
    s_long = convert_to_diff_sdd(nw_long)
    
    # 2. S -> T
    t_short = rf.s2t(s_short)
    t_long = rf.s2t(s_long)
    
    # 3. De-embedding (M = T_long * inv(T_short))
    num_points = len(freq)
    m_series = np.zeros_like(t_long)
    
    for i in range(num_points):
        try:
            t1_inv = np.linalg.inv(t_short[i])
            m_series[i] = np.matmul(t_long[i], t1_inv)
        except np.linalg.LinAlgError:
             # Identity matrix as fallback or NaN
            m_series[i] = np.eye(2) 
            
    # 4. Eigenvalues of M
    lambda_forward = calculate_eigenvalues(m_series)
    
    return freq, lambda_forward


def save_results(script_dir, freq, lambda_forward, base_name, suffix):
    """Save results to .s4p file"""
    # Create GMS S-matrix (S21=S12=S34=S43=lambda, others=0)
    num_points = len(freq)
    s_gms = np.zeros((num_points, 4, 4), dtype=complex)
    
    # Map lambda to transmission parameters
    # S21 (1,0), S12 (0,1), S43 (3,2), S34 (2,3)
    s_gms[:, 1, 0] = lambda_forward
    s_gms[:, 0, 1] = lambda_forward
    s_gms[:, 3, 2] = lambda_forward
    s_gms[:, 2, 3] = lambda_forward
    
    nw_gms = rf.Network(frequency=rf.Frequency.from_f(freq, unit='hz'), s=s_gms)
    
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{timestamp_str}_{base_name}_{suffix}.s4p"
    output_path = os.path.join(script_dir, output_filename)
    
    nw_gms.write_touchstone(output_path)
    print(f"\n[SUCCESS] GMS parameters saved to:\n  -> {output_path}")
    return output_path


def plot_results(freq, lambda_forward, title_suffix):
    """Plot Magnitude and Phase"""
    mag_db = 20 * np.log10(np.abs(lambda_forward) + 1e-15)
    # Phase unwrapping and wrapping to 360
    phase_rad = np.unwrap(np.angle(lambda_forward))
    phase_deg = np.degrees(phase_rad) % 360
    
    freq_ghz = freq / 1e9
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(freq_ghz, mag_db, label='GMS S21', color='#1f77b4')
    ax1.set_title(f'GMS S21 Magnitude ({title_suffix})')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(freq_ghz, phase_deg, label='Phase', color='#d62728')
    ax2.set_title(f'GMS S21 Phase ({title_suffix})')
    ax2.set_ylabel('Phase (Degrees)')
    ax2.set_xlabel('Frequency (GHz)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    print("Displaying plot... (Close window to exit)")
    plt.show()


def main():
    args = setup_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine mode logic
    # Priority: Command line flags > Default Global Variable
    if args.dual:
        use_dual_mode = True
    elif args.single:
        use_dual_mode = False
    else:
        # Fallback to variable if no flag specified
        # If files for dual mode are provided via args, prefer dual mode?
        # Actually, let's stick to the explicit flag or global default
        use_dual_mode = DEFAULT_ENABLE_DEEMBBEDDING

    try:
        if use_dual_mode:
            file_short = os.path.abspath(args.short)
            file_long = os.path.abspath(args.long)
            
            # Allow fallback to default if args are default but files don't exist?
            # No, keep it strict to what is provided/defaulted.
            
            base_name = os.path.splitext(os.path.basename(file_long))[0]
            
            freq, lambda_val = extract_gms_dual(file_short, file_long)
            mode_label = "diffline_deembedded"
        else:
            file_single = os.path.abspath(args.target)
            base_name = os.path.splitext(os.path.basename(file_single))[0]
            
            freq, lambda_val = extract_gms_single(file_single)
            mode_label = "diffline_single"
            
        save_results(script_dir, freq, lambda_val, base_name, mode_label)
        
        if not args.no_plot:
            plot_results(freq, lambda_val, mode_label)
            
    except Exception as e:
        # Print full traceback for debugging if needed, but keep it clean for users
        import traceback
        traceback.print_exc()
        print(f"\n[ERROR] Execution failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
