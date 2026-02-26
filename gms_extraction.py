import numpy as np
import skrf as rf
import matplotlib.pyplot as plt
import os
import datetime

# ==========================================
# 用户配置区域 / User Configuration
# ==========================================
# 模式选择 / Mode Selection
# True: 双线去嵌模式 (Dual Line De-embedding)
# False: 单线模式 (Single Line Mode)
ENABLE_DEEMBEDDING = False

# 输入文件名 / Input Filenames
# 如果开启去嵌模式，请设置 SHORT 和 LONG 文件名
SHORT_LINE_FILENAME = 'PCIE_S_para_10inch_S.s4p'
LONG_LINE_FILENAME = 'PCIE_S_para_30inch_S.s4p'

# 如果关闭去嵌模式 (单线模式)，请设置 SINGLE 文件名
SINGLE_LINE_FILENAME = 'demo_channel.s4p'
# ==========================================

def extract_gms_single(file_path):
    """
    Extracts GMS parameters from a single transmission line file.
    Does not perform de-embedding (M = T).
    """
    # 1. Read Touchstone file
    print(f"Loading Single file: {file_path}")
    try:
        nw = rf.Network(file_path)
        print(f"  File loaded. Points: {len(nw.f)}, Ports: {nw.nports}")
    except Exception as e:
        raise ValueError(f"Failed to load Touchstone file. Error: {e}")
        
    freq = nw.f
    num_points = len(freq)

    # 2. Handle Single-Ended vs. Differential
    if nw.nports == 2:
        s_mat = nw.s
    elif nw.nports == 4:
        # Create permuted S-matrix manually for proper Differential Mode extraction
        # P: 1(0)-2(1), N: 3(2)-4(3) -> Standard: In(0,2)->Diff1, Out(1,3)->Diff2
        s_orig = nw.s
        perm = [0, 2, 1, 3]
        s_perm = s_orig[:, perm, :][:, :, perm]
        nw_perm = rf.Network(frequency=nw.frequency, s=s_perm)
        nw_perm.se2gmm(p=2)
        s_mat = nw_perm.s[:, 0:2, 0:2] # Extract Sdd
    else:
        raise ValueError("Unsupported number of ports.")

    # 3. S to T
    t_mat = rf.s2t(s_mat)

    # 4. GMS Extraction (Eigenvalues of T)
    # Direct eigenvalue decomposition of T-matrix for single line
    lambda_forward = np.zeros(num_points, dtype=complex)
    
    for i in range(num_points):
        T = t_mat[i]
        try:
            eigenvalues = np.linalg.eigvals(T)
            # Select eigenvalue with magnitude <= 1 (passive)
            # We assume the line is passive
            if np.abs(eigenvalues[0]) < np.abs(eigenvalues[1]):
                lambda_forward[i] = eigenvalues[0]
            else:
                lambda_forward[i] = eigenvalues[1]
        except np.linalg.LinAlgError:
            lambda_forward[i] = np.nan + 1j * np.nan
            
    return freq, lambda_forward


def extract_gms(file_short, file_long):
    """
    Extracts the Generalized Modal S-parameters (GMS) representing the pure 
    propagation characteristics of a transmission line difference.
    
    Args:
        file_short (str): Path to the short transmission line Touchstone file.
        file_long (str): Path to the long transmission line Touchstone file.
        
    Returns:
        tuple: (frequency_axis_in_Hz, lambda_forward_array)
    """
    # 1. Read Touchstone files
    print(f"Loading files:\n  Short: {file_short}\n  Long:  {file_long}")
    try:
        nw_short = rf.Network(file_short)
        print(f"  Short line loaded. Points: {len(nw_short.f)}, Ports: {nw_short.nports}")
        nw_long = rf.Network(file_long)
        print(f"  Long line loaded. Points: {len(nw_long.f)}, Ports: {nw_long.nports}")
    except Exception as e:
        raise ValueError(f"Failed to load Touchstone files. Error: {e}")
    
    # Check if frequency points match exactly
    if not np.array_equal(nw_short.f, nw_long.f):
        raise ValueError("The frequency points of the two files do not match exactly.")
    
    freq = nw_short.f
    num_points = len(freq)
    
    # 2. Handle Single-Ended vs. Differential
    if nw_short.nports == 2 and nw_long.nports == 2:
        # 2-port (Single-ended): Keep the 2x2 S-parameter matrix as is
        s_short = nw_short.s
        s_long = nw_long.s
    elif nw_short.nports == 4 and nw_long.nports == 4:
        # 4-port (Differential): Convert to mixed-mode and extract Sdd
        # The user specified port mapping:
        # P: 1 -> 2
        # N: 3 -> 4
        # This means ports 1,3 are input (P,N) and ports 2,4 are output (P,N)
        # skrf se2gmm default expects ports 1,2 to be diff pair 1 and 3,4 to be diff pair 2.
        # We need to renumber the ports so that:
        # New Port 1 = Old Port 1 (Input P)
        # New Port 2 = Old Port 3 (Input N)
        # New Port 3 = Old Port 2 (Output P)
        # New Port 4 = Old Port 4 (Output N)
        
        # skrf.Network.renumber(from_ports, to_ports)
        # We want to reorder ports to match se2gmm expectation.
        # se2gmm(p=2) expects:
        #  - Diff Port 1 composed of (NewPort0, NewPort1)
        #  - Diff Port 2 composed of (NewPort2, NewPort3)
        # User config:
        #  - Input Diff Pair = (OldPort0, OldPort2)  [User said P:1, N:3]
        #  - Output Diff Pair = (OldPort1, OldPort3) [User said P:2, N:4]
        
        # So we want the new port order to be: [Old0, Old2, Old1, Old3]
        # In skrf.renumber, the signatures are tricky.
        # It's better to use subset indexing which acts as reordering.
        # new_ntwk = ntwk.subnetwork([0, 2, 1, 3])  <-- This might be cleaner if supported, 
        # but skrf usually uses default indexing for subsetting.
        
        # Let's use the explicit renumber method which swaps indices.
        # Actually, creating a new network from s-parameters with permuted indices is safer 
        # to ensure we don't misunderstand 'renumber'.
        
        # Original S-matrix shape: (freq, 4, 4)
        # We want to shuffle rows and columns.
        # New order of indices: [0, 2, 1, 3]
        # 0 -> 0
        # 1 -> 2
        # 2 -> 1
        # 3 -> 3
        
        # Let's implement this manually on the s-matrix to be absolutely sure.
        s_short_orig = nw_short.s
        s_long_orig = nw_long.s
        
        # Permutation vector
        perm = [0, 2, 1, 3]
        
        # Reorder s_short
        s_short_perm = s_short_orig[:, perm, :][:, :, perm]
        # Reorder s_long
        s_long_perm = s_long_orig[:, perm, :][:, :, perm]
        
        # Create temporary networks for conversion
        nw_short_perm = rf.Network(frequency=nw_short.frequency, s=s_short_perm)
        nw_long_perm = rf.Network(frequency=nw_long.frequency, s=s_long_perm)
        
        # Now convert to mixed mode
        nw_short_perm.se2gmm(p=2)
        nw_long_perm.se2gmm(p=2)
        
        # Extract Sdd (differential-to-differential) which is the first 2x2 block
        s_short = nw_short_perm.s[:, 0:2, 0:2]
        s_long = nw_long_perm.s[:, 0:2, 0:2]
    else:
        raise ValueError("Unsupported number of ports. Must be both 2-port or both 4-port.")
        
    # 3. S-parameters to T-parameters Conversion
    # Convert the 2x2 S-parameter matrix to 2x2 T-parameters
    t_short = rf.s2t(s_short)
    t_long = rf.s2t(s_long)
    
    # 4. The Core Patent Algorithm (Matrix Math & Eigenvalues)
    lambda_forward = np.zeros(num_points, dtype=complex)
    
    for i in range(num_points):
        T1 = t_short[i]
        T2 = t_long[i]
        
        try:
            # Calculate the de-embedding matrix M
            # According to the method: M = T_long * inv(T_short)
            # T1 is t_short, T2 is t_long
            T1_inv = np.linalg.inv(T1)
            M = np.matmul(T2, T1_inv)
            
            # Calculate the eigenvalues of matrix M
            eigenvalues = np.linalg.eigvals(M)
            
            # Select the correct eigenvalue (smaller magnitude, representing attenuation)
            # A passive transmission line must have attenuation, so magnitude <= 1
            # Note: Sometimes numerical noise might make it slightly > 1, but generally <= 1.
            # We select the one with the smaller magnitude.
            # However, we must be careful. The eigenvalues correspond to e^(-gamma*L) and e^(+gamma*L).
            # e^(-gamma*L) is the forward transmission (lossy), so magnitude < 1.
            # e^(+gamma*L) is the formulation for backward, magnitude > 1.
            
            # Let's sort by magnitude
            # We want the one with magnitude < 1 (or the smaller one if both are close to 1)
            # Also, check for phase continuity to avoid swapping?
            # For now, just magnitude based selection as per instructions.
            
            if np.abs(eigenvalues[0]) < np.abs(eigenvalues[1]):
                lambda_forward[i] = eigenvalues[0]
            else:
                lambda_forward[i] = eigenvalues[1]
                
        except np.linalg.LinAlgError:
            # Handle singular matrix inversion gracefully
            lambda_forward[i] = np.nan + 1j * np.nan
            
    return freq, lambda_forward

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generate timestamp string (YYYYMMDD_HHMMSS)
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        if ENABLE_DEEMBEDDING:
            if not SHORT_LINE_FILENAME or not LONG_LINE_FILENAME:
                raise ValueError("Error: De-embedding mode enabled but SHORT or LONG filename is missing.")
                
            print(f"[{timestamp_str}] Dual De-embedding Mode Selected.")
            file_short_path = os.path.join(script_dir, SHORT_LINE_FILENAME)
            file_long_path = os.path.join(script_dir, LONG_LINE_FILENAME)
            
            if not os.path.exists(file_short_path):
                raise FileNotFoundError(f"Error: Short line file not found at: {file_short_path}")
            if not os.path.exists(file_long_path):
                raise FileNotFoundError(f"Error: Long line file not found at: {file_long_path}")

            print(f"Reading Short Line: {file_short_path}")
            print(f"Reading Long Line:  {file_long_path}")
                
            freq, lambda_forward = extract_gms(file_short_path, file_long_path)
            mode_str = "deembedded"
            base_name = os.path.splitext(LONG_LINE_FILENAME)[0]
        
        else:
            if not SINGLE_LINE_FILENAME:
                 raise ValueError("Error: Single line mode enabled but SINGLE filename is missing.")
            
            print(f"[{timestamp_str}] Single File Mode Selected.")
            file_single_path = os.path.join(script_dir, SINGLE_LINE_FILENAME)
            if not os.path.exists(file_single_path):
                raise FileNotFoundError(f"Error: Single file not found at: {file_single_path}")

            print(f"Reading Single Line: {file_single_path}")
            freq, lambda_forward = extract_gms_single(file_single_path)
            mode_str = "single"
            base_name = os.path.splitext(SINGLE_LINE_FILENAME)[0]
        
        # 5. Construct the final GMS S21 and Plot
        # Calculate Magnitude in dB
        # Avoid log10(0) by adding a small epsilon if necessary, though lambda_forward shouldn't be 0
        mag_db = 20 * np.log10(np.abs(lambda_forward) + 1e-15)
        
        # Calculate Phase in degrees (unwrapped)
        # The phase should be unwrapped and then wrapped to 360 degrees
        phase_rad = np.unwrap(np.angle(lambda_forward))
        phase_deg = np.degrees(phase_rad)
        
        # Wrap phase to 360 degrees
        phase_deg = phase_deg % 360
        
        freq_ghz = freq / 1e9
        
        # 6. Output GMS parameters to .s4p file
        # Create a 4-port network with the same format as input
        # S11, S22, S33, S44 = 0
        # S21, S12, S43, S34 = lambda_forward (assuming symmetric and same for P and N)
        # Other cross terms = 0
        
        num_points = len(freq)
        s_gms = np.zeros((num_points, 4, 4), dtype=complex)
        
        # User specified P: 1-2, N: 3-4.
        # S21 (From Port 1 to Port 2) -> index [1, 0]
        s_gms[:, 1, 0] = lambda_forward
        # S12 (From Port 2 to Port 1) -> index [0, 1]
        s_gms[:, 0, 1] = lambda_forward
        
        # S43 (From Port 3 to Port 4) -> index [3, 2]
        s_gms[:, 3, 2] = lambda_forward
        # S34 (From Port 4 to Port 3) -> index [2, 3]
        s_gms[:, 2, 3] = lambda_forward
        
        nw_gms = rf.Network(frequency=rf.Frequency.from_f(freq, unit='hz'), s=s_gms)
        
        # Output filename: [time]_[filename]_diffline.s4p
        output_filename = f"{timestamp_str}_{base_name}_diffline.s4p"
        output_file_path = os.path.join(script_dir, output_filename)
        
        nw_gms.write_touchstone(output_file_path)
        print(f"GMS parameters saved to: {output_file_path}")
        
        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Subplot 1: Magnitude
        ax1.plot(freq_ghz, mag_db, label=f'GMS S21 ({mode_str})', color='b')
        ax1.set_title(f'GMS S21 Magnitude ({mode_str})')
        ax1.set_xlabel('Frequency (GHz)')
        ax1.set_ylabel('Magnitude (dB)')
        ax1.grid(True)
        ax1.legend()
        
        # Subplot 2: Phase
        ax2.plot(freq_ghz, phase_deg, label=f'GMS S21 Phase ({mode_str})', color='r')
        ax2.set_title(f'GMS S21 Phase ({mode_str})')
        ax2.set_xlabel('Frequency (GHz)')
        ax2.set_ylabel('Phase (Degrees)')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        print("Showing plot...")
        plt.show()
        print("Script finished successfully.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
