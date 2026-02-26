# GMS Parameter Extraction Tool (GMS 参数提取工具)

This tool implements the **Generalized Modal S-parameters (GMS)** extraction method using Python and `scikit-rf`. It is designed for high-speed signal integrity analysis, allowing users to extract pure transmission line properties from S-parameter measurements.

本工具基于 Python 和 `scikit-rf` 实现了 **通用模态 S 参数 (GMS)** 提取方法。专为高速信号完整性分析设计，支持从 S 参数测量中提取纯传输线特性。

## Features (功能特点)

- **Dual Line De-embedding (双线去嵌)**: Extracts GMS parameters using Short and Long transmission line data ($M = T_{long} \cdot T_{short}^{-1}$).
- **Single Line Analysis (单线分析)**: Analyzes a single transmission line for verification.
- **Auto Differential Handling (自动差分处理)**: Automatically converts 4-port S-parameters to mixed-mode Sdd parameters.
- **Visualization (可视化)**: Plots Magnitude and Phase of the extracted GMS S21 parameters.
- **Standard Output (标准输出)**: Saves results as Touchstone (`.s4p`) files with timestamps.

## Requirements (环境要求)

- Python 3.8+
- Dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Usage (使用方法)

### 1. Dual Line De-embedding (Recommended) / 双线去嵌模式（推荐）
Prepare two Touchstone files (Short line and Long line) and run:

```bash
python gms_workflow_open.py --dual --short path/to/short.s4p --long path/to/long.s4p
```

### 2. Single Line Mode / 单线模式
Analyze a single file:

```bash
python gms_workflow_open.py --single --target path/to/file.s4p
```

### 3. Default Run / 默认运行
If you edit the configuration variables at the top of `gms_workflow_open.py`, you can simply run:

```bash
python gms_workflow_open.py
```

## Output (输出)

The script generates:
1. A **Touchstone file** (`.s4p`) named with a timestamp, e.g., `20260226_103000_filename_diffline_deembedded.s4p`.
2. A **Matplotlib plot** showing the Insertion Loss (Magnitude) and Phase.

## License

MIT License
