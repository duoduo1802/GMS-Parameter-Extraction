# PCB 互连表征与 GMS 参数提取工具
## PCB Interconnect Characterization & GMS Extraction Tool

本项目旨在为高速信号完整性 (SI) 工程师提供一套基于 **IEEE P370 标准** 的自动化工具流程。通过 Python 与 `scikit-rf` 库的结合，实现了从 **自动夹具移除 (AFR)** 到 **通用模态 S 参数 (GMS)** 提取的完整链路。

### ✨ 核心功能 (Key Features)

1.  **IEEE P370 AFR 夹具去嵌 (Automatic Fixture Removal)**
    *   通过输入标准的 2xThru 短线（直通校准件），利用时域反射 (TDR) 二分法自动计算并剥离测试夹具。
    *   支持 2 端口（单端）与 4 端口（差分）网络。
    *   **优势**：无需依赖昂贵的 TRL 校准件，仅需一条短线即可实现高精度去嵌。
    *   **合规性**：算法核心直接调用 `scikit-rf` 官方实现的 `IEEEP370_MM_NZC_2xThru` 类，严格遵循 IEEE P370 标准流程。

2.  **GMS 参数提取 (Generalized Modal S-parameters)**
    *   将去嵌后的纯传输线 S 参数转换为 T 矩阵。
    *   通过特征值分解 (Eigenvalue Decomposition) 提取传输线的传播常数 $\gamma$（衰减与相位）。
    *   自动处理差分信号的混合模转换 (Mixed-Mode S-parameters)。

3.  **自动化与批处理**
    *   **一键运行**：代码内置默认配置，无需复杂参数即可运行演示。
    *   **灵活配置**：支持命令行参数 (`argparse`) 修改输入文件、切换单/双线模式、开启/关闭绘图等。

### 🛠️ 环境依赖 (Requirements)

请确保安装 Python 3.8 或更高版本，并安装以下依赖库：

```bash
pip install -r requirements.txt
```

核心依赖：
*   `scikit-rf >= 0.20` (必须，包含 IEEE P370 算法)
*   `numpy`
*   `matplotlib` (用于绘图)

### 🚀 快速开始 (Quick Start)

#### 1. 准备数据文件
*   准备一个 **2xThru 文件**（短线，包含左右夹具，无 DUT），例如 `short.s4p`。
*   准备一个 **Fixture-DUT-Fixture 文件**（长线，包含待测走线及夹具），例如 `long.s4p`。

#### 2. 运行双线去嵌与 GMS 提取（推荐）
这是最标准的流程：先用短线剥离夹具，再提取长线中纯 DUT 的损耗特性。

```bash
python pcb_interconnect_characterization.py --dual --short ./measurements/2xthru.s4p --long ./measurements/dut_total.s4p
```

#### 3. 运行单线模式
如果您已经拥有去嵌后的 S 参数文件，只想提取 GMS 参数：

```bash
python pcb_interconnect_characterization.py --single --target ./deembedded_dut.s4p
```

#### 4. 仅执行 AFR 去嵌（不提取 GMS）
如果您只需要获得去嵌后的 S 参数文件（例如用于 ADS/HSpice 仿真）：

```bash
python pcb_interconnect_characterization.py --no-gms
```

### 📂 输出文件 (Output)

脚本运行后，将在当前目录下生成以下文件：
*   **`AFR_..._short_half_left.s4p`**: 自动提取出的左侧夹具模型。
*   **`AFR_..._short_half_right.s4p`**: 自动提取出的右侧夹具模型。
*   **`AFR_..._deembedded_DUT.s4p`**: 最终去嵌后的纯 DUT S 参数文件。
*   **`YYYYMMDD_..._gms.s4p`**: (如果开启 GMS) 包含提取出的 GMS 模态 S 参数结果。

### 📄 开源协议 (License)

本项目采用 [MIT License](LICENSE) 开源。您可以自由地使用、修改和分发本代码。

---
By duoduo1802
