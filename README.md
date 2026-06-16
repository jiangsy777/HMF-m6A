# HMF-m6A
A Hierarchical Multimodal Framework for General RNA m6A Prediction via Context-Aware Large Language Model
Please download the complete datasets, models, and codes from Zenodo (https://doi.org/10.5281/zenodo.20534798) before running the following commands.

# HMF-m6A: Hierarchical Multimodal m6A RNA Methylation Predictor

## 1. Environment Setup

### 1.1 One-Click Setup (Recommended)

A one-click launcher script is provided. It automatically creates a virtual environment, installs dependencies, and launches the web app.

```bash
bash Start_Venv.sh
```

**Features:**
- Creates an isolated `.venv/` directory in the current folder
- Installs PyTorch with CUDA 12.1 support (falls back to CPU if unavailable)
- Uses Tsinghua PyPI mirror for fast downloads
- On subsequent runs, skips installation and launches directly

### 1.2 Manual Setup (venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_venv.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
streamlit run app.py
```

### 1.3 Conda Setup

```bash
# Option 1: Using environment.yml
conda env create -f environment.yml
conda activate m6A

# Option 2: Manual conda setup
conda create -n m6A python=3.9
conda activate m6A
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 2. Data Preparation

Due to GitHub storage limits, the DNABERT3 pretrained model, training data, and trained model weights are hosted on Zenodo. Please complete the following steps before running any code.

### 2.1 Download DNABERT3 (3-mer)

Download the DNABERT3 pretrained model from [Zhihan's Lab](https://github.com/Zhihan1996/DNABERT_3) and place it under the project root directory:

```
HMF-m6A/
└── DNABERT3/          # DNABERT3 pretrained weights (pytorch_model.bin, config.json, etc.)
```

> The same `DNABERT3/` folder can be shared across all ablation model directories via symlink.

### 2.2 Download Data & Trained Models from Zenodo

Download the following from Zenodo and place them in the project root directory:

| Item | Description |
|------|-------------|
| `all_train_samples.tsv` | Training dataset |
| `data/` | Reference genome data |
| `HIV_data/` | HIV m6A dataset for cross-species prediction |
| `test_motif_results/` | Independent test set (18 motif-grouped CSVs) |
| `checkpoints/` | Trained fusion trunk weights (`multimodal_fusion_trunk.pt`) |
| `classifiers/` | Trained motif-specific classifiers (18 `.pt` files + `routing_map.json`) |
| `features/` | Extracted feature matrix (`m6a_features.npz`) |
| `grouped_data/` | Motif-grouped train/val splits |

## 3. Model Training

### 3.1 Train the Full Model

```bash
source .venv/bin/activate  # or conda activate m6A
python main.py train
```

**Optional arguments:**
- `--seed`: Random seed (default: 42)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)

### 3.2 Training Process

The training pipeline includes:
1. **Fusion Trunk Training**: Train the multimodal fusion model
2. **Feature Extraction**: Extract features using the trained trunk
3. **Motif Grouping**: Group samples by 5-mer motifs
4. **Motif Classifier Training**: Train 18 motif-specific classifiers

## 4. Prediction

### 4.1 Predict on Independent Test Set

```bash
python main.py predict
```

Predicts on the `test_motif_results/` directory and generates results.

### 4.2 Predict on Custom CSV

```bash
python main.py predict --input your_file.csv
```

### 4.3 General Prediction (Custom Column)

```bash
python main_general.py predict HIV_data/GSE280563_SAC_seq_D28-30_Mock_vs_HIV-mRNA-sites_cleaned_with_context.csv context_find
```

**Arguments:**
- First argument: Path to CSV file
- Second argument: Column name containing RNA sequences

## 5. Evaluation & Visualization

### 5.1 Dimensionality Reduction

```bash
python dim_reduction_visualize.py
```

**Output:** PCA/t-SNE scatter plots with clustering metrics.

**Figures:**
- **PCA Projection**: Scatter plot colored by prediction label
- **t-SNE Visualization**: Non-linear dimensionality reduction
- **Feature Distribution**: Violin plots of embedding dimensions

### 5.2 ROC Curves & Confusion Matrices

```bash
python eval_test_results.py
```

**Output:** Per-motif-group and overall performance metrics.

**Figures:**
- **ROC-AUC Curves**: Receiver Operating Characteristic curves per motif group
- **Confusion Matrices**: True vs predicted labels heatmaps
- **Performance Summary**: Bar charts of accuracy, precision, recall, F1-score

## 6. Web User Interface (Web UI)

### 6.1 Launch the Web App

```bash
bash Start_Venv.sh
```

Or manually:

```bash
source .venv/bin/activate
streamlit run app.py --server.port 8501 --server.headless true
```

Open `http://localhost:8501` in your browser.

### 6.2 Web App Usage

**Inputs:**
- Upload CSV/TSV file containing RNA sequences
- Select sequence column from dropdown
- Adjust prediction threshold (default: 0.5)
- Configure visualization options

**Adjustable Parameters:**
- **Prediction Threshold**: 0-1 slider for m6A classification cutoff
- **Batch Size**: Number of samples processed per batch
- **Device**: Auto/GPU/CPU selection
- **DPI**: Figure resolution (300/600/900/1200)

### 6.3 Output Figures

- **Fig 1 — Motif Distribution Pareto Analysis**: Bar chart showing motif group frequencies with cumulative percentage curve.

- **Fig 2 — Joint PCA Mapping**: PCA scatter plots with marginal violin densities for DNABERT embeddings, RNA secondary structure, physicochemical features, and classifier hidden states.

- **Fig 3 — Motif-Group Enrichment Clustermap**: Hierarchically clustered heatmap of Z-score normalized motif-group vs label co-occurrence.

- **Fig 4 — MFE & Physicochemical Profiling**:
  - **Fig 4a**: Violin plot comparing minimum free energy distributions between m6A and non-m6A samples.
  - **Fig 4b**: Grouped boxplot of 29 physicochemical feature dimensions stratified by prediction label.
  - **Fig 4c**: Heatmap of Z-score normalized mean physicochemical profiles across motif groups.

## 7. Directory Structure

```
HMF-m6A/
├── DNABERT3/                    # DNABERT3 pretrained model
├── all_train_samples.tsv        # Training dataset
├── data/                        # Reference genome data
├── HIV_data/                    # HIV m6A dataset
├── test_motif_results/          # Independent test set
├── checkpoints/                 # Fusion trunk weights
├── classifiers/                 # Motif-specific classifiers
├── features/                    # Extracted feature matrix
├── grouped_data/                # Motif-grouped splits
├── main.py                      # Main training/prediction script
├── main_general.py              # General prediction script
├── fusion_models.py             # Model architecture
├── motif_grouper.py             # Motif grouping utilities
├── dim_reduction_visualize.py   # Dimensionality reduction
├── eval_test_results.py         # ROC/CM evaluation
├── app.py                       # Streamlit web app
├── Start_Venv.sh                # One-click launcher
├── requirements_venv.txt        # Minimal dependencies
├── requirements.txt             # Full dependencies
└── environment.yml              # Conda environment
```

## 8. References

- DNABERT3: [https://github.com/Zhihan1996/DNABERT_3](https://github.com/Zhihan1996/DNABERT_3)
- Trained models and data: Zenodo repository

# HMF-m6A: 层级多模态 m6A RNA 甲基化预测器

## 1. 环境配置

### 1.1 一键启动（推荐）

提供了一个一键启动脚本，自动创建虚拟环境、安装依赖并启动网页应用。

```bash
bash Start_Venv.sh
```

**特性：**
- 在当前目录下创建隔离的 `.venv/` 虚拟环境
- 安装支持 CUDA 12.1 的 PyTorch（无 GPU 时自动回退到 CPU 版本）
- 使用清华 PyPI 镜像，下载速度快
- 再次运行时跳过安装步骤，直接启动

### 1.2 手动配置（venv）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_venv.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
streamlit run app.py
```

### 1.3 Conda 配置

```bash
# 方式一：使用 environment.yml
conda env create -f environment.yml
conda activate m6A

# 方式二：手动配置
conda create -n m6A python=3.9
conda activate m6A
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 2. 数据准备

由于 GitHub 存储空间限制，DNABERT3 预训练模型、训练数据和模型权重托管在 Zenodo。请在运行代码前完成以下步骤。

### 2.1 下载 DNABERT3（3-mer）

从 [Zhihan's Lab](https://github.com/Zhihan1996/DNABERT_3) 下载 DNABERT3 预训练模型，并放置在项目根目录：

```
HMF-m6A/
└── DNABERT3/          # DNABERT3 预训练权重（pytorch_model.bin, config.json 等）
```

> 同一个 `DNABERT3/` 文件夹可通过符号链接在所有消融模型目录间共享。

### 2.2 从 Zenodo 下载数据和模型

从 Zenodo 下载以下文件并放置在项目根目录：

| 文件/目录 | 说明 |
|------|------|
| `all_train_samples.tsv` | 训练数据集 |
| `data/` | 参考基因组数据 |
| `HIV_data/` | HIV m6A 数据集（跨物种预测） |
| `test_motif_results/` | 独立测试集（18 个 motif 分组 CSV） |
| `checkpoints/` | 训练好的融合主干权重（`multimodal_fusion_trunk.pt`） |
| `classifiers/` | 训练好的 motif 特异性分类器（18 个 `.pt` 文件 + `routing_map.json`） |
| `features/` | 提取的特征矩阵（`m6a_features.npz`） |
| `grouped_data/` | 按 motif 分组的训练/验证划分 |

## 3. 模型训练

### 3.1 训练完整模型

```bash
source .venv/bin/activate  # 或 conda activate m6A
python main.py train
```

**可选参数：**
- `--seed`：随机种子（默认：42）
- `--epochs`：训练轮数（默认：50）
- `--batch_size`：批大小（默认：32）
- `--lr`：学习率（默认：1e-4）

### 3.2 训练流程

训练管道包括：
1. **融合主干训练**：训练多模态融合模型
2. **特征提取**：使用训练好的主干提取特征
3. **Motif 分组**：按 5-mer motif 对样本分组
4. **Motif 分类器训练**：训练 18 个 motif 特异性分类器

## 4. 预测

### 4.1 预测独立测试集

```bash
python main.py predict
```

对 `test_motif_results/` 目录进行预测并生成结果。

### 4.2 预测自定义 CSV 文件

```bash
python main.py predict --input your_file.csv
```

### 4.3 通用预测（指定列名）

```bash
python main_general.py predict HIV_data/GSE280563_SAC_seq_D28-30_Mock_vs_HIV-mRNA-sites_cleaned_with_context.csv context_find
```

**参数说明：**
- 第一个参数：CSV 文件路径
- 第二个参数：包含 RNA 序列的列名

## 5. 评估与可视化

### 5.1 降维分析

```bash
python dim_reduction_visualize.py
```

**输出：** 带聚类指标的 PCA/t-SNE 散点图。

**图表说明：**
- **PCA 投影**：按预测标签着色的散点图
- **t-SNE 可视化**：非线性降维分析
- **特征分布**：嵌入维度的提琴图

### 5.2 ROC 曲线与混淆矩阵

```bash
python eval_test_results.py
```

**输出：** 各 motif 分组及整体性能指标。

**图表说明：**
- **ROC-AUC 曲线**：各 motif 分组的接收者操作特征曲线
- **混淆矩阵**：真实标签与预测标签的热力图
- **性能汇总**：准确率、精确率、召回率、F1 分数的柱状图

## 6. 网页用户界面 (Web UI)

### 6.1 启动网页应用

```bash
bash Start_Venv.sh
```

或手动启动：

```bash
source .venv/bin/activate
streamlit run app.py --server.port 8502
```

在浏览器中打开 `http://localhost:8502`。

### 6.2 网页应用使用说明

**输入：**
- 上传包含 RNA 序列的 CSV/TSV 文件
- 从下拉菜单选择序列列
- 调节预测阈值（默认：0.5）
- 配置可视化选项

**可调参数：**
- **预测阈值**：m6A 分类截止值（0-1 滑块）
- **批大小**：每批处理的样本数
- **设备**：自动/显卡/CPU 选择
- **DPI**：图表分辨率（300/600/900/1200）

### 6.3 输出图表说明

- **图 1 — Motif 分布帕累托分析**：展示各 motif 分组频率的柱状图及累积百分比曲线。

- **图 2 — 联合 PCA 映射**：DNABERT 嵌入、RNA 二级结构、理化性质特征和分类器隐藏状态的 PCA 散点图及边缘提琴图。

- **图 3 — Motif 分组富集聚类热力图**：按 motif 分组与样本标签的 Z-score 标准化交叉表层次聚类热力图。

- **图 4 — MFE 与理化性质分析**：
  - **图 4a**：m6A 与非 m6A 样本间最小自由能分布的提琴图比较。
  - **图 4b**：按预测标签分层的 29 维理化性质指标的分组箱线图。
  - **图 4c**：各 motif 分组的 Z-score 标准化平均理化性质热力图。

## 7. 目录结构

```
HMF-m6A/
├── DNABERT3/                    # DNABERT3 预训练模型
├── all_train_samples.tsv        # 训练数据集
├── data/                        # 参考基因组数据
├── HIV_data/                    # HIV m6A 数据集
├── test_motif_results/          # 独立测试集
├── checkpoints/                 # 融合主干权重
├── classifiers/                 # Motif 特异性分类器
├── features/                    # 提取的特征矩阵
├── grouped_data/                # Motif 分组划分
├── main.py                      # 主训练/预测脚本
├── main_general.py              # 通用预测脚本
├── fusion_models.py             # 模型架构
├── motif_grouper.py             # Motif 分组工具
├── dim_reduction_visualize.py   # 降维可视化
├── eval_test_results.py         # ROC/混淆矩阵评估
├── app.py                       # Streamlit 网页应用
├── Start_Venv.sh                # 一键启动脚本
├── requirements_venv.txt        # 精简依赖清单
├── requirements.txt             # 完整依赖清单
└── environment.yml              # Conda 环境配置
```

## 8. 参考资料

- DNABERT3: [https://github.com/Zhihan1996/DNABERT_3](https://github.com/Zhihan1996/DNABERT_3)
- 训练模型和数据：Zenodo 仓库

