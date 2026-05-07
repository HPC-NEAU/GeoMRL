# GeoMRL: A Unified Geometric Framework for Multi-scale Molecular Representation Learning

[![Paper](https://img.shields.io/badge/Paper-Nature_Communications_Target-blue)](https://github.com/your-username/GeoMRL)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![RDKit](https://img.shields.io/badge/RDKit-2023+-green.svg)](https://www.rdkit.org/)

**GeoMRL** (Geometric Multi-scale Representation Learning) is a state-of-the-art geometric deep learning framework designed to bridge intra- and inter-molecular property predictions. By internalizing universal physical constraints through continuous RBF kernels and geometric gated cross-attention, GeoMRL excels at tasks ranging from small-molecule ADMET prediction to complex protein-ligand binding and protein-protein interactions.


## 🌟 Key Features

- **Continuous Geometric Induction Bias**: Unlike traditional models that use discrete distance binning, GeoMRL uses **Radial Basis Function (RBF)** kernels to map precise spatial distances, preserving physical continuity.
- **Geometric Gated Cross-Attention**: A novel interaction module that dynamically pinpoints biophysically compatible "hotspots" between ligands and protein backgrounds.
- **Dynamic Label Scaling (DLS)**: Standardizes target variance on-the-fly to ensure stable fine-tuning across heterogeneous datasets (e.g., pKd, hydration free energy).
- **Multi-scale Unified Pipeline**: A single architecture supporting:
    - **MPP**: MoleculeNet (BBBP, BACE, ClinTox, etc.)
    - **DTI**: Drug-Target Interaction (BioSNAP, PDBbind)
    - **DDI**: Drug-Drug Interaction (ZhongDDI)
    - **PPI**: Protein-Protein Interaction (D-SCRIPT)


### 💾 Data Availability

GeoMRL is evaluated on a comprehensive set of multi-scale biochemical benchmarks. All datasets used in this study are publicly available:

*   **Pre-training Dataset**:
    *   **PubChem**: A large-scale repository of 5 million drug-like compounds used for the M4 pre-training strategy. Available at [PubChem FTP](https://pubchem.ncbi.nlm.nih.gov/).
*   **Intra-molecular Property Prediction (MPP)**:
    *   **MoleculeNet**: Includes 9 benchmarks (BBBP, BACE, ClinTox, Tox21, ToxCast, SIDER, FreeSolv, ESOL, Lipophilicity). Available through the [DeepChem library](https://moleculenet.org/).
*   **Inter-molecular Interaction & Affinity**:
    *   **DTA (Drug-Target Affinity)**: The **PDBBind (v2020)** Refined Set provides 3D structures and affinity labels. Available at [PDBBind-CN](http://www.pdbbind.org.cn/).
    *   **DTI (Drug-Target Interaction)**: The **BioSNAP** dataset consists of 58,964 interaction pairs. Available at [SNAP Biomedical Datasets](http://snap.stanford.edu/biodata/).
    *   **DDI (Drug-Drug Interaction)**: The **ZhongDDI** dataset evaluates drug-drug conflicts across 191,808 pairs.
    *   **PPI (Protein-Protein Interaction)**: The **D-SCRIPT Human** dataset serves as a benchmark for macromolecular assemblies. Available at [D-SCRIPT Official](https://d-script.readthedocs.io/).


## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/HPC-NEAU/GeoMRL.git
cd GeoMRL

# Install dependencies
conda create -n geomrl python=3.9
conda activate geomrl
pip install torch torch-geometric rdkit tqdm scipy pandas scikit-learn
```

### 2. Data Preparation
- **Pre-training**: Download the PubChem SDF (e.g., `Compound_002000001_002500000.sdf`).
- **Fine-tuning**: Prepare MoleculeNet CSV files or PDBbind v2020 datasets.

Use the provided scripts to process raw data:
```bash
# Pre-process ligand data for pre-training
python data_process/data_process.py --mode pretrain

# Build Memory-mapped (Mmap) dataset for high-speed training
python data_process/prepare_mmap.py --input processed_data/pretrain --output processed_data/pretrain/mmap_data
```

---

## 🚂 Training & Fine-tuning

### Distributed Pre-training (DDP)
GeoMRL supports large-scale pre-training using Distributed Data Parallel and Mmap datasets for zero-memory overhead.
```bash
torchrun --nproc_per_node=8 main.py \
    --dataroot /path/to/mmap_data \
    --batch_size 32 \
    --lr 1e-4
```

### Fine-tuning on MoleculeNet (MPP)
Evaluate on benchmarks with scaffold-splitting and 5-seed robust testing:
```bash
python finetune/finetune_mpp_5seed.py \
    --task bbbp \
    --ckpt checkpoints/pretrain_model.pth \
    --batch_size 64 \
    --epochs 100
```

### Drug-Target Interaction (DTI) & Affinity (DTA)
```bash
python finetune/DTI.py \
    --data_root data/pdbbind_v2020.pkl \
    --ckpt checkpoints/pretrain_model.pth \
    --gpu 0
```

---

## 📊 Results

GeoMRL achieves significant improvements across diverse benchmarks:

| Dataset | Metric | GROVER | Uni-Mol | SCAGE | **GeoMRL (Ours)** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BBBP** | AUC-ROC ↑ | 0.721 | 0.718 | 0.734 | **0.925** |
| **ClinTox**| AUC-ROC ↑ | 0.751 | 0.873 | 0.927 | **0.960** |
| **FreeSolv**| RMSE ↓ | 2.198 | 1.725 | 1.688 | **1.337** |
| **DTI (BioSNAP)** | AUC-ROC ↑ | - | - | - | **0.912** |

---

## 🔬 Interpretability
GeoMRL's attention mechanism functions as an *in silico* steric clash detector. In DDI tasks, it automatically assigns maximum weights to overlapping reactive functional groups (e.g., carboxyl groups), identifying toxic structural incompatibilities without human supervision.

---

## 📁 Repository Structure
```text

├── config/                 # Configuration files
│   └── config_finetune.yaml      # Hyperparameters for downstream tasks
├── data_process/           # Data cleaning and tensor conversion
│   ├── compound_tools.py         # RDKit wrappers & molecular feature extraction
│   ├── data_collator.py          # Batching logic for MPP/Pretrain
│   ├── data_collator_dti.py      # Specialized collator for Drug-Target pairs
│   ├── function_group_constant.py# Chemical functional group definitions
│   ├── distant_util.py           # Geometry and spatial calculation utilities
│   └── algos.pyx                 # Cython-accelerated algorithms for speed
├── models/                 # Model architectures
│   ├── GeoMRL.py                 # Core GeoMRL architecture (Backbone + Heads)
│   ├── GeoMRL_ablation.py        # Model variant for ablation studies
│   ├── layers.py                 # Transformer and SE(3) attention layers
│   └── layers/                   # Internal projection & attention modules
├── utils/                  # Shared helper functions
│   ├── public_util.py            # Seed setting, logging, and checkpointing
│   ├── metric_util.py            # Metrics (RMSE, Pearson, AUC, AUPR)
│   └── global_var_util.py        # Global variables and runtime controls
├── main.py                 # Main entry point for pre-training (DDP supported)
├── data_process.py         # Top-level data processing script
├── dataset.py              # PyTorch Dataset definitions
├── _config.py              # Helper script for path and task management
└── README.md               # Project documentation
```

---

## 📝 Citation
If you find our work useful in your research, please cite:
```bibtex
@article{liu2025geomrl,
  title={GeoMRL: A Unified Geometric Framework for Multi-scale Molecular Representation Learning},
  author={Liu, Yutong and Zhou, Changjian},
  journal={arXiv preprint},
  year={2025}
}
```

## 📬 Contact
For questions or collaborations, please contact: **zhouchangjian@neau.edu.cn**
