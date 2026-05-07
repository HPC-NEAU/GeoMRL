import torch

class Config:
    # --- 路径配置 ---
    # 根据你的环境修改
    PUBCHEM_SDF = "./data/pubchem_data/compound/Compound_002000001_002500000.sdf"
    PDBBIND_ROOT = "./data/pdbbind_v2020"
    PROCESSED_DIR = "./processed_data1"
    SAVE_DIR = "./checkpoints"
    
    # --- 运行模式 ---
    MODE = 'pretrain' 
    
    # --- 设备 ---
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 训练超参数  ---
    BATCH_SIZE = 8       # 原论文: 8 (虽然很小，为了复现请保持；如果想快点可以改32)
    LR = 5e-5            # 原论文: 0.00005
    EPOCHS = 100         # 原论文: 100
    SEED = 8             # 原论文: 8
    WEIGHT_DECAY = 1e-4  # 原论文: 0.0001
    NUM_WORKERS = 24     # 原论文: 24 (DataLoader线程数)
    
    # --- 模型结构参数  ---
    ATOM_DIM = 512       # 原论文: --emdedding_dim 512
    NUM_KERNEL = 128     # 保持默认
    NUM_LAYERS = 6       # 原论文: --layer_num 6 (之前是4)
    NUM_HEADS = 16       # 原论文: --num_heads 16 (之前是8)
    

    HIDDEN_SIZE = 256    # 原论文: --hidden_dim 256
    
    FG_NUM_CLASSES = 190 
    MAX_ATOMS_LIGAND = 128
    MAX_ATOMS_POCKET = 256
    POCKET_RADIUS = 6.0
    KNN_K = 12