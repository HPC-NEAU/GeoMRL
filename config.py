import torch

class Config:
    # --- 路径配置 ---
    # 根据你的环境修改
    PUBCHEM_SDF = "./data/pubchem_data/compound/Compound_002000001_002500000.sdf"
    PDBBIND_ROOT = "./data/pdbbind_v2020"
    PROCESSED_DIR = "./processed_data"
    SAVE_DIR = "./checkpoints"
    
    # --- 运行模式 ---
    MODE = 'pretrain' 
    
    # --- 设备 ---
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 训练超参数  ---
    BATCH_SIZE = 8      
    LR = 5e-5          
    EPOCHS = 100      
    SEED = 8            
    WEIGHT_DECAY = 1e-4  
    NUM_WORKERS = 24    
    
    # --- 模型结构参数  ---
    ATOM_DIM = 512      
    NUM_KERNEL = 128    
    NUM_LAYERS = 6       
    NUM_HEADS = 16       
    

    HIDDEN_SIZE = 256    # 原论文: --hidden_dim 256
    
    FG_NUM_CLASSES = 190 
    MAX_ATOMS_LIGAND = 128
    MAX_ATOMS_POCKET = 256
    POCKET_RADIUS = 6.0
    KNN_K = 12
