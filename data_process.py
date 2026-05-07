import os
import torch
import numpy as np
from rdkit import Chem
from Bio.PDB import PDBParser
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import glob

# 引入你的配置和工具
from config import Config
from utils_chem import get_functional_group_labels, get_bond_angles, get_fingerprint, get_atom_features
from utils_chem import get_rich_atom_features

# 标准氨基酸映射表
AA_TO_ID = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
}

# ==========================================
# 核心修改：将处理单个配体的逻辑提取为全局函数
# ==========================================
def process_single_ligand(args):
    """
    单个配体处理函数，用于多进程调用
    args: (mol, index, save_dir)
    """
    mol, i, save_dir = args
    
    if mol is None:
        return False
    
    try:
        # 0. 基础过滤
        if mol.GetNumAtoms() > Config.MAX_ATOMS_LIGAND:
            return False
            
        # 1. 构象提取
        conf = mol.GetConformer()
        pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
        
        # 2. SCAGE 特征计算
        # 注意：这里会消耗 CPU
        x = get_atom_features(mol)
        y_fg = get_functional_group_labels(mol)     # 官能团 Target
        y_angle = get_bond_angles(mol, conf)        # 角度 Target
        y_fp = get_fingerprint(mol)                 # 指纹 Target
        
        # 3. 组装数据
        data = {
            'x': x,
            'pos': pos,
            'mask_type': torch.zeros_like(x), # 0=Ligand
            'y_fg': y_fg,
            'y_angle': y_angle,
            'y_fp': y_fp,
            'id': f"lig_{i}"
        }
        
        # 4. 保存文件
        # 警告：大量小文件写入会造成 IO 瓶颈，建议使用 SSD
        save_path = os.path.join(save_dir, f"lig_{i}.pt")
        torch.save(data, save_path)
        return True

    except Exception as e:
        # 可以在这里 print(e) 调试，但生产环境建议忽略以保持进度条整洁
        return False

def process_pubchem_parallel():
    """多进程处理配体预训练数据"""
    print(f"Processing PubChem SDF from {Config.PUBCHEM_SDF}...")
    save_dir = os.path.join(Config.PROCESSED_DIR, "pretrain")
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 读取所有分子到内存
    # 如果你的内存小于 16GB 且处理几百万数据，这里可能需要改为分块读取
    print("Reading SDF file into memory... (Please wait)")
    suppl = Chem.SDMolSupplier(Config.PUBCHEM_SDF, removeHs=False)
    
    # 构建任务列表
    # 我们只保留非空的分子，并将它们打包成 (mol, index, save_dir) 的元组
    tasks = []
    for i, mol in enumerate(suppl):
        if mol is not None:
            tasks.append((mol, i, save_dir))
            
    total_mols = len(tasks)
    print(f"Loaded {total_mols} valid molecules. Starting multiprocessing...")
    
    # 2. 开启多进程池
    # cpu_count() 会自动获取你的核心数（比如 8）
    num_cores = 8
    print(f"Using {num_cores} CPU cores.")
    
    count = 0
    with Pool(processes=num_cores) as pool:
        # 使用 imap_unordered 处理任务
        # chunksize=10 表示每次给一个核分发10个任务，减少进程间通信开销
        results = list(tqdm(pool.imap_unordered(process_single_ligand, tasks, chunksize=10), total=total_mols))
        
        # 统计成功数量
        count = sum(results)

    print(f"Successfully saved {count} ligands.")

# ==========================================
# PDBBind 部分通常数据量较小 (1-2万)，
# 且涉及复合物处理，多进程可能引发 Pickling 错误，
# 如果需要也可以改，但通常单核跑个几十分钟也能接受。
# 这里保持原样或稍作优化。
# ==========================================
def load_pdbbind_labels(index_path):
    """
    读取 PDBbind 的 index 文件，返回 {pdb_id: affinity_value} 字典
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Label file not found: {index_path}")

    label_dict = {}
    print(f"Loading labels from {index_path}...")
    
    with open(index_path, 'r') as f:
        for line in f:
            # 跳过注释行
            if line.startswith('#'):
                continue
            
            parts = line.split()
            # PDBbind index 格式:
            # [0]PDB_code  [1]Res  [2]Year  [3]-logKd/Ki  ...
            if len(parts) >= 4:
                pdb_code = parts[0]
                try:
                    affinity = float(parts[3]) # 获取 -logKd/Ki 值
                    label_dict[pdb_code] = affinity
                except ValueError:
                    continue
                    
    return label_dict

def process_pdbbind():
    """处理 DTA 微调数据 (适配 AtomEmbedding + ResidueEmbedding)"""
    print(f"Processing PDBbind from {Config.PDBBIND_ROOT}...")
    save_dir = os.path.join(Config.PROCESSED_DIR, "finetune")
    os.makedirs(save_dir, exist_ok=True)
    
    # [NEW 1] 定义索引文件路径 (请确保在 config.py 中定义了 INDEX_FILE)
    # 假设文件名为 INDEX_general_PL_data.2020，位于 PDBBIND_ROOT 下
    index_file_path = os.path.join(Config.PDBBIND_ROOT, "INDEX_general_PL_data.2020") 
    
    # [NEW 2] 预先加载所有标签到内存
    affinity_dict = load_pdbbind_labels(index_file_path)
    
    pdb_parser = PDBParser(QUIET=True)
    codes = [d for d in os.listdir(Config.PDBBIND_ROOT) if os.path.isdir(os.path.join(Config.PDBBIND_ROOT, d))]
    
    for code in tqdm(codes, desc="Processing PDBBind"):
        try:
            # [NEW 3] 检查当前 code 是否有标签，没有标签就跳过
            if code not in affinity_dict:
                # print(f"Warning: No label found for {code}, skipping.")
                continue

            lig_path = os.path.join(Config.PDBBIND_ROOT, code, f"{code}_ligand.sdf")
            prot_path = os.path.join(Config.PDBBIND_ROOT, code, f"{code}_protein.pdb")
            
            if not os.path.exists(lig_path) or not os.path.exists(prot_path):
                continue

            # --- 1. 处理配体 ---
            suppl = Chem.SDMolSupplier(lig_path)
            if len(suppl) == 0: continue
            mol = suppl[0]
            if mol is None: continue
            
            lig_pos = mol.GetConformer().GetPositions()
            lig_x = get_rich_atom_features(mol) 
            
            # --- 2. 处理蛋白 ---
            structure = pdb_parser.get_structure(code, prot_path)
            lig_center = np.mean(lig_pos, axis=0)
            
            prot_aa_ids = [] 
            prot_pos = []
            
            for atom in structure.get_atoms():
                if atom.element == 'H': continue
                coord = atom.get_coord()
                if np.linalg.norm(coord - lig_center) < Config.POCKET_RADIUS:
                    res_name = atom.get_parent().get_resname()
                    aa_id = AA_TO_ID.get(res_name.upper(), 20)
                    prot_aa_ids.append(aa_id)
                    prot_pos.append(coord)
            
            if len(prot_aa_ids) == 0: continue
            
            if len(prot_aa_ids) > Config.MAX_ATOMS_POCKET:
                prot_aa_ids = prot_aa_ids[:Config.MAX_ATOMS_POCKET]
                prot_pos = prot_pos[:Config.MAX_ATOMS_POCKET]

            x_prot_tensor = torch.zeros((len(prot_aa_ids), 7), dtype=torch.long)
            x_prot_tensor[:, 0] = torch.tensor(prot_aa_ids, dtype=torch.long)
            prot_pos_tensor = torch.tensor(np.array(prot_pos), dtype=torch.float32)

            x = torch.cat([lig_x, x_prot_tensor], dim=0) 
            pos = torch.cat([torch.tensor(lig_pos, dtype=torch.float32), prot_pos_tensor], dim=0)
            
            mask_type = torch.cat([
                torch.zeros(len(lig_x), dtype=torch.long),
                torch.ones(len(prot_aa_ids), dtype=torch.long)
            ])
            
            # --- 5. 获取真实标签 [NEW 4] ---
            # 从字典中取出真实数值
            true_affinity = affinity_dict[code]
            y_affinity = torch.tensor([true_affinity], dtype=torch.float32) 
            
            data = {
                'x': x, 
                'pos': pos, 
                'mask_type': mask_type,
                'y_affinity': y_affinity,
                'id': code
            }
            torch.save(data, os.path.join(save_dir, f"{code}.pt"))
            
        except Exception as e:
            continue

if __name__ == "__main__":
    # Windows/Linux 兼容性保护
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    if Config.MODE == 'pretrain':
        process_pubchem_parallel()  # 调用新的并行函数
    else:
        process_pdbbind()