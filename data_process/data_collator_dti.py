import random
from copy import deepcopy
from typing import List, Dict, Union
import numpy as np
import torch
from echo_logger import *

from data_process.masks import *
from data_process.paddings import *
from naive_fg import all_possible_fg_nums
from utils.global_var_util import GlobalVar
from .compound_tools import CompoundKit, Compound3DKit, get_dist_bar
from .function_group_constant import nfg

# 白名单
additional_popping_attributes = {
    'pair_distances', 'triple_angles', 'atom_pos', 'edges', 'attention_mask',
    'label', 'angel_mask', 'atom_length', 'angles_atom_index', 'angles_bond_index',
    'morgan2048_fp', 'bond_angles', 'edge_distances', 'atom_dist_bar', 'edge_dist_bar', 'atom_bond_dist_bar',
    'spatial_pos_bar', 'atom_bond_distances', 'text_name_embedding',
    'function_group_index', 'function_group_bond_index', 
    'spatial_pos', 'pair_distances_bin', 'bond_angles_bin',
    'label_y', 'label_cliff', 'fg_number', 'bond_distances',
    'protein_x', 'protein_pos' # [DTI] 蛋白相关
}

atom_id_names = set(list(CompoundKit.atom_vocab_dict.keys()) + CompoundKit.atom_float_names)
none_int_names = {'van_der_waals_radis', 'partial_charge', 'mass', 'atom_pos', 'protein_pos', 'morgan2048_fp', 'label'}
RICH_FEATURE_KEYS = ['atomic_num', 'chirality', 'degree', 'formal_charge', 'num_h', 'num_rad_e', 'hybridization']

# ==========================================
# 核心函数：配体处理 (返回 data 和 valid_items)
# ==========================================
def collator_pretrain_pkl_bin(items: List[Dict[str, Union[List[int], np.ndarray, str, int]]]):
    # 1. 过滤逻辑
    valid_items = []
    MAX_ALLOWED_ATOMS = 150 
    for item in items:
        # 基础检查
        if 'atomic_num' not in item: continue
        n_atoms = len(item['atomic_num'])
        n_pos = len(item['atom_pos']) if 'atom_pos' in item else 0
        
        # 长度一致性检查
        if n_atoms != n_pos: continue
        # 长度限制
        if n_atoms > MAX_ALLOWED_ATOMS: continue
        valid_items.append(item)
    
    items = valid_items
    #if not items: return {}, []

    max_len = max([len(item['atomic_num']) for item in items])
    max_angels = max([len(item.get('angles_atom_index', [])) for item in items]) if len(items) > 0 else 1

    # 2. 字段清理
    item_pop = []
    for i in items[0].keys():
        if i not in atom_id_names and i not in additional_popping_attributes:
            item_pop.append(i)
    for item in items:
        for i in item_pop: 
            if i in item: item.pop(i)

    # 3. Padding
    for item in items:
        data_len = len(item['atomic_num'])
        
        # FG 处理
        if 'function_group_index' in item:
            fg = item['function_group_index']
            if not isinstance(fg, np.ndarray): fg = np.array(fg)
            if fg.ndim == 0: fg = fg.reshape(-1)
            if fg.dtype.kind in 'fc': fg = np.nan_to_num(fg).astype(int)
            item['function_group_index'] = padding_function_group_index(fg, max_len)

        # 基础 Padding
        item['bond_angles_bin'] = padding_1d_sequence(item.get('bond_angles_bin', np.full(max_angels, -5)), max_angels, -5)
        item['angles_atom_index'] = padding_2d_loong(item.get('angles_atom_index', np.zeros((max_angels,3))), max_angels, 3)
        item['spatial_pos'] = padding_2d_sequence(item.get('spatial_pos', np.full((max_len, max_len), -1)), max_len, -1)
        item['pair_distances'] = padding_2d_sequence(item.get('pair_distances', np.zeros((max_len, max_len))), max_len, 0)
        item['atom_pos'] = padding_atoms_pos(item.get('atom_pos', np.zeros((max_len, 3))), max_len, 0)
        
        for name in atom_id_names:
            if name in item: item[name] = padding_1d_sequence(item[name], max_len, -1) + 1
        
        item["atom_attention_mask"] = attention_mask_using_0(data_len + 1, max_len + 1, 0)

    # 4. 堆叠 Tensor
    data = {}
    keys = list(items[0].keys())
    for name in keys:
        try:
            val_list = [item[name] for item in items]
            if name not in none_int_names:
                data[name] = torch.tensor(np.array(val_list), dtype=torch.int)
            else:
                data[name] = torch.tensor(np.array(val_list), dtype=torch.float)
        except: pass

    # 生成 atom_feature
    feats = []
    for k in RICH_FEATURE_KEYS:
        if k in data: feats.append(data[k])
    if feats:
        data['atom_feature'] = torch.stack(feats, dim=-1).long()

    # [重要修改] 返回两个值：处理好的data，和过滤后的items
    return data, items

# ==========================================
# [新增] DTI 专用 Collator
# ==========================================
def collator_dti(items):
    # 1. 处理配体 (拿到过滤后的 valid_items)
    batch_dict, valid_items = collator_pretrain_pkl_bin(items)
    
    if not batch_dict: return {}
    
    # 2. 处理蛋白 (基于 valid_items，保证一一对应)
    if 'protein_x' not in valid_items[0]:
        return batch_dict # 如果没有蛋白数据，就退化为纯配体

    max_prot_len = max([len(item['protein_x']) for item in valid_items])
    max_prot_len = min(max_prot_len, 256) # 限制最大长度
    
    prot_x_list = []
    prot_pos_list = []
    
    for item in valid_items:
        # Sequence
        p_x = item['protein_x']
        l = min(len(p_x), max_prot_len)
        pad_x = np.zeros(max_prot_len, dtype=int)
        pad_x[:l] = p_x[:l]
        prot_x_list.append(torch.tensor(pad_x, dtype=torch.long))
        
        # Pos
        p_pos = item['protein_pos']
        pad_pos = np.zeros((max_prot_len, 3))
        pad_pos[:l] = p_pos[:l]
        prot_pos_list.append(torch.tensor(pad_pos, dtype=torch.float))
        
    batch_dict['protein_x'] = torch.stack(prot_x_list)
    batch_dict['protein_pos'] = torch.stack(prot_pos_list)
    
    return batch_dict

# 兼容旧接口 (只返回 data)
def collator_finetune_pkl(items):
    data, _ = collator_pretrain_pkl_bin(items)
    return data

def collator_cliff_pkl(items):
    data, _ = collator_pretrain_pkl_bin(items)
    return data