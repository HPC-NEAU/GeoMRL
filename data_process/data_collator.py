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

# 需要堆叠成 Tensor 的特征
RICH_FEATURE_KEYS = [
    'atomic_num', 'chirality', 'degree', 'formal_charge', 
    'num_h', 'num_rad_e', 'hybridization'
]

# [关键修复] 确保这些键都在白名单里，不会被当做垃圾删掉
additional_popping_attributes = {
    'pair_distances', 'triple_angles', 'atom_pos', 'edges', 'attention_mask',
    'label', 'angel_mask', 'atom_length', 'angles_atom_index', 'angles_bond_index',
    'morgan2048_fp', # 指纹
    'bond_angles', 'edge_distances', 'atom_dist_bar', 'edge_dist_bar', 'atom_bond_dist_bar',
    'spatial_pos_bar', 'atom_bond_distances', 'text_name_embedding',
    'function_group_index', 'function_group_bond_index', 
    'spatial_pos',     # SP 任务
    'pair_distances_bin', 
    'bond_angles_bin', # Angle 任务
    'label_y', 'label_cliff', 'fg_number', 'bond_distances'
}

atom_id_names = set(list(CompoundKit.atom_vocab_dict.keys()) + CompoundKit.atom_float_names)

# 不需要转 Int 的浮点数据
none_int_names = {
    'van_der_waals_radis', 'partial_charge', 'mass', 'atom_pos', 'pair_distances',
    'triple_angles', 'bond_distances', 'bond_angles', 'edge_distances', 'atom_dist_bar',
    'edge_dist_bar', 'atom_bond_dist_bar', 'atom_bond_distances', 'text_name_embedding', 'label_y', 'fg_atom_count',
    'bond_distances', 'label', 'morgan2048_fp'
}

def collator_pretrain_pkl_bin(items: List[Dict[str, Union[List[int], np.ndarray, str, int]]]):
    # 1. 过滤逻辑 (保留)
    valid_items = []
    MAX_ALLOWED_ATOMS = 120 
    for item in items:
        n_atoms = len(item['atomic_num'])
        n_pos = len(item['atom_pos'])
        if n_atoms != n_pos: continue
        if n_atoms > MAX_ALLOWED_ATOMS: continue
        valid_items.append(item)
    
    items = valid_items
    if len(items) == 0: return {}

    max_len = max([len(item['atomic_num']) for item in items])
    # 兼容处理
    max_angels = max([len(item.get('angles_atom_index', [])) for item in items]) if len(items) > 0 else 0
    if max_angels == 0: max_angels = 1
    
    item_pop = []
    for i in items[0].keys():
        if i not in atom_id_names and i not in additional_popping_attributes:
            item_pop.append(i)

    for item in items:
        for i in item_pop:
            if item.get(i) is not None: item.pop(i)
                
        data_len = len(item['atomic_num'])
        edge_len = len(item['edges'])
        item['atom_length'] = data_len
        item['edge_length'] = edge_len

        ## [新增/修改] 强力清洗 function_group_index
        if 'function_group_index' in item:
            fg = item['function_group_index']
            # 转 numpy
            if not isinstance(fg, np.ndarray): fg = np.array(fg)
            
            # [关键修复] 如果是 0维标量 (比如 array(0) 或 array(nan))，强制展平为 1维
            if fg.ndim == 0:
                fg = fg.reshape(-1)
            
            # 如果是浮点数，替换 NaN 并转 Int
            if fg.dtype.kind in 'fc': 
                fg = np.nan_to_num(fg, nan=0).astype(int)
            
            item['function_group_index'] = padding_function_group_index(fg, max_len)

        # Padding 其他字段
        if 'bond_angles_bin' in item:
            item['bond_angles_bin'] = padding_1d_sequence(item['bond_angles_bin'], max_angels, -5)
        else:
            item['bond_angles_bin'] = np.full(max_angels, -5)

        if 'angles_atom_index' in item:
            item['angles_atom_index'] = padding_2d_loong(item['angles_atom_index'], max_angels, 3)
        else:
            item['angles_atom_index'] = np.zeros((max_angels, 3))

        for name in atom_id_names:
            if name in item:
                item[name] = padding_1d_sequence(item[name], max_len, -1) + 1

        if 'spatial_pos' in item:
            item['spatial_pos'] = padding_2d_sequence(item['spatial_pos'], max_len, -1)
            item['atom_dist_bar'] = get_dist_bar(item['spatial_pos'], GlobalVar.dist_bar)
        else:
            item['spatial_pos'] = np.full((max_len, max_len), -1)

        if 'pair_distances' in item:
            item['pair_distances'] = padding_2d_sequence(item['pair_distances'], max_len, 0)
        
        if 'atom_pos' in item:
            item['atom_pos'] = padding_atoms_pos(item['atom_pos'], max_len, 0)

        item["atom_attention_mask"] = attention_mask_using_0(data_len + 1, max_len + 1, 0)

    # Tensor 转换
    data = {}
    keys = list(items[0].keys())
    for name in keys:
        try:
            val_list = [item[name] for item in items]
            data[name] = np.array(val_list)
            if name not in none_int_names:
                data[name] = torch.tensor(data[name], dtype=torch.int)
            else:
                data[name] = torch.tensor(data[name], dtype=torch.float)
        except:
            pass

    feats_to_stack = []
    for k in RICH_FEATURE_KEYS:
        if k in data:
            feats_to_stack.append(data[k])
    if len(feats_to_stack) > 0:
        data['atom_feature'] = torch.stack(feats_to_stack, dim=-1).long()

    return data

# finetune 用同一个逻辑即可
def collator_finetune_pkl(items):
    return collator_pretrain_pkl_bin(items)

def collator_cliff_pkl(items):
    return collator_pretrain_pkl_bin(items)