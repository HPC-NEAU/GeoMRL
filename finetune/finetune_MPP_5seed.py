import argparse
import sys
import os
import shutil
import json
import pickle
from datetime import datetime
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
import yaml
import numpy as np
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, f1_score
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# 1. 路径修复
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 2. 引入新模型 (SE3 Ready)
from models.GeoMRL import GeoMRL

# 3. 引入原版工具
from _config import get_downstream_task_names
from data_process.compound_tools import CompoundKit
from data_process.data_collator import collator_finetune_pkl
from data_process.split import create_splitter
from datasets.dataloader import FinetuneDataset as FinetuneDataset_pkl
from utils.global_var_util import GlobalVar, RoutineControl
from utils.public_util import set_seed, EarlyStopping
from utils.metric_util import compute_reg_metric, compute_cls_metric_tensor
from data_process.function_group_constant import nfg
from utils.userconfig_util import config_current_user, config_dataset_form

warnings.filterwarnings("ignore")

# ==================== 数据集信息配置 ====================
DATASET_INFO = {
    # 分类任务
    'bbbp': {
        'type': 'classification',
        'num_tasks': 1,
        'metric': 'roc_auc',
        'suggested_k': 5,
        'has_scaffold': True,
        'description': 'Blood-Brain Barrier Penetration'
    },
    'bace': {
        'type': 'classification',
        'num_tasks': 1,
        'metric': 'roc_auc',
        'suggested_k': 5,
        'has_scaffold': True,
        'description': 'Beta-Secretase Inhibitors'
    },
    'clintox': {
        'type': 'classification',
        'num_tasks': 2,
        'metric': 'roc_auc',
        'suggested_k': 5,
        'has_scaffold': True,
        'description': 'Clinical Toxicity'
    },
    'tox21': {
        'type': 'classification',
        'num_tasks': 12,
        'metric': 'roc_auc',
        'suggested_k': 5,
        'has_scaffold': True,
        'description': 'Toxicity 21 Challenge'
    },
    'toxcast': {
        'type': 'classification',
        'num_tasks': 617,
        'metric': 'roc_auc',
        'suggested_k': 5,
        'has_scaffold': True,
        'description': 'ToxCast Data'
    },
    'sider': {
        'type': 'classification',
        'num_tasks': 27,
        'metric': 'roc_auc',
        'suggested_k': 5,
        'has_scaffold': True,
        'description': 'Side Effect Resource'
    },
    'muv': {
        'type': 'classification',
        'num_tasks': 17,
        'metric': 'roc_auc',
        'suggested_k': 5,
        'has_scaffold': True,
        'description': 'Maximum Unbiased Validation'
    },
    'hiv': {
        'type': 'classification',
        'num_tasks': 1,
        'metric': 'roc_auc',
        'suggested_k': 5,
        'has_scaffold': True,
        'description': 'HIV Replication Inhibition'
    },
    
    # 回归任务
    'freesolv': {
        'type': 'regression',
        'num_tasks': 1,
        'metric': 'rmse',
        'suggested_k': 10,
        'has_scaffold': True,
        'description': 'Free Solvation Database'
    },
    'esol': {
        'type': 'regression',
        'num_tasks': 1,
        'metric': 'rmse',
        'suggested_k': 5,
        'has_scaffold': True,
        'description': 'Water Solubility'
    },
    'lipophilicity': {
        'type': 'regression',
        'num_tasks': 1,
        'metric': 'rmse',
        'suggested_k': 5,
        'has_scaffold': True,
        'description': 'Lipophilicity'
    },
    'qm7': {
        'type': 'regression',
        'num_tasks': 1,
        'metric': 'mae',
        'suggested_k': 5,
        'has_scaffold': False,
        'description': 'Quantum Chemistry 7'
    },
    'qm8': {
        'type': 'regression',
        'num_tasks': 12,
        'metric': 'mae',
        'suggested_k': 5,
        'has_scaffold': False,
        'description': 'Quantum Chemistry 8'
    },
    'qm9': {
        'type': 'regression',
        'num_tasks': 12,
        'metric': 'mae',
        'suggested_k': 5,
        'has_scaffold': False,
        'description': 'Quantum Chemistry 9'
    }
}

# ==================== 骨架K折划分 ====================
class ScaffoldKFold:
    """基于分子骨架的K折划分"""
    
    @staticmethod
    def split(dataset, indices=None, k=5, seed=42, task_name='bbbp'):
        """
        基于骨架的K折划分
        dataset: 完整数据集
        indices: 要划分的索引列表（如果是None，则使用整个数据集）
        返回: 包含训练和验证索引的列表
        """
        print(f"Generating scaffold-based {k}-fold splits for {task_name}...")
        
        if indices is None:
            indices = list(range(len(dataset)))
        
        # 收集所有分子的骨架
        scaffold_to_indices = defaultdict(list)
        
        for idx in indices:
            try:
                # 尝试从数据中获取SMILES
                data = dataset[idx]
                smiles = data.get('smiles', '')
                if not smiles:
                    # 如果数据集中没有smiles，尝试其他方式获取
                    if hasattr(data, 'smiles'):
                        smiles = data.smiles
                    elif 'graph' in data and hasattr(data['graph'], 'smiles'):
                        smiles = data['graph'].smiles
                
                if smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                        scaffold_smiles = Chem.MolToSmiles(scaffold)
                        scaffold_to_indices[scaffold_smiles].append(idx)
                    else:
                        # 如果无法解析SMILES，分配到单独组
                        scaffold_to_indices[f'unknown_{idx}'].append(idx)
                else:
                    # 如果没有SMILES，分配到单独组
                    scaffold_to_indices[f'no_smiles_{idx}'].append(idx)
            except Exception as e:
                print(f"Error processing molecule {idx}: {e}")
                scaffold_to_indices[f'error_{idx}'].append(idx)
        
        # 按骨架大小排序（大的骨架优先分配）
        scaffold_sets = []
        for scaffold, indices_list in scaffold_to_indices.items():
            scaffold_sets.append({
                'scaffold': scaffold,
                'indices': indices_list,
                'size': len(indices_list)
            })
        
        # 按大小降序排序
        scaffold_sets.sort(key=lambda x: x['size'], reverse=True)
        
        print(f"Total scaffolds: {len(scaffold_sets)}")
        if scaffold_sets:
            print(f"Largest scaffold size: {scaffold_sets[0]['size']}")
        
        # 初始化各折
        fold_indices = [[] for _ in range(k)]
        fold_sizes = [0] * k
        
        # 分配骨架到各折，尽量保持平衡
        np.random.seed(seed)
        for scaffold_set in scaffold_sets:
            # 随机选择一个当前最小的折
            min_folds = np.where(fold_sizes == np.min(fold_sizes))[0]
            selected_fold = np.random.choice(min_folds)
            
            fold_indices[selected_fold].extend(scaffold_set['indices'])
            fold_sizes[selected_fold] += scaffold_set['size']
        
        # 生成K折划分
        splits = []
        for fold_idx in range(k):
            val_indices = fold_indices[fold_idx]
            train_indices = []
            
            for other_idx in range(k):
                if other_idx != fold_idx:
                    train_indices.extend(fold_indices[other_idx])
            
            splits.append((train_indices, val_indices))
        
        # 打印各折大小
        print(f"\nScaffold-based fold sizes:")
        for i, (train_idx, val_idx) in enumerate(splits):
            print(f"Fold {i+1}: Train={len(train_idx)}, Val={len(val_idx)}")
        
        return splits

# ==================== 渐进解冻混合类 ====================
class ProgressiveUnfreezeMixin:
    """渐进解冻混合类"""
    
    def setup_progressive_unfreeze(self, freeze_epochs=5, unfreeze_layers=2):
        """设置渐进解冻参数"""
        self.freeze_epochs = freeze_epochs
        self.unfreeze_layers = unfreeze_layers
        self.current_phase = 'freeze'  # freeze, partial, full
        
        # 识别模型的层
        self._identify_layers()
    
    def _identify_layers(self):
        """识别模型中的可训练层"""
        self.layer_names = []
        self.classifier_layers = []
        
        for name, param in self.net.named_parameters():
            if 'classifier' in name or 'fc' in name or 'head' in name:
                self.classifier_layers.append(name)
            elif 'layer' in name or 'encoder' in name:
                self.layer_names.append(name)
        
        # 按层深度排序（从浅到深）
        self.layer_names.sort(key=lambda x: self._get_layer_depth(x))
        
        print(f"Found {len(self.layer_names)} backbone layers")
        print(f"Found {len(self.classifier_layers)} classifier layers")
    
    def _get_layer_depth(self, layer_name):
        """获取层深度（用于排序）"""
        # 根据层名提取深度信息
        parts = layer_name.split('.')
        for part in parts:
            if part.isdigit():
                return int(part)
        return 0
    
    def apply_freeze_phase(self):
        """应用冻结阶段：只训练分类头"""
        print(f"\nPhase 1: Freeze backbone, train classifier only (epochs 1-{self.freeze_epochs})")
        
        # 冻结所有骨干层
        for name, param in self.net.named_parameters():
            if name in self.layer_names:
                param.requires_grad = False
        
        # 只解冻分类层
        for name, param in self.net.named_parameters():
            if name in self.classifier_layers:
                param.requires_grad = True
        
        # 更新优化器（只优化可训练参数）
        trainable_params = filter(lambda p: p.requires_grad, self.net.parameters())
        self.optim = torch.optim.AdamW(
            trainable_params, 
            lr=self.config['optim']['init_lr'] * 5,  # 分类头用较大学习率
            weight_decay=self.config['optim']['weight_decay']
        )
        
        self.current_phase = 'freeze'
    
    def apply_partial_unfreeze(self):
        """应用部分解冻：解冻最后几层"""
        print(f"\nPhase 2: Unfreeze last {self.unfreeze_layers} layers")
        
        # 解冻最后几层
        layers_to_unfreeze = self.layer_names[-self.unfreeze_layers:] if self.unfreeze_layers > 0 else []
        
        for name, param in self.net.named_parameters():
            if name in layers_to_unfreeze or name in self.classifier_layers:
                param.requires_grad = True
            elif name in self.layer_names:
                param.requires_grad = False
        
        # 更新优化器
        trainable_params = filter(lambda p: p.requires_grad, self.net.parameters())
        self.optim = torch.optim.AdamW(
            trainable_params,
            lr=self.config['optim']['init_lr'] * 2,  # 中等学习率
            weight_decay=self.config['optim']['weight_decay']
        )
        
        self.current_phase = 'partial'
    
    def apply_full_unfreeze(self):
        """应用完全解冻：解冻所有层"""
        print(f"\nPhase 3: Full fine-tuning")
        
        # 解冻所有层
        for param in self.net.parameters():
            param.requires_grad = True
        
        # 更新优化器（全部参数，较小学习率）
        self.optim = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.config['optim']['init_lr'],  # 基础学习率
            weight_decay=self.config['optim']['weight_decay']
        )
        
        # 添加学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim,
            T_max=self.config['epochs'] - self.freeze_epochs,
            eta_min=self.config['optim']['init_lr'] * 0.01
        )
        
        self.current_phase = 'full'



# ==================== 统一K折训练器 ====================
class UnifiedKFoldTrainer(ProgressiveUnfreezeMixin):
    """统一的K折训练器，支持所有数据集"""
    
    def __init__(self, config, fold_idx=None, total_folds=5, 
                 use_scaffold_split=True, test_ratio=0.1, 
                 train_indices=None, val_indices=None, test_indices=None):
        """
        参数说明：
        - config: 配置字典
        - fold_idx: 当前折索引（0-based）
        - total_folds: 总折数
        - use_scaffold_split: 是否使用骨架划分
        - test_ratio: 独立测试集比例
        - train_indices: 训练集索引（如果提供，直接使用）
        - val_indices: 验证集索引（如果提供，直接使用）
        - test_indices: 测试集索引（如果提供，直接使用）
        """
        self.config = config
        self.fold_idx = fold_idx
        self.total_folds = total_folds
        self.use_scaffold_split = use_scaffold_split
        self.test_ratio = test_ratio
        
        # 获取数据集信息
        self.task_name = config['task_name']
        self.dataset_info = DATASET_INFO.get(self.task_name.lower(), {
            'type': 'classification',
            'num_tasks': 1,
            'metric': 'roc_auc',
            'suggested_k': 5,
            'has_scaffold': True
        })
        
        # 初始化路径
        self._setup_paths()
        
        # 加载数据集
        self.dataset = self._load_dataset()
        
        # 划分数据
        if train_indices is not None and val_indices is not None and test_indices is not None:
            # 直接使用提供的索引
            self.train_loader, self.val_loader, self.test_loader = self.create_data_loaders_from_indices(
                train_indices, val_indices, test_indices
            )
        elif fold_idx is not None:
            # K折模式：需要划分独立测试集
            self.train_loader, self.val_loader, self.test_loader = self.get_kfold_data_loaders_with_independent_test()
        else:
            # 单次划分模式
            self.train_loader, self.val_loader, self.test_loader = self.get_single_split_data_loaders()
        
        # 初始化模型
        self.net = self._get_net()
        
        # 初始化损失函数
        self.criterion = self._get_loss_fn()
        
        # 初始化优化器
        self.optim = self._get_optim()
        
        # 初始化学习率调度器
        self.scheduler = None
        
        # 训练状态
        self.start_epoch = 1
        self.best_metric = -np.inf if self.dataset_info['type'] == 'classification' else np.inf
        self.best_val_metric = -np.inf if self.dataset_info['type'] == 'classification' else np.inf
        self.best_test_metric = 0
        self.best_model_state = None
        
        # 设置渐进解冻
        self.setup_progressive_unfreeze(
            freeze_epochs=config.get('freeze_epochs', 5),
            unfreeze_layers=config.get('unfreeze_layers', 2)
        )
        
        print(f"\nInitialized trainer for {self.task_name} "
              f"(Fold {fold_idx+1 if fold_idx is not None else 'Single'})")
        print(f"Task type: {self.dataset_info['type']}")
        print(f"Number of tasks: {self.dataset_info['num_tasks']}")
        print(f"Evaluation metric: {self.dataset_info['metric']}")
    
    def _setup_paths(self):
        """设置保存路径"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if self.fold_idx is not None:
            run_name = f"{self.task_name}_fold{self.fold_idx+1}_{timestamp}_seed{self.config['seed']}"
            self.save_dir = f"../train_result/finetune_result/kfold/{run_name}"
        else:
            run_name = f"{self.task_name}_{timestamp}_seed{self.config['seed']}"
            self.save_dir = f"../train_result/finetune_result/{run_name}"
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(self.save_dir)
        self.txtfile = os.path.join(self.save_dir, 'training_log.txt')
        self.model_save_path = os.path.join(self.save_dir, 'best_model.pth')
        
        print(f"Results will be saved to: {self.save_dir}")
    
    def _load_dataset(self):
        """加载数据集"""
        try:
            dataset = FinetuneDataset_pkl(
                root=self.config['userconfig']['mpp']['dataset_dir'], 
                task_name=self.task_name
            )
            print(f"Loaded dataset: {len(dataset)} samples")
            return dataset
        except Exception as e:
            print(f"Error loading dataset {self.task_name}: {e}")
            raise
    
    def create_data_loaders_from_indices(self, train_idx, val_idx, test_idx):
        """从给定的索引创建数据加载器"""
        print(f"\nCreating data loaders:")
        print(f"  Train: {len(train_idx)} samples")
        print(f"  Validation: {len(val_idx)} samples")
        print(f"  Test: {len(test_idx)} samples")
        
        loader_args = dict(
            batch_size=self.config['batch_size'], 
            num_workers=4, 
            collate_fn=collator_finetune_pkl
        )
        
        train_dataset = Subset(self.dataset, train_idx)
        val_dataset = Subset(self.dataset, val_idx)
        test_dataset = Subset(self.dataset, test_idx)
        
        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_args)
        val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_args)
        test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_args)
        
        return train_loader, val_loader, test_loader
    
    def get_kfold_data_loaders_with_independent_test(self):
        """K折交叉验证 + 独立测试集"""
        print(f"\n{'='*60}")
        print(f"K-FOLD WITH INDEPENDENT TEST SET")
        print(f"Fold {self.fold_idx+1}/{self.total_folds}")
        print(f"{'='*60}")
        
        # 1. 获取标签用于分层抽样
        labels = self._get_labels()
        
        # 2. 划分独立测试集
        all_indices = np.arange(len(self.dataset))
        
        if self.dataset_info['type'] == 'classification' and labels is not None:
            # 分类任务：分层抽样
            train_val_idx, test_idx = train_test_split(
                all_indices,
                test_size=self.test_ratio,
                stratify=labels,
                random_state=self.config['seed']
            )
        else:
            # 回归任务：随机抽样
            train_val_idx, test_idx = train_test_split(
                all_indices,
                test_size=self.test_ratio,
                random_state=self.config['seed']
            )
        
        print(f"Independent test set: {len(test_idx)} samples ({self.test_ratio*100:.0f}% of total)")
        
        # 3. 在train_val_idx上进行K折划分
        if self.use_scaffold_split and self.dataset_info['has_scaffold']:
            # 骨架划分
            train_val_dataset = Subset(self.dataset, train_val_idx)
            splits = ScaffoldKFold.split(
                train_val_dataset, 
                indices=None,
                k=self.total_folds, 
                seed=self.config['seed'],
                task_name=self.task_name
            )
            
            if self.fold_idx >= len(splits):
                raise ValueError(f"Fold index {self.fold_idx} exceeds number of splits {len(splits)}")
            
            train_sub_idx, val_sub_idx = splits[self.fold_idx]
            
            # 映射回原始索引
            train_idx = [train_val_idx[i] for i in train_sub_idx]
            val_idx = [train_val_idx[i] for i in val_sub_idx]
            
        elif self.dataset_info['type'] == 'classification' and labels is not None:
            # 标准分层K折
            train_val_labels = labels[train_val_idx]
            skf = StratifiedKFold(
                n_splits=self.total_folds, 
                shuffle=True, 
                random_state=self.config['seed']
            )
            splits = list(skf.split(train_val_idx, train_val_labels))
            
            if self.fold_idx >= len(splits):
                raise ValueError(f"Fold index {self.fold_idx} exceeds number of splits {len(splits)}")
            
            train_sub_idx, val_sub_idx = splits[self.fold_idx]
            train_idx = train_val_idx[train_sub_idx]
            val_idx = train_val_idx[val_sub_idx]
            
        else:
            # 回归任务：随机K折
            kf = KFold(n_splits=self.total_folds, shuffle=True, random_state=self.config['seed'])
            splits = list(kf.split(train_val_idx))
            
            if self.fold_idx >= len(splits):
                raise ValueError(f"Fold index {self.fold_idx} exceeds number of splits {len(splits)}")
            
            train_sub_idx, val_sub_idx = splits[self.fold_idx]
            train_idx = train_val_idx[train_sub_idx]
            val_idx = train_val_idx[val_sub_idx]
        
        print(f"Fold {self.fold_idx+1}: Train={len(train_idx)}, Val={len(val_idx)}")
        
        # 4. 创建数据加载器
        return self.create_data_loaders_from_indices(train_idx, val_idx, test_idx)
    
    def get_single_split_data_loaders(self):
        """获取单次划分的数据加载器"""
        splitter = create_splitter(self.config['split_type'], self.config['seed'])
        train_dataset, val_dataset, test_dataset = splitter.split(self.dataset, self.task_name)
        
        print(f"Single split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        loader_args = dict(
            batch_size=self.config['batch_size'], 
            num_workers=4, 
            collate_fn=collator_finetune_pkl
        )
        
        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_args)
        val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_args)
        test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_args)
        
        return train_loader, val_loader, test_loader
    
    def _get_labels(self):
        """获取所有样本的标签（用于分层抽样）"""
        labels = []
        
        for i in range(len(self.dataset)):
            data = self.dataset[i]
            label = data['label']
            
            if isinstance(label, torch.Tensor):
                if label.dim() == 0:
                    # 单任务
                    label_val = label.item()
                else:
                    # 多任务：取第一个任务用于分层
                    label_val = label[0].item() if len(label) > 0 else 0
            elif isinstance(label, list) or isinstance(label, np.ndarray):
                # 多任务：取第一个任务
                label_val = label[0] if len(label) > 0 else 0
            else:
                # 标量
                label_val = label
            
            labels.append(label_val)
        
        return np.array(labels) if labels else None
    
    def _get_net(self):
        """获取并初始化模型"""
        model = GeoMRL(
            mode='finetune',
            atom_names=CompoundKit.atom_vocab_dict.keys(),
            atom_embed_dim=self.config['model']['atom_embed_dim'],
            num_kernel=self.config['model'].get('num_kernel', 128),
            layer_num=self.config['model']['layer_num'],
            num_heads=self.config['model']['num_heads'],
            atom_FG_class=nfg() + 1,
            hidden_size=self.config['model']['hidden_size'],
            num_tasks=self.dataset_info['num_tasks']
        ).cuda()

        # 加载预训练权重
        ckpt_path = self.config.get('checkpoint')
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"Loading pretrained backbone from: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location='cuda')
            state_dict = checkpoint if 'model' not in checkpoint else checkpoint['model']
            
            # 智能加载：只加载匹配的层
            model_dict = model.state_dict()
            pretrained_dict = {}
            
            for k, v in state_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    pretrained_dict[k] = v
            
            print(f"Loaded {len(pretrained_dict)}/{len(state_dict)} pretrained layers")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
        else:
            print("Warning: No pretrained checkpoint found, training from scratch!")

        return model
    
    def _get_loss_fn(self):
        """获取损失函数"""
        if self.dataset_info['type'] == 'classification':
            # 对于多任务分类，使用加权BCE损失
            if self.dataset_info['num_tasks'] > 1:
                def multi_task_bce(pred, target):
                    loss = 0
                    num_tasks = pred.shape[1]
                    
                    for i in range(num_tasks):
                        task_pred = pred[:, i:i+1]
                        task_target = target[:, i:i+1]
                        
                        # 处理无效标签
                        is_valid = (task_target != -1)
                        
                        if is_valid.sum() > 0:
                            task_loss = F.binary_cross_entropy_with_logits(
                                task_pred[is_valid],
                                task_target[is_valid],
                                reduction='mean'
                            )
                            loss = loss + task_loss
                    
                    return loss / num_tasks  # 平均损失
                
                return multi_task_bce
            else:
                # 单任务分类，计算类别权重
                labels = self._get_labels()
                if labels is not None:
                    pos_count = np.sum(labels == 1)
                    neg_count = np.sum(labels == 0)
                    
                    if pos_count > 0 and neg_count > 0:
                        pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).cuda()
                        print(f"Class weights - Positive: {pos_weight.item():.2f}x")
                        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                
                return nn.BCEWithLogitsLoss()
        else:
            # 回归任务
            return nn.MSELoss()
    
    def _get_optim(self):
        """获取优化器"""
        return torch.optim.AdamW(
            self.net.parameters(),
            lr=self.config['optim']['init_lr'],
            weight_decay=self.config['optim']['weight_decay']
        )
    
    def _step(self, batch):
        """单步训练"""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor): 
                batch[k] = v.cuda()

        pred_dict = self.net(batch)
        pred = pred_dict['graph_feature']
        label = batch['label'].float()
        
        loss = self.criterion(pred, label)
        
        return loss, pred
    
    def _evaluate(self, loader):
        """评估模型"""
        self.net.eval()
        preds, targets = [], []
        
        with torch.no_grad():
            for batch in loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor): 
                        batch[k] = v.cuda()
                
                _, pred = self._step(batch)
                preds.append(pred)
                targets.append(batch['label'])
        
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        
        if self.dataset_info['type'] == 'classification':
            # 分类任务：计算AUC
            if self.dataset_info['num_tasks'] > 1:
                # 多任务：计算平均AUC
                task_aucs = []
                num_tasks = preds.shape[1]
                
                for task_idx in range(num_tasks):
                    task_pred = preds[:, task_idx]
                    task_target = targets[:, task_idx]
                    
                    # 只考虑有效标签
                    valid_mask = task_target != -1
                    if valid_mask.sum() > 0:
                        valid_pred = task_pred[valid_mask]
                        valid_target = task_target[valid_mask]
                        
                        # 计算AUC
                        pred_np = torch.sigmoid(valid_pred).cpu().numpy()
                        target_np = valid_target.cpu().numpy()
                        
                        if len(np.unique(target_np)) == 2:
                            try:
                                auc = roc_auc_score(target_np, pred_np)
                                task_aucs.append(auc)
                            except:
                                pass
                
                return np.mean(task_aucs) if task_aucs else 0.5
            else:
                # 单任务：使用原有函数
                return compute_cls_metric_tensor(targets, preds)
        else:
            # 回归任务：计算RMSE
            mae, rmse = compute_reg_metric(targets, preds)
            return rmse  # 返回RMSE
    
    def train(self):
        """训练主循环"""
        print(f"\n{'='*60}")
        print(f"Starting training for {self.task_name} ")
        print(f"(Fold {self.fold_idx+1 if self.fold_idx is not None else 'Single'})")
        print(f"{'='*60}")
        
        # 早停器
        stopper = EarlyStopping(
            mode='lower' if self.dataset_info['type'] == 'regression' else 'higher', 
            patience=self.config['patience']
        )
        
        # 训练循环
        for epoch in range(1, self.config['epochs'] + 1):
            # 阶段切换
            if epoch == 1:
                self.apply_freeze_phase()
            elif epoch == self.config.get('freeze_epochs', 5) + 1:
                self.apply_partial_unfreeze()
            elif epoch == self.config.get('freeze_epochs', 5) * 2 + 1:
                self.apply_full_unfreeze()
            
            # 训练一个epoch
            train_loss = self._train_epoch(epoch)
            
            # 验证
            val_metric = self._evaluate(self.val_loader)
            
            # 测试集评估（只在特定条件下）
            should_evaluate_test = (
                epoch == 1 or  # 第一个epoch
                epoch % 10 == 0 or  # 每10个epoch
                epoch == self.config['epochs'] or  # 最后一个epoch
                (epoch > 1 and (
                    (self.dataset_info['type'] == 'classification' and val_metric > self.best_metric) or
                    (self.dataset_info['type'] == 'regression' and val_metric < self.best_metric)
                ))
            )
            
            if should_evaluate_test:
                test_metric = self._evaluate(self.test_loader)
            else:
                test_metric = 0 if self.dataset_info['type'] == 'classification' else float('inf')
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 记录最佳模型
            is_best = False
            if self.dataset_info['type'] == 'classification':
                if val_metric > self.best_metric:
                    self.best_metric = val_metric
                    self.best_val_metric = val_metric
                    self.best_test_metric = test_metric if should_evaluate_test else self.best_test_metric
                    self.best_model_state = deepcopy(self.net.state_dict())
                    is_best = True
            else:
                if val_metric < self.best_metric:
                    self.best_metric = val_metric
                    self.best_val_metric = val_metric
                    self.best_test_metric = test_metric if should_evaluate_test else self.best_test_metric
                    self.best_model_state = deepcopy(self.net.state_dict())
                    is_best = True
            
            # 记录日志
            self._log_epoch(epoch, train_loss, val_metric, test_metric, is_best, should_evaluate_test)
            
            # 早停检查
            if stopper.step(val_metric, self.net):
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        # 保存最佳模型
        if self.best_model_state is not None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.best_model_state,
                'val_metric': self.best_val_metric,
                'test_metric': self.best_test_metric,
                'config': self.config,
                'fold': self.fold_idx,
                'task_name': self.task_name
            }, self.model_save_path)
        
        print(f"\nTraining completed!")
        if self.dataset_info['type'] == 'classification':
            print(f"Best Validation AUC: {self.best_val_metric:.4f}")
            print(f"Corresponding Test AUC: {self.best_test_metric:.4f}")
        else:
            print(f"Best Validation RMSE: {self.best_val_metric:.4f}")
            print(f"Corresponding Test RMSE: {self.best_test_metric:.4f}")
        
        return self.best_val_metric, self.best_test_metric
    
    def _train_epoch(self, epoch):
        """训练一个epoch"""
        self.net.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            self.optim.zero_grad()
            loss, _ = self._step(batch)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            
            self.optim.step()
            total_loss += loss.item()
            
            # 更新进度条
            current_lr = self.optim.param_groups[0]['lr']
            pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f} | LR: {current_lr:.6f}")
        
        return total_loss / num_batches
    
    def _log_epoch(self, epoch, train_loss, val_metric, test_metric, is_best, evaluated_test=True):
        """记录epoch结果"""
        metric_name = "AUC" if self.dataset_info['type'] == 'classification' else "RMSE"
        
        msg = f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
        msg += f"Val {metric_name}: {val_metric:.4f} | "
        
        if evaluated_test:
            msg += f"Test {metric_name}: {test_metric:.4f} | "
        
        msg += f"Best Val: {self.best_metric:.4f}"
        
        if is_best:
            msg += " [BEST]"
        
        print(msg)
        
        # 写入文件
        with open(self.txtfile, 'a') as f:
            f.write(f"{msg}\n")
        
        # TensorBoard记录
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar(f'Metric/val', val_metric, epoch)
        
        if evaluated_test:
            self.writer.add_scalar(f'Metric/test', test_metric, epoch)
        
        # 记录学习率
        if self.optim.param_groups:
            self.writer.add_scalar('LR', self.optim.param_groups[0]['lr'], epoch)

# ==================== K折交叉验证运行器 ====================
def run_kfold_cross_validation(config, task_name, k=5, test_ratio=0.1):
    """运行K折交叉验证"""
    
    # 获取数据集信息
    dataset_info = DATASET_INFO.get(task_name.lower(), {
        'type': 'classification',
        'num_tasks': 1,
        'metric': 'roc_auc',
        'suggested_k': 5,
        'has_scaffold': True
    })
    
    print(f"\n{'='*80}")
    print(f"K-FOLD CROSS VALIDATION FOR {task_name.upper()}")
    print(f"Description: {dataset_info['description']}")
    print(f"Task type: {dataset_info['type'].upper()}")
    print(f"Number of tasks: {dataset_info['num_tasks']}")
    print(f"Evaluation metric: {dataset_info['metric'].upper()}")
    print(f"K: {k}, Test ratio: {test_ratio}")
    print(f"{'='*80}")
    
    # 创建汇总目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_dir = f"../train_result/finetune_result/kfold_summary_{task_name}_{timestamp}_seed{config['seed']}"
    os.makedirs(summary_dir, exist_ok=True)
    
    # 保存配置
    config_copy = deepcopy(config)
    config_copy['task_name'] = task_name
    config_copy['kfold'] = k
    config_copy['test_ratio'] = test_ratio
    config_copy['dataset_info'] = dataset_info
    
    with open(os.path.join(summary_dir, 'config.json'), 'w') as f:
        json.dump(config_copy, f, indent=2, default=str)
    
    # 加载数据集
    try:
        dataset = FinetuneDataset_pkl(
            root=config['userconfig']['mpp']['dataset_dir'], 
            task_name=task_name
        )
        print(f"Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # 获取标签用于分层抽样
    labels = None
    if dataset_info['type'] == 'classification':
        labels = []
        for i in range(len(dataset)):
            data = dataset[i]
            label = data['label']
            
            if isinstance(label, torch.Tensor):
                if label.dim() == 0:
                    label_val = label.item()
                else:
                    label_val = label[0].item() if len(label) > 0 else 0
            elif isinstance(label, (list, np.ndarray)):
                label_val = label[0] if len(label) > 0 else 0
            else:
                label_val = label
            
            labels.append(label_val)
        
        labels = np.array(labels)
    
    # 划分独立测试集
    all_indices = np.arange(len(dataset))
    
    if dataset_info['type'] == 'classification' and labels is not None:
        train_val_idx, test_idx = train_test_split(
            all_indices,
            test_size=test_ratio,
            stratify=labels,
            random_state=config['seed']
        )
    else:
        train_val_idx, test_idx = train_test_split(
            all_indices,
            test_size=test_ratio,
            random_state=config['seed']
        )
    
    print(f"\nData Split:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Independent test set: {len(test_idx)} samples ({test_ratio*100:.0f}%)")
    print(f"  Train+Validation: {len(train_val_idx)} samples ({(1-test_ratio)*100:.0f}%)")
    
    # 保存测试集信息
    test_info = {
        'test_indices': test_idx.tolist(),
        'test_size': len(test_idx),
        'test_ratio': test_ratio
    }
    with open(os.path.join(summary_dir, 'test_set_info.json'), 'w') as f:
        json.dump(test_info, f, indent=2)
    
    # 在train_val_idx上进行K折划分
    train_val_dataset = Subset(dataset, train_val_idx)
    use_scaffold = config.get('use_scaffold_split', True) and dataset_info['has_scaffold']
    
    if use_scaffold:
        print(f"\nGenerating scaffold-based {k}-fold splits...")
        splits = ScaffoldKFold.split(
            train_val_dataset,
            indices=None,
            k=k,
            seed=config['seed'],
            task_name=task_name
        )
    elif dataset_info['type'] == 'classification' and labels is not None:
        print(f"\nGenerating stratified {k}-fold splits...")
        train_val_labels = labels[train_val_idx]
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=config['seed'])
        splits = []
        for train_sub_idx, val_sub_idx in skf.split(train_val_idx, train_val_labels):
            train_idx = train_val_idx[train_sub_idx]
            val_idx = train_val_idx[val_sub_idx]
            splits.append((list(train_idx), list(val_idx)))
    else:
        print(f"\nGenerating random {k}-fold splits...")
        kf = KFold(n_splits=k, shuffle=True, random_state=config['seed'])
        splits = []
        for train_sub_idx, val_sub_idx in kf.split(train_val_idx):
            train_idx = train_val_idx[train_sub_idx]
            val_idx = train_val_idx[val_sub_idx]
            splits.append((list(train_idx), list(val_idx)))
    
    # 运行每一折
    fold_results = []
    model_paths = []
    
    for fold_idx in range(min(k, len(splits))):
        print(f"\n{'='*60}")
        print(f"TRAINING FOLD {fold_idx+1}/{k}")
        print(f"{'='*60}")
        
        try:
            train_idx, val_idx = splits[fold_idx]
            
            # 创建训练器
            trainer = UnifiedKFoldTrainer(
                config=config,
                fold_idx=fold_idx,
                total_folds=k,
                train_indices=train_idx,
                val_indices=val_idx,
                test_indices=test_idx,
                use_scaffold_split=False,  # 已经划分好了
                test_ratio=0  # 已经提供了测试集
            )
            
            # 训练当前折
            best_val, best_test = trainer.train()
            
            # 保存结果
            fold_results.append({
                'fold': fold_idx + 1,
                'val_metric': best_val,
                'test_metric': best_test
            })
            
            # 保存模型路径
            model_paths.append(trainer.model_save_path)
            
            # 复制模型到汇总目录
            fold_model_path = os.path.join(summary_dir, f'fold{fold_idx+1}_best_model.pth')
            shutil.copy(trainer.model_save_path, fold_model_path)
            
            print(f"✓ Fold {fold_idx+1} completed.")
            
        except Exception as e:
            print(f"✗ Error training fold {fold_idx+1}: {e}")
            import traceback
            traceback.print_exc()
            
            fold_results.append({
                'fold': fold_idx + 1,
                'val_metric': np.nan,
                'test_metric': np.nan
            })
            model_paths.append('')
    
    # 计算统计结果
    val_metrics = [r['val_metric'] for r in fold_results if not np.isnan(r['val_metric'])]
    test_metrics = [r['test_metric'] for r in fold_results if not np.isnan(r['test_metric'])]
    
    if len(val_metrics) > 0:
        val_mean = np.mean(val_metrics)
        val_std = np.std(val_metrics)
        val_median = np.median(val_metrics)
        val_min = np.min(val_metrics)
        val_max = np.max(val_metrics)
    else:
        val_mean = val_std = val_median = val_min = val_max = np.nan
    
    if len(test_metrics) > 0:
        test_mean = np.mean(test_metrics)
        test_std = np.std(test_metrics)
        test_median = np.median(test_metrics)
        test_min = np.min(test_metrics)
        test_max = np.max(test_metrics)
    else:
        test_mean = test_std = test_median = test_min = test_max = np.nan
    
    # 打印结果
    print(f"\n{'='*80}")
    print(f"K-FOLD CROSS VALIDATION RESULTS")
    print(f"Task: {task_name.upper()}")
    print(f"K: {k}, Test ratio: {test_ratio}")
    print(f"Metric: {dataset_info['metric'].upper()}")
    print(f"{'='*80}")
    
    metric_name = "AUC" if dataset_info['type'] == 'classification' else "RMSE"
    
    for result in fold_results:
        val_str = f"{result['val_metric']:.4f}" if not np.isnan(result['val_metric']) else "N/A"
        test_str = f"{result['test_metric']:.4f}" if not np.isnan(result['test_metric']) else "N/A"
        print(f"Fold {result['fold']}: Val {metric_name} = {val_str}, Test {metric_name} = {test_str}")
    
    print(f"\nOverall Statistics:")
    print(f"Validation {metric_name}: {val_mean:.4f} ± {val_std:.4f}")
    print(f"  Median: {val_median:.4f}, Range: [{val_min:.4f}, {val_max:.4f}]")
    print(f"Test {metric_name}:       {test_mean:.4f} ± {test_std:.4f}")
    print(f"  Median: {test_median:.4f}, Range: [{test_min:.4f}, {test_max:.4f}]")
    
    # 保存结果到CSV
    df_results = pd.DataFrame(fold_results)
    csv_path = os.path.join(summary_dir, 'kfold_results.csv')
    df_results.to_csv(csv_path, index=False)
    
    # 保存汇总统计
    summary_path = os.path.join(summary_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Task: {task_name}\n")
        f.write(f"Description: {dataset_info['description']}\n")
        f.write(f"Task type: {dataset_info['type']}\n")
        f.write(f"Number of tasks: {dataset_info['num_tasks']}\n")
        f.write(f"Evaluation metric: {dataset_info['metric']}\n")
        f.write(f"K-Fold Cross Validation Results (k={k})\n")
        f.write(f"Independent test set ratio: {test_ratio}\n")
        f.write(f"Seed: {config['seed']}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total samples: {len(dataset)}\n")
        f.write(f"Test set size: {len(test_idx)}\n")
        f.write(f"Train+Validation size: {len(train_val_idx)}\n\n")
        
        f.write("Per-fold results:\n")
        for result in fold_results:
            val_str = f"{result['val_metric']:.4f}" if not np.isnan(result['val_metric']) else "N/A"
            test_str = f"{result['test_metric']:.4f}" if not np.isnan(result['test_metric']) else "N/A"
            f.write(f"Fold {result['fold']}: Val={val_str}, Test={test_str}\n")
        
        f.write(f"\nOverall Statistics:\n")
        f.write(f"Validation {metric_name}: Mean={val_mean:.4f}, Std={val_std:.4f}, Median={val_median:.4f}\n")
        f.write(f"Test {metric_name}:       Mean={test_mean:.4f}, Std={test_std:.4f}, Median={test_median:.4f}\n")
        
        # 性能分析
        if len(val_metrics) > 1:
            if dataset_info['type'] == 'classification':
                # 对于AUC，计算变异系数
                cv_val = (val_std / val_mean * 100) if val_mean != 0 else 0
                cv_test = (test_std / test_mean * 100) if test_mean != 0 else 0
                
                f.write(f"\nStability Analysis (Coefficient of Variation):\n")
                f.write(f"Validation CV: {cv_val:.2f}% ({'Excellent' if cv_val < 5 else 'Good' if cv_val < 10 else 'Moderate' if cv_val < 15 else 'High variability'})\n")
                f.write(f"Test CV: {cv_test:.2f}% ({'Excellent' if cv_test < 5 else 'Good' if cv_test < 10 else 'Moderate' if cv_test < 15 else 'High variability'})\n")
            
            # 分析验证集和测试集差异
            if not np.isnan(val_mean) and not np.isnan(test_mean):
                diff = test_mean - val_mean if dataset_info['type'] == 'classification' else val_mean - test_mean
                f.write(f"\nValidation vs Test Difference: {diff:.4f}\n")
                
                if dataset_info['type'] == 'classification':
                    if diff > 0.05:
                        f.write("Note: Test performance is significantly better than validation.\n")
                        f.write("      This may indicate that the test set is easier or validation set is more challenging.\n")
                    elif diff < -0.05:
                        f.write("Note: Test performance is significantly worse than validation.\n")
                        f.write("      This may indicate overfitting to the validation set.\n")
                    else:
                        f.write("Note: Test and validation performances are consistent.\n")
    
    print(f"\nResults saved to: {summary_dir}")
    
    # 创建集成模型（如果所有折都成功）
    successful_folds = sum(1 for r in fold_results if not np.isnan(r['val_metric']))
    if successful_folds == k:
        print(f"\nCreating ensemble model from {k} folds...")
        try:
            # 只使用有效的模型路径
            valid_model_paths = [p for p in model_paths if p and os.path.exists(p)]
            
            if len(valid_model_paths) >= 2:
                # 保存集成信息
                ensemble_info = {
                    'ensemble_size': len(valid_model_paths),
                    'fold_model_paths': valid_model_paths,
                    'val_mean': val_mean,
                    'val_std': val_std,
                    'test_mean': test_mean,
                    'test_std': test_std,
                    'config': config,
                    'task_name': task_name,
                    'dataset_info': dataset_info
                }
                
                ensemble_path = os.path.join(summary_dir, 'ensemble_info.json')
                with open(ensemble_path, 'w') as f:
                    json.dump(ensemble_info, f, indent=2, default=str)
                
                print(f"Ensemble information saved to: {ensemble_path}")
            else:
                print(f"Not enough valid models for ensemble ({len(valid_model_paths)}/{k})")
                
        except Exception as e:
            print(f"Error creating ensemble: {e}")
    
    return {
        'task_name': task_name,
        'val_mean': val_mean,
        'val_std': val_std,
        'val_median': val_median,
        'test_mean': test_mean,
        'test_std': test_std,
        'test_median': test_median,
        'summary_dir': summary_dir,
        'fold_results': fold_results
    }

# ==================== 批量运行所有数据集 ====================
def run_all_benchmarks(config, benchmark_tasks=None, k=5, test_ratio=0.1):
    """批量运行所有基准测试"""
    
    if benchmark_tasks is None:
        # 默认运行所有数据集
        benchmark_tasks = list(DATASET_INFO.keys())
    
    all_results = {}
    
    print(f"\n{'='*80}")
    print(f"STARTING BENCHMARK EVALUATION (Seed: {config['seed']})")
    print(f"Total tasks: {len(benchmark_tasks)}")
    print(f"K: {k}, Test ratio: {test_ratio}")
    print(f"{'='*80}")
    
    for task_name in benchmark_tasks:
        print(f"\n{'='*80}")
        print(f"BENCHMARK: {task_name.upper()}")
        print(f"{'='*80}")
        
        try:
            # 检查数据集是否在支持列表中
            if task_name.lower() not in DATASET_INFO:
                print(f"⚠️ Warning: {task_name} is not in the supported dataset list.")
                print(f"  Supported datasets: {list(DATASET_INFO.keys())}")
                continue
            
            dataset_info = DATASET_INFO[task_name.lower()]
            
            print(f"Description: {dataset_info['description']}")
            print(f"Task type: {dataset_info['type']}")
            print(f"Number of tasks: {dataset_info['num_tasks']}")
            print(f"Suggested K: {dataset_info['suggested_k']}")
            
            # 使用建议的K值或用户指定的K值
            current_k = dataset_info['suggested_k'] if k is None else k
            
            # 运行K折交叉验证
            results = run_kfold_cross_validation(
                config=config,
                task_name=task_name,
                k=current_k,
                test_ratio=test_ratio
            )
            
            if results is not None:
                all_results[task_name] = results
                
                metric_name = "AUC" if dataset_info['type'] == 'classification' else "RMSE"
                print(f"\n✓ {task_name.upper()} completed!")
                print(f"  Validation {metric_name}: {results['val_mean']:.4f} ± {results['val_std']:.4f}")
                print(f"  Test {metric_name}:       {results['test_mean']:.4f} ± {results['test_std']:.4f}")
            
        except Exception as e:
            print(f"\n✗ {task_name.upper()} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[task_name] = {'error': str(e)}
    
    # 生成最终报告
    generate_final_report(all_results)
    
    return all_results

def generate_final_report(all_results):
    """生成最终报告"""
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK SUMMARY REPORT")
    print(f"{'='*80}")
    
    report_data = []
    
    for task_name, results in all_results.items():
        if isinstance(results, dict) and 'error' in results:
            report_data.append({
                'Dataset': task_name.upper(),
                'Type': 'Error',
                'Tasks': '-',
                'Validation': '-',
                'Test': '-',
                'Status': 'Failed',
                'Note': results['error'][:50] + '...' if len(results['error']) > 50 else results['error']
            })
        else:
            dataset_info = DATASET_INFO.get(task_name.lower(), {})
            metric = dataset_info.get('metric', 'roc_auc').upper()
            
            if dataset_info.get('type') == 'classification':
                val_str = f"{results['val_mean']:.4f} ± {results['val_std']:.4f}"
                test_str = f"{results['test_mean']:.4f} ± {results['test_std']:.4f}"
            else:
                val_str = f"{results['val_mean']:.4f} ± {results['val_std']:.4f}"
                test_str = f"{results['test_mean']:.4f} ± {results['test_std']:.4f}"
            
            report_data.append({
                'Dataset': task_name.upper(),
                'Type': dataset_info.get('type', 'Unknown'),
                'Tasks': dataset_info.get('num_tasks', 1),
                'Validation': val_str,
                'Test': test_str,
                'Status': 'Completed',
                'Note': dataset_info.get('description', '')
            })
    
    # 打印表格
    import pandas as pd
    df = pd.DataFrame(report_data)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)
    
    print(df.to_string(index=False))
    
    # 保存CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f"../train_result/benchmark_summary_{timestamp}.csv"
    df.to_csv(report_path, index=False)
    
    print(f"\nReport saved to: {report_path}")
    
    # 保存详细结果
    detailed_path = f"../train_result/benchmark_details_{timestamp}.json"
    with open(detailed_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"Detailed results saved to: {detailed_path}")

# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description='Unified K-Fold Cross Validation for MoleculeNet')
    parser.add_argument('--task', type=str, default='bbbp', 
                       help='Task name (bbbp, bace, clintox, tox21, toxcast, sider, freesolv, esol, lipophilicity, etc.)')
    parser.add_argument('--ckpt', type=str, default=None, 
                       help='Pretrained checkpoint path')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Initial random seed (will increment 5 times)')
    parser.add_argument('--gpu', type=str, default='0', 
                       help='GPU ID')
    parser.add_argument('--config', type=str, 
                       default='./config/config_finetune.yaml', 
                       help='Config file path')
    parser.add_argument('--kfold', type=int, default=5, 
                       help='Number of folds for cross validation (0 for single split)')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, 
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, 
                       help='Learning rate')
    parser.add_argument('--freeze_epochs', type=int, default=5, 
                       help='Epochs to freeze backbone')
    parser.add_argument('--unfreeze_layers', type=int, default=2, 
                       help='Number of layers to unfreeze in phase 2')
    parser.add_argument('--no_scaffold', action='store_true', 
                       help='Disable scaffold split (use stratified/random K-fold)')
    parser.add_argument('--test_ratio', type=float, default=0.1, 
                       help='Independent test set ratio (0.0-0.3)')
    parser.add_argument('--run_all', action='store_true', 
                       help='Run all benchmark tasks')
    parser.add_argument('--tasks', type=str, nargs='+', 
                       help='List of specific tasks to run')
    parser.add_argument('--task_type', type=str, choices=['classification', 'regression', 'all'], 
                       default='all', help='Type of tasks to run')
    
    args = parser.parse_args()

    # 设置全局变量
    GlobalVar.parallel_train = False
    GlobalVar.dist_bar = [20, 50]
    
    # 1. 读取配置文件
    if os.path.exists(args.config):
        print(f"Loading config from {args.config}")
        with open(args.config, 'r') as f:
            base_config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        print("Warning: Config file not found! Using defaults.")
        base_config = {
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'patience': 20,
            'split_type': 'scaffold' if not args.no_scaffold else 'random',
            'model': {'atom_embed_dim': 512, 'num_kernel': 128, 'layer_num': 6, 'num_heads': 8, 'hidden_size': 2048},
            'optim': {'init_lr': args.lr, 'weight_decay': 1e-4},
            'userconfig': {'mpp': {'dataset_dir': './processed_data1/mpp/pkl', 'split_dir': './processed_data1/mpp/split/'}}
        }
    
    # 2. 命令行参数覆盖
    base_config['task_name'] = args.task
    if args.ckpt is not None:
        base_config['checkpoint'] = args.ckpt
    
    base_config['batch_size'] = args.batch_size
    base_config['epochs'] = args.epochs
    base_config['optim']['init_lr'] = args.lr
    base_config['freeze_epochs'] = args.freeze_epochs
    base_config['unfreeze_layers'] = args.unfreeze_layers
    base_config['use_scaffold_split'] = not args.no_scaffold
    
    # 3. 路径初始化
    base_config['root'] = './processed_data1/mpp/pkl' 
    base_config = get_downstream_task_names(base_config)
    user = 'mpp'
    base_config = config_current_user(user, base_config)
    base_config = config_dataset_form('pkl', base_config)

    # =======================================================
    # 多种子运行逻辑 (Multi-Seed Execution)
    # =======================================================
    initial_seed = args.seed
    num_seed_runs = 5
    seed_results = [] 

    print(f"\n{'#'*80}")
    print(f"MULTI-SEED EXECUTION STARTED")
    print(f"Seeds to run: {[initial_seed + i for i in range(num_seed_runs)]}")
    print(f"{'#'*80}")

    for run_idx in range(num_seed_runs):
        current_seed = initial_seed + run_idx
        print(f"\n>>>>> Running with Seed {current_seed} ({run_idx + 1}/{num_seed_runs}) <<<<<")
        
        config = deepcopy(base_config)
        config['seed'] = current_seed
        set_seed(current_seed)
        
        # 4. 根据参数选择运行模式
        if args.run_all or args.tasks:
            # 批量运行模式 (略过，不在此处处理)
            print("Batch run mode detected, skipping single task logging logic.")
            seed_results.append(None)
            
        elif args.kfold > 0:
            # K折交叉验证
            task_name = args.task.lower()
            try:
                results = run_kfold_cross_validation(
                    config=config,
                    task_name=task_name,
                    k=args.kfold,
                    test_ratio=args.test_ratio
                )
                if results is not None:
                    seed_results.append(results['test_mean'])
                else:
                    seed_results.append(np.nan)
            except Exception as e:
                print(f"Error in seed {current_seed}: {e}")
                seed_results.append(np.nan)
            
        else:
            # 单次划分模式
            print(f"\nRunning single split training for {args.task} with seed {current_seed}...")
            try:
                trainer = UnifiedKFoldTrainer(config=config, fold_idx=None, use_scaffold_split=False)
                best_val, best_test = trainer.train()
                seed_results.append(best_test)
            except Exception as e:
                print(f"Error in seed {current_seed}: {e}")
                seed_results.append(np.nan)

    # =======================================================
    # 跨种子统计与日志写入 (Cross-Seed Statistics & Logging)
    # =======================================================
    # 只有在非批量模式下才执行汇总写入
    if not (args.run_all or args.tasks):
        print(f"\n{'#'*80}")
        print(f"MULTI-SEED SUMMARY FOR TASK: {args.task.upper()}")
        print(f"{'#'*80}")
        
        # 1. 这里的定义非常重要，必须放在 try 块外面
        valid_results = [r for r in seed_results if r is not None and not np.isnan(r)]
        mean_score = 0.0
        std_score = 0.0
        
        # 计算统计量
        if valid_results:
            mean_score = np.mean(valid_results)
            std_score = np.std(valid_results)
        
        # 2. 打印到控制台
        dataset_info = DATASET_INFO.get(args.task.lower(), {})
        metric_name = "AUC" if dataset_info.get('type') == 'classification' else "RMSE"
        
        print(f"Metric: {metric_name}")
        for i, res in enumerate(seed_results):
            seed_val = initial_seed + i
            res_str = f"{res:.4f}" if res is not None and not np.isnan(res) else "Failed"
            print(f"  Seed {seed_val}: {res_str}")
        
        if valid_results:
            print(f"\nFinal Test {metric_name} (Over {len(valid_results)} seeds): {mean_score:.4f} ± {std_score:.4f}")
        else:
            print("\nNo valid results collected.")
        
        # 3. 写入总汇总文件
        master_log_dir = "../train_result"
        os.makedirs(master_log_dir, exist_ok=True)
        master_log_path = os.path.join(master_log_dir, "FINAL_BENCHMARK_SUMMARY.txt")
        
        try:
            with open(master_log_path, 'a') as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Task: {args.task.upper()}\n")
                f.write(f"GPU: {args.gpu}\n")
                f.write(f"Mode: {'K-Fold ('+str(args.kfold)+')' if args.kfold > 0 else 'Single Split'}\n")
                f.write(f"Config: Epochs={args.epochs}, Batch={args.batch_size}, LR={args.lr}\n")
                f.write(f"{'-'*30}\n")
                
                for i, res in enumerate(seed_results):
                    seed_val = initial_seed + i
                    res_str = f"{res:.4f}" if res is not None and not np.isnan(res) else "Failed"
                    f.write(f"Seed {seed_val}: {res_str}\n")
                
                f.write(f"{'-'*30}\n")
                
                if valid_results:
                    f.write(f"FINAL RESULT ({metric_name}): {mean_score:.4f} ± {std_score:.4f}\n")
                else:
                    f.write(f"FINAL RESULT: ALL FAILED\n")
                f.write(f"{'='*50}\n")
                
            print(f"\n[Success] Summary appended to: {master_log_path}")
            
        except Exception as e:
            # 这里打印错误，但不会崩溃
            print(f"\n[Warning] Failed to write master log: {e}")

if __name__ == '__main__':
    main()