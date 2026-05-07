import argparse
import sys
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, matthews_corrcoef, roc_auc_score, 
                             average_precision_score)
import warnings

# 引入你的模型与工具 (请确保路径正确)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from models.GeoMRL import GeoMRL
from data_process.data_collator_dti import collator_dti 
from data_process.function_group_constant import nfg
from data_process.compound_tools import CompoundKit
from utils.public_util import set_seed, EarlyStopping
from utils.global_var_util import GlobalVar
import pandas as pd
from data_process.compound_tools import CompoundKit

warnings.filterwarnings("ignore")

# ==================== 数据集类 ====================
class CSV_DTIDataset(Dataset):
    def __init__(self, csv_path):
        print(f"Loading CSV data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # 确保你的 CSV 文件中有这三列：'smiles', 'sequence', 'label'
        assert 'smiles' in self.df.columns, "CSV 必须包含 'smiles' 列"
        assert 'sequence' in self.df.columns, "CSV 必须包含 'sequence' 列"
        assert 'label' in self.df.columns, "CSV 必须包含 'label' 列"

    def __len__(self): 
        return len(self.df)
        
    def __getitem__(self, idx): 
        row = self.df.iloc[idx]
        smiles = row['smiles']
        prot_seq = row['sequence']
        label = float(row['label'])
        
        # =======================================================
        # 核心步骤：将 SMILES 和 Sequence 实时转化为模型需要的 Tensor
        # =======================================================
        
        # 1. 提取小分子的特征和 3D 坐标 (MMFF)
        # 注意：你需要根据你实际的 CompoundKit 方法名进行调用
        # 这里假设 CompoundKit.smiles_to_graph 会返回包含 atom_feature 和 pos 的字典
        ligand_data = CompoundKit.smiles_to_graph(smiles) 
        
        # 2. 提取蛋白质特征 (将字母转化为数字索引)
        # 假设你有类似的方法将序列转为 tensor
        prot_data = self._sequence_to_tensor(prot_seq)
        
        # 3. 组合为一个样本字典
        sample = {
            'atom_feature': ligand_data['atom_feature'],
            'atom_pos': ligand_data['pos'],
            'protein_x': prot_data,
            'label': torch.tensor(label, dtype=torch.float32)
        }
        
        return sample

    def _sequence_to_tensor(self, seq):
        """简单的氨基酸序列转数字索引示例 (你需要根据你实际的词表调整)"""
        vocab = { "A":1, "C":2, "D":3, "E":4, "F":5, "G":6, "H":7, "I":8, "K":9, "L":10, 
                  "M":11, "N":12, "P":13, "Q":14, "R":15, "S":16, "T":17, "V":18, "W":19, "Y":20 }
        seq_idx = [vocab.get(aa, 0) for aa in seq]
        return torch.tensor(seq_idx, dtype=torch.long)

# ==================== 分类训练器类 ====================
class DTIClassifierTrainer(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.save_dir = f"./train_result/dti_classification_result/"
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.save_dir, f"{config['task_name']}_training_log.txt")
        with open(self.log_file, 'w') as f:
            f.write(f"Classification Training Log for {config['task_name']}\n{'='*40}\n")
            
        self.train_loader, self.val_loader, self.test_loader = self.get_dataloaders()
        self.model = self.get_model()
        
        # 二分类任务的损失函数：自带 Sigmoid，数值更稳定
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 记录最佳模型 (以 AUC 为主要评价标准)
        self.best_auc = 0.0
        self.best_metrics = {}

    def log(self, message):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")

    def get_dataloaders(self):
        pkl_path = self.config['data_root']
        with open(pkl_path, 'rb') as f:
            full_data = pickle.load(f)
        
        valid_data = [d for d in full_data if d is not None]
        dataset = CSV_DTIDataset(pkl_path)
        total_len = len(dataset)
        self.log(f"Total valid samples: {total_len}")
        
        # 8:1:1 随机划分
        indices = list(range(total_len))
        np.random.shuffle(indices) 
        
        train_end = int(total_len * 0.8)
        val_end = int(total_len * 0.9)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        self.log(f"Split sizes -> Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
        
        args = dict(batch_size=self.config['batch_size'], num_workers=4, collate_fn=collator_dti)
        
        return (DataLoader(train_dataset, shuffle=True, **args),
                DataLoader(val_dataset, shuffle=False, **args),
                DataLoader(test_dataset, shuffle=False, **args))

    def get_model(self):
        model = GeoMRL(
            mode='dti',
            atom_names=CompoundKit.atom_vocab_dict.keys(),
            atom_embed_dim=self.config['model']['atom_embed_dim'],
            num_kernel=self.config['model']['num_kernel'], 
            layer_num=self.config['model']['layer_num'],
            num_heads=self.config['model']['num_heads'],
            atom_FG_class=nfg() + 1,
            hidden_size=self.config['model']['hidden_size'],
            num_tasks=1 # 输出维度为1，BCEWithLogitsLoss 会自动处理它
        ).to(self.device)

        if self.config['ckpt']:
            ckpt = torch.load(self.config['ckpt'], map_location=self.device)
            state_dict = ckpt['model'] if 'model' in ckpt else ckpt
            model_dict = model.state_dict()
            pretrained = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained)
            model.load_state_dict(model_dict, strict=False)
            self.log(f"Loaded {len(pretrained)} pretrained layers.")
            
        return model

    def train(self):
        # 对新加入的 DTI 模块使用更大的学习率
        new_params = [p for n, p in self.model.named_parameters() if "prot_encoder" in n or "cross_" in n or "dti_head" in n]
        old_params = [p for n, p in self.model.named_parameters() if not any(x in n for x in ["prot_encoder", "cross_", "dti_head"])]

        self.optimizer = torch.optim.AdamW([
            {'params': old_params, 'lr': self.config['lr']},
            {'params': new_params, 'lr': self.config['lr'] * 10} # 差异化学习率
        ], weight_decay=1e-3)
        
        # 使用 AUC 进行学习率衰减 (mode='max')
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.8, patience=10, verbose=True
        )
        
        stopper = EarlyStopping(mode='higher', patience=25) # AUC 越高越好
        
        self.log("\n>>> Starting Classification Training...")
        
        for epoch in range(1, self.config['epochs'] + 1):
            self.model.train()
            loss_epoch = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
            for batch in pbar:
                if not batch: continue
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                self.optimizer.zero_grad()
                logits = self.model(batch)['affinity'].squeeze()
                
                # 确保标签是 Float 并且形状与 logits 一致
                labels = batch['label'].float().squeeze() 
                
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                loss_epoch += loss.item()
                pbar.set_postfix(Loss=f"{loss.item():.4f}")
            
            # 评估
            val_metrics = self.evaluate(self.val_loader)
            test_metrics = self.evaluate(self.test_loader)
            
            val_auc = val_metrics['AUC']
            self.scheduler.step(val_auc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            log_msg = (f"Epoch {epoch:03d} | LR: {current_lr:.2e} | "
                       f"Val AUC: {val_auc:.4f} | Test AUC: {test_metrics['AUC']:.4f} | "
                       f"Test AUPR: {test_metrics['AUPR']:.4f} | Test ACC: {test_metrics['ACC']:.4f}")
            self.log(log_msg)
            
            # 保存最佳模型
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.best_metrics = test_metrics
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_dti_classifier.pth"))
                self.log("--> New Best Model Saved (Highest Val AUC)!")
                
            if stopper.step(val_auc, self.model):
                self.log(f"Early Stopping at epoch {epoch}")
                break
        
        self.save_final_results()

    def evaluate(self, loader):
        self.model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                if not batch: continue
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # 获取 Logits 并通过 Sigmoid 转换为概率 (0~1)
                logits = self.model(batch)['affinity'].squeeze()
                probs = torch.sigmoid(logits)
                
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
                
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # 将概率转化为二分类预测值 (阈值为 0.5)
        all_preds = (all_probs >= 0.5).astype(int)
        
        # 计算 7 大分类指标
        metrics = {}
        try:
            metrics['ACC'] = accuracy_score(all_labels, all_preds)
            metrics['Precision'] = precision_score(all_labels, all_preds, zero_division=0)
            metrics['Recall'] = recall_score(all_labels, all_preds, zero_division=0)
            metrics['F1'] = f1_score(all_labels, all_preds, zero_division=0)
            metrics['MCC'] = matthews_corrcoef(all_labels, all_preds)
            # AUC 和 AUPR 需要输入连续的概率值
            metrics['AUC'] = roc_auc_score(all_labels, all_probs)
            metrics['AUPR'] = average_precision_score(all_labels, all_probs)
        except ValueError:
            # 防止某个 batch 只有单类别标签报错
            metrics = {k: 0.0 for k in ['ACC', 'Precision', 'Recall', 'F1', 'MCC', 'AUC', 'AUPR']}
            
        return metrics

    def save_final_results(self):
        result_file = os.path.join(self.save_dir, "classification_metrics_summary.txt")
        m = self.best_metrics
        with open(result_file, "a") as f:
            f.write(f"\n{'='*40}\n")
            f.write(f"Dataset: {self.config['task_name']}\n")
            f.write(f"Best Val AUC: {self.best_auc:.4f}\n")
            f.write(f"Test AUC:  {m.get('AUC', 0):.4f}\n")
            f.write(f"Test AUPR: {m.get('AUPR', 0):.4f}\n")
            f.write(f"Test ACC:  {m.get('ACC', 0):.4f}\n")
            f.write(f"Test F1:   {m.get('F1', 0):.4f}\n")
            f.write(f"Test MCC:  {m.get('MCC', 0):.4f}\n")
            f.write(f"Test Prec: {m.get('Precision', 0):.4f}\n")
            f.write(f"Test Rec:  {m.get('Recall', 0):.4f}\n")
            f.write(f"{'='*40}\n")
        self.log(f"\nFinal Test Metrics saved to {result_file}")

# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to .pkl data file')
    parser.add_argument('--dataset_name', type=str, default='DrugBank', help='DrugBank, Davis, or KIBA')
    parser.add_argument('--ckpt', type=str, default=None, help='Pretrained model path')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(args.seed) 
    GlobalVar.dist_bar = [20, 50]

    config = {
        'data_root': args.data_root,
        'ckpt': args.ckpt,
        'task_name': args.dataset_name,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'model': {
            'atom_embed_dim': 512, 
            'hidden_size': 2048, 
            'layer_num': 6, 
            'num_heads': 8, 
            'num_kernel': 128
        }
    }
    
    print(f"Starting Classification Training for {args.dataset_name} on GPU {args.gpu}...")
    trainer = DTIClassifierTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()