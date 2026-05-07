import argparse
import sys
import os
import shutil
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import warnings

# 1. 路径修复
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 2. 引入模型与工具
from models.GeoMRL import GeoMRL
from data_process.data_collator_dti import collator_dti 
from data_process.function_group_constant import nfg
from data_process.compound_tools import CompoundKit
from utils.public_util import set_seed, EarlyStopping
from utils.global_var_util import GlobalVar

warnings.filterwarnings("ignore")

# ==================== 标签缩放器 ====================
class LabelScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def fit(self, dataset, indices):
        labels = []
        print("[Scaler] Computing mean/std from training set...")
        for i in tqdm(indices, desc="Fitting Scaler", leave=False):
            item = dataset[i]
            labels.append(float(item['label']))
        self.mean = np.mean(labels)
        self.std = np.std(labels)
        if self.std == 0: self.std = 1.0
        print(f"[Scaler] Train Mean: {self.mean:.4f}, Std: {self.std:.4f}")

    def transform(self, labels):
        if self.mean is None: return labels
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        labels = labels.to(self.device)
        return (labels - self.mean) / self.std

    def inverse_transform(self, preds):
        if self.mean is None: return preds
        return preds * self.std + self.mean

# ==================== 数据集类 ====================
class DTIDataset(Dataset):
    def __init__(self, data_or_path):
        if isinstance(data_or_path, str):
            print(f"Loading {data_or_path}...")
            with open(data_or_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = data_or_path

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# ==================== 训练器类 ====================
class DTITrainer(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # [修复] 1. 先初始化日志路径和文件，确保 log 函数立即可用
        self.save_dir = f"./train_result/dti_result/"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 定义日志文件路径
        self.log_file = os.path.join(self.save_dir, "training_debug_log.txt1")
        # 清空旧日志，写入开头
        with open(self.log_file, 'w') as f:
            f.write(f"Training Log for {config['task_name']}\n============================\n")
            
        self.writer = SummaryWriter(os.path.join(self.save_dir, config['task_name']))
        
        # 2. 初始化 Scaler
        self.scaler = LabelScaler()
        
        # 3. 再加载数据 (现在 get_dataloaders 里的 self.log() 可以正常工作了)
        self.train_loader, self.val_loader, self.test_loader = self.get_dataloaders()
        
        # 4. 加载模型
        self.model = self.get_model()
        self.criterion = nn.SmoothL1Loss()
        
        self.best_rmse = float('inf')
        self.best_metrics = {}

    def log(self, message):
        """辅助函数：同时打印到控制台和写入 debug 日志文件"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")

    def get_dataloaders(self):
        pkl_path = self.config['data_root']
        self.log(f"Loading data from {pkl_path}...")
        
        with open(pkl_path, 'rb') as f:
            full_data = pickle.load(f)
        
        valid_data = [d for d in full_data if d is not None]
        dataset = DTIDataset(valid_data)
        total_len = len(dataset)
        self.log(f"Total valid samples: {total_len}")
        
        indices = list(range(total_len))
        np.random.shuffle(indices) 
        
        train_end = int(total_len * 0.8)
        val_end = int(total_len * 0.9)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Fit Scaler
        self.scaler.fit(dataset, train_indices)
        
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
            num_tasks=1
        ).to(self.device)

        if self.config['ckpt']:
            self.log(f"Loading Pretrain: {self.config['ckpt']}")
            ckpt = torch.load(self.config['ckpt'], map_location=self.device)
            state_dict = ckpt['model'] if 'model' in ckpt else ckpt
            
            model_dict = model.state_dict()
            pretrained = {k: v for k, v in state_dict.items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained)
            model.load_state_dict(model_dict, strict=False)
            self.log(f"Loaded {len(pretrained)} pretrained layers.")
            
        return model

    def train(self):
        finetune_lr = self.config['lr']
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=finetune_lr, weight_decay=1e-3)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=10, verbose=True
        )
        
        stopper = EarlyStopping(mode='lower', patience=25)
        self.log(f"\n>>> Starting Training with LR={finetune_lr}...")
        
        for epoch in range(1, self.config['epochs'] + 1):
            self.model.train()
            loss_epoch = 0
            
            # 使用 leave=False 确保进度条在结束后自动消失，减少干扰
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False, disable=False)
            for batch in pbar:
                if not batch: continue
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                self.optimizer.zero_grad()
                pred = self.model(batch)['affinity'].squeeze()
                label = batch['label']
                target_scaled = self.scaler.transform(label)
                
                if torch.isnan(pred).any():
                    self.log(f"!!! Error: NaN detected at Epoch {epoch} !!!")
                    return 

                loss = self.criterion(pred, target_scaled)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                loss_epoch += loss.item()
                # 移除 pbar.set_description 里的多余打印，只保留 Loss
                pbar.set_postfix(Loss=f"{loss.item():.4f}")
            
            # --- 评估环节 ---
            # 评估时不再打印每个 batch 的 debug 信息，只在最后汇总打印一次
            val_metrics = self.evaluate(self.val_loader, prefix="Val", epoch=epoch)
            test_metrics = self.evaluate(self.test_loader, prefix="Test", epoch=epoch)
            
            val_rmse = val_metrics['RMSE']
            self.scheduler.step(val_rmse)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 最终的一行简洁汇总日志
            log_msg = (f"Epoch {epoch:03d} | LR: {current_lr:.2e} | "
                       f"Val RMSE: {val_rmse:.4f} | Test RMSE: {test_metrics['RMSE']:.4f} | "
                       f"Pearson: {test_metrics['Pearson']:.4f}")
            self.log(log_msg)
            
            if val_rmse < self.best_rmse:
                self.best_rmse = val_rmse
                self.best_metrics = test_metrics
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_dti_model.pth"))
                self.log("--> New Best Performance Saved!")
                
            if stopper.step(val_rmse, self.model):
                self.log(f"Early Stopping at epoch {epoch}")
                break
        
        self.save_result_to_txt()

    def evaluate(self, loader, prefix="Val", epoch=0):
        self.model.eval()
        preds, labels = [], []
        
        # 将 Debug 信息存入变量，而不是直接 print
        first_batch_debug = ""
        
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if not batch: continue
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                pred_scaled = self.model(batch)['affinity'].squeeze()
                pred_raw = self.scaler.inverse_transform(pred_scaled)
                
                # 只保留第一组数据做 Debug
                if i == 0:
                    p_np = pred_raw[:5].cpu().numpy()
                    l_np = batch['label'][:5].cpu().numpy()
                    first_batch_debug = (f"DEBUG {prefix} | Pred: {np.round(p_np, 2)} | Label: {np.round(l_np, 2)}")
                
                preds.extend(pred_raw.cpu().numpy())
                labels.extend(batch['label'].cpu().numpy())
        
        # 评估结束后的日志记录（不会干扰 tqdm）
        if first_batch_debug:
            # 你可以选择只在 training_debug_log.txt 里看，不打印在屏幕上
            with open(self.log_file, 'a') as f:
                f.write(first_batch_debug + "\n")
        
        preds = np.array(preds)
        labels = np.array(labels)
        
        if np.isnan(preds).all(): return {'RMSE': 999.0, 'Pearson': 0.0}

        rmse = np.sqrt(((preds - labels) ** 2).mean())
        try:
            if np.std(preds) < 1e-6: pearson = 0.0
            else: pearson = pearsonr(preds, labels)[0]
        except: pearson = 0.0
            
        return {'RMSE': rmse, 'Pearson': pearson}

    def save_result_to_txt(self):
        result_file = os.path.join(self.save_dir, "dti_results_summary.txt")
        m = self.best_metrics
        with open(result_file, "a") as f:
            f.write(f"\n{'='*40}\n")
            f.write(f"Task: {self.config['task_name']}\n")
            f.write(f"Best Val RMSE: {self.best_rmse:.4f}\n")
            f.write(f"Pearson: {m.get('Pearson', 0):.4f}\n")
            f.write(f"RMSE: {m.get('RMSE', 0):.4f}\n")
            f.write(f"{'='*40}\n")
        self.log(f"\nDetailed results appended to {result_file}")

# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to .pkl data file')
    parser.add_argument('--ckpt', type=str, default=None, help='Pretrained model path')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for finetuning')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=250)
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(args.seed) 
    GlobalVar.dist_bar = [20, 50]

    config = {
        'data_root': args.data_root,
        'ckpt': args.ckpt,
        'task_name': 'pdbbind',
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
    
    print(f"Starting Training on GPU {args.gpu}...")
    trainer = DTITrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()