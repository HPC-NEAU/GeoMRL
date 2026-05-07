import argparse
import sys
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial

# 1. 路径修复与模型引入
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from models.GeoMRL import GeoMRL
from data_process.function_group_constant import nfg
from data_process.compound_tools import CompoundKit
from utils.public_util import set_seed, EarlyStopping

warnings.filterwarnings("ignore")

# ==================== 1. 序列解析与并行处理 (保持不变) ====================
def read_fasta(fasta_path):
    seq_dict = {}
    with open(fasta_path, 'r') as f:
        prot_id = ""
        seq = []
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if prot_id: seq_dict[prot_id] = "".join(seq)
                prot_id = line[1:].split()[0] 
                seq = []
            else:
                seq.append(line)
        if prot_id: seq_dict[prot_id] = "".join(seq)
    return seq_dict

def sequence_to_tensor_global(seq, max_len=1000):
    chars = "ARNDCQEGHILKMFPSTWYV" 
    vocab = {char: i + 1 for i, char in enumerate(chars)}
    seq_idx = [vocab.get(char.upper(), 0) for char in str(seq)[:max_len]]
    if len(seq_idx) < max_len:
        seq_idx += [0] * (max_len - len(seq_idx))
    return torch.tensor(seq_idx, dtype=torch.long)

def process_single_ppi_row(row_data, fasta_dict):
    prot_a_id, prot_b_id, label = row_data
    seq_a = fasta_dict.get(prot_a_id)
    seq_b = fasta_dict.get(prot_b_id)
    if seq_a is None or seq_b is None: return None
    return {
        'label': float(label),
        'protein_a_seq': sequence_to_tensor_global(seq_a, max_len=1000),
        'protein_b_seq': sequence_to_tensor_global(seq_b, max_len=1000)
    }

# ==================== 2. Dataset 类 (增加 Rank 打印控制) ====================
class PPIDataset(Dataset):
    def __init__(self, tsv_path, fasta_path, rank=0):
        self.tsv_path = tsv_path
        self.fasta_path = fasta_path
        self.rank = rank
        
        cache_file = tsv_path.replace('.tsv', '_ppi_parallel_cache.pkl')
        if os.path.exists(cache_file):
            if self.rank == 0: print(f"[*] Loading cached data from {cache_file}...")
            with open(cache_file, 'rb') as f:
                self.processed_data = pickle.load(f)
        else:
            if self.rank == 0: print(f"[*] No cache found. Processing TSV and FASTA files...")
            self.processed_data = self._process_parallel()
            if self.rank == 0:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.processed_data, f)
                print(f"[*] Data cached to {cache_file}")

    def _process_parallel(self):
        if self.rank == 0: print(f"-> Reading FASTA: {self.fasta_path}")
        fasta_dict = read_fasta(self.fasta_path)
        
        df = pd.read_csv(self.tsv_path, sep='\t', header=None) 
        rows = [(row[0], row[1], row[2]) for _, row in df.iterrows()]
        processed_list = []
        num_workers = min(cpu_count(), 50)
        
        with Pool(processes=num_workers) as pool:
            func = partial(process_single_ppi_row, fasta_dict=fasta_dict)
            # 只有主进程(rank 0)打印进度条
            iterator = tqdm(pool.imap_unordered(func, rows), total=len(rows), desc="Processing") if self.rank == 0 else pool.imap_unordered(func, rows)
            for result in iterator:
                if result is not None:
                    processed_list.append(result)
        return processed_list

    def __len__(self): return len(self.processed_data)
    def __getitem__(self, idx): return self.processed_data[idx]

def collator_ppi(batch):
    batched_data = {'label': torch.tensor([item['label'] for item in batch], dtype=torch.float32)}
    batched_data['protein_a_seq'] = torch.stack([b['protein_a_seq'] for b in batch])
    batched_data['protein_b_seq'] = torch.stack([b['protein_b_seq'] for b in batch])
    return batched_data

# ==================== 3. DDP 训练器 ====================
class PPITrainerDDP(object):
    def __init__(self, config):
        self.config = config
        
        # 获取分布式环境变量
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # 设置当前进程的 GPU
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device('cuda', self.local_rank)
        
        self.save_dir = f"./train_result/ppi_ddp_result/"
        if self.global_rank == 0:
            os.makedirs(self.save_dir, exist_ok=True)
            
        self.train_loader, self.val_loader, self.test_loader = self.get_dataloaders()
        
        self.model = self.get_model()
        # 将模型包裹进 DDP
        self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.best_auc = 0.0

    def get_dataloaders(self):
        dataset = PPIDataset(self.config['data_root'], self.config['fasta_root'], rank=self.global_rank)
        
        # 分割数据集
        indices = list(range(len(dataset)))
        # 为了保证多卡的切分一致，必须设置固定种子
        np.random.seed(42)
        np.random.shuffle(indices)
        
        t_end, v_end = int(len(dataset)*0.8), int(len(dataset)*0.9)
        train_ds = Subset(dataset, indices[:t_end])
        val_ds = Subset(dataset, indices[t_end:v_end])
        test_ds = Subset(dataset, indices[v_end:])
        
        # [核心] 使用 DistributedSampler，让每张卡读不同的数据块
        train_sampler = DistributedSampler(train_ds, num_replicas=self.world_size, rank=self.global_rank, shuffle=True)
        # 测试和验证集为了计算指标方便，通常只在一张卡(或聚合后)跑，这里简便起见不在 val/test 使用 DDP 采样
        val_sampler = DistributedSampler(val_ds, num_replicas=self.world_size, rank=self.global_rank, shuffle=False)
        
        args = dict(batch_size=self.config['batch_size'], num_workers=4, collate_fn=collator_ppi, pin_memory=True)
        return DataLoader(train_ds, sampler=train_sampler, **args), \
               DataLoader(val_ds, sampler=val_sampler, **args), \
               DataLoader(test_ds, shuffle=False, batch_size=self.config['batch_size'], num_workers=4, collate_fn=collator_ppi)

    def get_model(self):
        model = GeoMRL(
            mode='ppi', 
            atom_names=CompoundKit.atom_vocab_dict.keys(),
            atom_embed_dim=512, num_kernel=128, layer_num=6, num_heads=8,
            atom_FG_class=nfg()+1, hidden_size=2048, num_tasks=1,
            ablation=False, use_gating=True
        ).to(self.device)
        return model

    def train(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'], weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=10)
        stopper = EarlyStopping(mode='higher', patience=20)
        
        if self.global_rank == 0:
            print(f"\n>>> Starting DDP PPI Training on {self.world_size} GPUs...")
            
        for epoch in range(1, self.config['epochs'] + 1):
            # 必须调用 set_epoch 确保打乱
            self.train_loader.sampler.set_epoch(epoch)
            self.model.train()
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}") if self.global_rank == 0 else self.train_loader
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                logits = self.model(batch)['affinity'].squeeze()
                loss = self.criterion(logits, batch['label'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                if self.global_rank == 0:
                    pbar.set_postfix(Loss=f"{loss.item():.4f}")
            
            # --- 分布式验证评估 ---
            # 只有主进程(rank 0)进行全局的测试和保存
            if self.global_rank == 0:
                val_metrics = self.evaluate(self.val_loader)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch} | LR: {current_lr:.2e} | Val AUC: {val_metrics['AUC']:.4f} | Val ACC: {val_metrics['ACC']:.4f}")
                
                scheduler.step(val_metrics['AUC'])
                
                if val_metrics['AUC'] > self.best_auc:
                    self.best_auc = val_metrics['AUC']
                    # 保存 DDP 模型需要保存 model.module
                    torch.save(self.model.module.state_dict(), os.path.join(self.save_dir, "best_ppi_model.pth"))
                
                if stopper.step(val_metrics['AUC'], self.model): 
                    break
                    
            # 确保其他进程等待主进程处理完毕
            dist.barrier()

        if self.global_rank == 0:
            print("\n[*] Training Complete. Running Final Test...")
            self.model.module.load_state_dict(torch.load(os.path.join(self.save_dir, "best_ppi_model.pth")))
            test_metrics = self.evaluate(self.test_loader, save_tsne=True)
            print(f"\n================ FINAL TEST RESULTS ================")
            print(f"Test AUC  : {test_metrics['AUC']:.4f}")
            print(f"Test AUPR : {test_metrics['AUPR']:.4f}")
            print(f"Test ACC  : {test_metrics['ACC']:.4f}")
            print(f"Test F1   : {test_metrics['F1']:.4f}")
            print(f"====================================================")

    def evaluate(self, loader, save_tsne=False):
        self.model.eval()
        probs, labels, feats = [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = self.model.module(batch) 
                p = torch.sigmoid(out['affinity'].squeeze()).cpu().numpy()
                probs.extend(p)
                labels.extend(batch['label'].cpu().numpy())
                if save_tsne and 'latent_feature' in out: 
                    feats.extend(out['latent_feature'].cpu().numpy())
        
        if save_tsne and len(feats) > 0:
            np.save(os.path.join(self.save_dir, "ppi_tsne_feats.npy"), np.array(feats))
            np.save(os.path.join(self.save_dir, "ppi_tsne_labels.npy"), np.array(labels))
            print(f"[*] t-SNE data saved to {self.save_dir}")

        probs = np.array(probs)
        labels = np.array(labels)
        preds = (probs >= 0.5).astype(int)
        
        try:
            return {
                'AUC': roc_auc_score(labels, probs),
                'AUPR': average_precision_score(labels, probs),
                'ACC': accuracy_score(labels, preds),
                'F1': f1_score(labels, preds, zero_division=0)
            }
        except ValueError:
            return {'AUC': 0.0, 'AUPR': 0.0, 'ACC': 0.0, 'F1': 0.0}

# ==================== 4. Main ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to pairs TSV')
    parser.add_argument('--fasta_root', type=str, required=True, help='Path to seqs FASTA')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32) # 注意: 这里是单卡的 batch_size
    parser.add_argument('--epochs', type=int, default=100)
    
    # 增加 local_rank 参数，DDP 会自动传进来
    parser.add_argument('--local_rank', type=int, default=os.environ.get('LOCAL_RANK', 0))
    args = parser.parse_args()

    # 初始化分布式环境
    dist.init_process_group(backend='nccl')

    set_seed(42) 

    config = vars(args)
    config['model'] = {'atom_embed_dim': 512, 'hidden_size': 2048, 'layer_num': 6, 'num_heads': 8, 'num_kernel': 128}
    
    PPITrainerDDP(config).train()
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()