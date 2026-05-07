import argparse
import sys
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# 【核心修改 1】：引入了你需要的四大评价指标！
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem
from multiprocessing import Pool, cpu_count
from functools import partial

# 1. 路径修复与模型引入
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from models.GeoMRL import GeoMRL
from data_process.function_group_constant import nfg
from data_process.compound_tools import CompoundKit
from utils.public_util import set_seed, EarlyStopping
from utils.global_var_util import GlobalVar

warnings.filterwarnings("ignore")

# ==================== 1. 并行处理全局函数 ====================
def sequence_to_tensor_global(seq, max_len=150):
    chars = "CNOfsClBrPHI#=()[]+-ARNDCQEGHILKMFPSTWYV"
    vocab = {char: i + 1 for i, char in enumerate(chars)}
    seq_idx = [vocab.get(char, 0) for char in str(seq)[:max_len]]
    if len(seq_idx) < max_len:
        seq_idx += [0] * (max_len - len(seq_idx))
    return torch.tensor(seq_idx, dtype=torch.long)

def _get_atom_features_internal(mol):
    all_atom_features = []
    for atom in mol.GetAtoms():
        features = [
            int(atom.GetAtomicNum()), int(atom.GetDegree()),
            int(atom.GetFormalCharge() + 5), int(atom.GetChiralTag()),
            int(atom.GetTotalNumHs()), int(atom.GetHybridization()),
            int(atom.GetIsAromatic())
        ]
        all_atom_features.append(features)
    return torch.tensor(all_atom_features, dtype=torch.long)

def process_single_row_parallel(row_data, task_type):
    row, idx = row_data
    row_keys = {str(k).strip().lower(): v for k, v in row.items()}
    
    try:
        if task_type == 'ddi':
            drug_a_smi = row_keys.get('drug_a')
            drug_b_smi = row_keys.get('drug_b')
            
            raw_label = float(row_keys.get('ddi', 0))
            label = 1.0 if raw_label > 0 else 0.0 
            
            if drug_a_smi is None or drug_b_smi is None: return None
            
            mol_a = Chem.MolFromSmiles(drug_a_smi)
            if mol_a is None: return None
            
            mol_a = Chem.AddHs(mol_a)
            if AllChem.EmbedMolecule(mol_a, randomSeed=42, useExpTorsionAnglePrefs=True, useBasicKnowledge=True) < 0:
                AllChem.EmbedMolecule(mol_a, randomSeed=42)
            if mol_a.GetNumConformers() == 0: return None
            
            AllChem.MMFFOptimizeMolecule(mol_a)
            mol_a = Chem.RemoveHs(mol_a)
            
            return {
                'label': label,
                'atom_feature': _get_atom_features_internal(mol_a),
                'atom_pos': torch.tensor(mol_a.GetConformer().GetPositions(), dtype=torch.float32),
                'drug_b_tokens': sequence_to_tensor_global(drug_b_smi, max_len=150)
            }
            
        elif task_type == 'ppi':
            prot_a = row_keys.get('protein_a')
            prot_b = row_keys.get('protein_b')
            
            raw_label = float(row_keys.get('label', row_keys.get('ppi', 0)))
            label = 1.0 if raw_label > 0 else 0.0
            
            if prot_a is None or prot_b is None: return None
            
            return {
                'label': label,
                'protein_a_seq': sequence_to_tensor_global(prot_a, max_len=1000),
                'protein_b_seq': sequence_to_tensor_global(prot_b, max_len=1000)
            }
    except Exception:
        return None
    return None

# ==================== 2. Dataset 类 ====================
class InteractCSVDataset(Dataset):
    def __init__(self, csv_path, task_type):
        self.csv_path = csv_path
        self.task_type = task_type
        self.df = pd.read_csv(csv_path)
        
        cache_file = csv_path.replace('.csv', f'_{task_type}_parallel_cache.pkl')
        if os.path.exists(cache_file):
            print(f"[*] Loading cached data from {cache_file}...")
            with open(cache_file, 'rb') as f:
                self.processed_data = pickle.load(f)
        else:
            print(f"[*] No cache! Using {min(cpu_count(), 50)} cores to process {len(self.df)} samples...")
            self.processed_data = self._process_parallel()
            with open(cache_file, 'wb') as f:
                pickle.dump(self.processed_data, f)
            print(f"[*] Data cached to {cache_file}")

    def _process_parallel(self):
        rows = [(self.df.iloc[i].to_dict(), i) for i in range(len(self.df))]
        processed_list = []
        num_workers = min(cpu_count(), 50)
        
        with Pool(processes=num_workers) as pool:
            func = partial(process_single_row_parallel, task_type=self.task_type)
            for result in tqdm(pool.imap_unordered(func, rows), total=len(rows), desc="Parallel Processing"):
                if result is not None:
                    processed_list.append(result)
        return processed_list

    def __len__(self): return len(self.processed_data)
    def __getitem__(self, idx): return self.processed_data[idx]

# ==================== 3. Collator & Trainer ====================
def collator_interact(batch):
    from torch.nn.utils.rnn import pad_sequence
    batched_data = {'label': torch.tensor([item['label'] for item in batch], dtype=torch.float32)}
    
    if 'atom_feature' in batch[0]: 
        batched_data['atom_feature'] = pad_sequence([b['atom_feature'] for b in batch], batch_first=True, padding_value=0)
        batched_data['atom_pos'] = pad_sequence([b['atom_pos'] for b in batch], batch_first=True, padding_value=0)
        batched_data['drug_b_tokens'] = torch.stack([b['drug_b_tokens'] for b in batch])
    else: 
        batched_data['protein_a_seq'] = torch.stack([b['protein_a_seq'] for b in batch])
        batched_data['protein_b_seq'] = torch.stack([b['protein_b_seq'] for b in batch])
    return batched_data

class InteractTrainer(object):
    def __init__(self, config):
        self.config = config
        self.task = config['task']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        suffix = ""
        if config.get('ablation', False): suffix += "_noRBF"
        if config.get('no_gating', False): suffix += "_noGate"
        
        self.save_dir = f"./train_result/{self.task}_result{suffix}/"
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.train_loader, self.val_loader, self.test_loader = self.get_dataloaders()
        self.model = self.get_model()
        self.criterion = nn.BCEWithLogitsLoss()
        self.best_auc = 0.0

    def get_dataloaders(self):
        dataset = InteractCSVDataset(self.config['data_root'], self.task)
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        
        t_end, v_end = int(len(dataset)*0.8), int(len(dataset)*0.9)
        train_ds = Subset(dataset, indices[:t_end])
        val_ds = Subset(dataset, indices[t_end:v_end])
        test_ds = Subset(dataset, indices[v_end:])
        
        args = dict(batch_size=self.config['batch_size'], num_workers=4, collate_fn=collator_interact)
        return DataLoader(train_ds, shuffle=True, **args), \
               DataLoader(val_ds, shuffle=False, **args), \
               DataLoader(test_ds, shuffle=False, **args)

    def get_model(self):
        ablation = self.config.get('ablation', False)
        use_gating = not self.config.get('no_gating', False)
        
        model = GeoMRL(
            mode=self.task, 
            atom_names=CompoundKit.atom_vocab_dict.keys(),
            atom_embed_dim=512, num_kernel=128, layer_num=6, num_heads=8,
            atom_FG_class=nfg()+1, hidden_size=2048, num_tasks=1,
            ablation=ablation, use_gating=use_gating
        ).to(self.device)

        if self.config['ckpt'] and self.task == 'ddi':
            print(f"[*] Loading weights: {self.config['ckpt']}")
            ckpt = torch.load(self.config['ckpt'], map_location=self.device)
            state_dict = ckpt['model'] if 'model' in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
        return model

    def train(self):
        new_params = [p for n, p in self.model.named_parameters() if "head" in n or "cross_" in n or "drug_b" in n]
        old_params = [p for n, p in self.model.named_parameters() if not any(x in n for x in ["head", "cross_", "drug_b"])]

        optimizer = torch.optim.AdamW([
            {'params': old_params, 'lr': self.config['lr']},
            {'params': new_params, 'lr': self.config['lr'] * 5} 
        ], weight_decay=1e-3)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=10)
        stopper = EarlyStopping(mode='higher', patience=20)
        
        for epoch in range(1, self.config['epochs'] + 1):
            self.model.train()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                logits = self.model(batch)['affinity'].squeeze()
                loss = self.criterion(logits, batch['label'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                pbar.set_postfix(Loss=f"{loss.item():.4f}")
            
            # 【核心修改 2】：每个 epoch 打印验证集的四大指标
            val_metrics = self.evaluate(self.val_loader)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch} | LR: {current_lr:.2e} | Val AUC: {val_metrics['AUC']:.4f} | Val AUPR: {val_metrics['AUPR']:.4f} | Val ACC: {val_metrics['ACC']:.4f} | Val F1: {val_metrics['F1']:.4f}")
            
            scheduler.step(val_metrics['AUC'])
            
            if val_metrics['AUC'] > self.best_auc:
                self.best_auc = val_metrics['AUC']
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model.pth"))
            
            if stopper.step(val_metrics['AUC'], self.model): break

        print("\n[*] Training Complete. Extracting features and running Final Test...")
        best_model_path = os.path.join(self.save_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            
        # 【核心修改 3】：训练结束，跑测试集，并醒目地打印最终的四大指标！
        test_metrics = self.evaluate(self.test_loader, save_tsne=True)
        print("\n" + "="*50)
        print(f"               FINAL TEST RESULTS                 ")
        print("="*50)
        print(f"Test AUC  (ROC-AUC)        : {test_metrics['AUC']:.4f}")
        print(f"Test AUPR (PR-AUC)         : {test_metrics['AUPR']:.4f}")
        print(f"Test ACC  (Accuracy)       : {test_metrics['ACC']:.4f}")
        print(f"Test F1   (F1-Score)       : {test_metrics['F1']:.4f}")
        print("="*50 + "\n")

    # 【核心修改 4】：evaluate 函数完整计算四大指标，并处理边界情况
    def evaluate(self, loader, save_tsne=False):
        self.model.eval()
        probs, labels, feats = [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = self.model(batch)
                
                # 必须用 Sigmoid 将 Logits 转化为 0~1 的概率
                p = torch.sigmoid(out['affinity'].squeeze()).cpu().numpy()
                probs.extend(p)
                labels.extend(batch['label'].cpu().numpy())
                
                if save_tsne and 'latent_feature' in out: 
                    feats.extend(out['latent_feature'].cpu().numpy())
        
        if save_tsne and len(feats) > 0:
            np.save(os.path.join(self.save_dir, f"{self.task}_tsne_feats.npy"), np.array(feats))
            np.save(os.path.join(self.save_dir, f"{self.task}_tsne_labels.npy"), np.array(labels))
            print(f"[*] t-SNE data saved to {self.save_dir}")

        probs = np.array(probs)
        labels = np.array(labels)
        preds = (probs >= 0.5).astype(int) # 大于等于0.5判为1，否则为0
        
        try:
            return {
                'AUC': roc_auc_score(labels, probs),
                'AUPR': average_precision_score(labels, probs),
                'ACC': accuracy_score(labels, preds),
                'F1': f1_score(labels, preds, zero_division=0)
            }
        except ValueError:
            # 防止极小 batch 时全是同一类别导致 roc_auc_score 报错
            return {'AUC': 0.0, 'AUPR': 0.0, 'ACC': 0.0, 'F1': 0.0}

# ==================== 4. Main ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['ddi', 'ppi'])
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--gpu', type=str, default='0')
    
    parser.add_argument('--ablation', action='store_true', help='Set to disable RBF (w/o RBF Bias)')
    parser.add_argument('--no_gating', action='store_true', help='Set to disable Geo-Gating (w/o Gating)')
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(42) 

    config = vars(args)
    config['model'] = {'atom_embed_dim': 512, 'hidden_size': 2048, 'layer_num': 6, 'num_heads': 8, 'num_kernel': 128}
    
    InteractTrainer(config).train()

if __name__ == "__main__":
    main()