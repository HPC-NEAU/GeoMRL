import argparse
import sys
import os
import shutil
import pickle
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import warnings

# 路径修复
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from models.uniscage import UniScage
from _config import *
from naive_fg import all_possible_fg_nums
from data_process.function_group_constant import nfg
from utils.public_util import set_seed
from data_process.compound_tools import CompoundKit
from data_process.data_collator import collator_pretrain_pkl_bin
from utils.global_var_util import GlobalVar, set_dist_bar_two

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def write_record(path, message):
    with open(path, 'a') as f:
        f.write(f'{message}\n')

# [核心] Mmap Dataset: 零内存占用，极速读取
class MmapDataset(Dataset):
    def __init__(self, mmap_dir):
        self.mmap_dir = mmap_dir
        meta_path = os.path.join(mmap_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file not found at {meta_path}. Please run preprocess_mmap.py first.")
            
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)
        self.N = self.meta['N']
        
        # 只读模式加载 Memmap
        self.data = {}
        for key in self.meta['shapes']:
            shape = tuple(self.meta['shapes'][key])
            dtype_str = self.meta['dtypes'][key]
            
            # 简单类型映射
            if 'float' in dtype_str: np_dtype = np.float32
            elif 'bool' in dtype_str: np_dtype = np.bool_
            else: np_dtype = np.int32
            
            # 打开内存映射 (此时不占物理内存)
            self.data[key] = np.memmap(
                os.path.join(mmap_dir, f"{key}.npy"), 
                dtype=np_dtype, mode='r', shape=shape
            )

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        item = {}
        for key in self.data:
            # 读取数据 (此时发生内存拷贝，但仅限于这一个样本)
            val = np.array(self.data[key][idx]) # 强转 array 防止 copy-on-write 问题
            
            if key in ['atom_pos', 'morgan2048_fp']:
                item[key] = torch.from_numpy(val).float()
            elif key == 'atom_attention_mask':
                item[key] = torch.from_numpy(val).bool()
            else:
                item[key] = torch.from_numpy(val).long()
        return item

def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return local_rank, rank, world_size
    return 0, 0, 1

class PreTrainer(object):
    def __init__(self, config, local_rank, rank):
        self.config = config
        self.local_rank = local_rank
        self.rank = rank
        
        self.train_loader, self.test_loader = self.get_data_loaders()
        self.model = self._get_net()
        self.optim = self._get_optim()
        
        self.start_epoch = 1
        if self.rank == 0:
            if config['checkpoint']:
                self.load_ckpt(self.config['checkpoint'])
            else:
                run_name = f"pretrain_ddp_{datetime.now().strftime('%b%d_%H_%M')}"
                self.writer = SummaryWriter(f'../train_result/pretrain_result/{run_name}')
                self.txtfile = os.path.join(self.writer.log_dir, 'record.txt')
        
        self.batch_considered = 200 
        self.loss_init = torch.zeros(GlobalVar.loss_num, 200, device=local_rank)
        self.loss_last = torch.zeros(GlobalVar.loss_num, self.batch_considered // 10, device=local_rank)
        self.loss_last2 = torch.zeros(GlobalVar.loss_num, self.batch_considered // 10, device=local_rank)
        self.cur_loss_step = torch.zeros(1, dtype=torch.long, device=local_rank)
        self.optim_steps = 0

    def get_data_loaders(self):
        if self.rank == 0: print(f"Loading Mmap data from {self.config['root']}...")
        
        # [修改] 使用 MmapDataset
        dataset = MmapDataset(self.config['root'])
        
        if self.rank == 0: print(f'Total dataset size: {len(dataset)}')
        
        train_len = int(len(dataset) * 0.99)
        train_dataset = Subset(dataset, range(0, train_len))
        test_dataset = Subset(dataset, range(train_len, len(dataset)))
        
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        
        # [优化] num_workers 可以开到 6-8，且不会爆内存
        loader_args = dict(batch_size=self.config['batch_size'], num_workers=6, 
                           collate_fn=None, 
                           pin_memory=True, persistent_workers=True, prefetch_factor=2)
        
        return DataLoader(train_dataset, sampler=train_sampler, **loader_args), \
               DataLoader(test_dataset, sampler=test_sampler, **loader_args)

    def _get_net(self):
        real_fg_num = GlobalVar.fg_number 
        
        model = UniScage(
            mode='pretrain_bin',
            atom_names=CompoundKit.atom_vocab_dict.keys(),
            atom_embed_dim=self.config['model']['atom_embed_dim'],
            num_kernel=self.config['model'].get('num_kernel', 128),
            layer_num=self.config['model']['layer_num'],
            num_heads=self.config['model']['num_heads'],
            atom_FG_class=real_fg_num, 
            hidden_size=self.config['model']['hidden_size'],
            num_tasks=None
        ).to(self.local_rank)

        model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        return model

    def _get_optim(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config['optim']['init_lr'])

    def _safe_ce(self, pred, target, n_classes, ignore_index=-100):
        p_flat = pred.reshape(-1, n_classes)
        t_flat = target.reshape(-1).long()
        n = min(p_flat.shape[0], t_flat.shape[0])
        
        mask = (t_flat[:n] >= 0) & (t_flat[:n] < n_classes)
        t_flat_safe = torch.full_like(t_flat[:n], ignore_index)
        t_flat_safe[mask] = t_flat[:n][mask]
        
        return F.cross_entropy(p_flat[:n], t_flat_safe, ignore_index=ignore_index)

    def calc_mt_loss(self, loss_list):
        loss_list = torch.stack(loss_list).to(self.local_rank)
        # 简化版逻辑，防止除以0
        return sum(loss_list)

    def _step(self, model, batch):
        pred_dict = model(batch)
        loss_list = []
        
        # 1. Fingerprint
        if 'finger' in GlobalVar.pretrain_task:
            if 'morgan2048_fp' in batch:
                pred = pred_dict['atom_finger_feature']
                target = batch['morgan2048_fp'].float()
                if torch.isnan(target).any(): target = torch.nan_to_num(target, 0.0)
                loss_list.append(F.binary_cross_entropy_with_logits(pred, target))
            else:
                loss_list.append(torch.tensor(0.0, device=self.local_rank, requires_grad=True))

        # 2. FG
        if 'fg' in GlobalVar.pretrain_task:
            if 'function_group_index' in batch:
                pred = pred_dict['atom_fg']
                target = batch['function_group_index']
                if target.dim() == 3: target = target.argmax(dim=-1)
                loss_list.append(self._safe_ce(pred, target, GlobalVar.fg_number, ignore_index=-100))
            else:
                loss_list.append(torch.tensor(0.0, device=self.local_rank, requires_grad=True))

        # 3. SP
        if 'sp' in GlobalVar.pretrain_task:
            if 'spatial_pos' in batch:
                pred = pred_dict['spatial_pos_pred']
                target = batch['spatial_pos']
                loss_list.append(self._safe_ce(pred, target, 21, ignore_index=-100))
            else:
                loss_list.append(torch.tensor(0.0, device=self.local_rank, requires_grad=True))

        # 4. Angle
        if 'angle' in GlobalVar.pretrain_task:
            if 'bond_angles_bin' in batch:
                pred = pred_dict['angle_pred']
                target = batch['bond_angles_bin']
                loss_list.append(self._safe_ce(pred, target, 20, ignore_index=-5))
            else:
                loss_list.append(torch.tensor(0.0, device=self.local_rank, requires_grad=True))

        loss = sum(loss_list) if len(loss_list) > 0 else torch.tensor(0.0, device=self.local_rank, requires_grad=True)
        return loss, pred_dict, loss_list

    def train(self):
        self.model.train()
        accumulation_steps = 4 
        
        for i in range(1, self.config['epochs'] + 1):
            self.train_loader.sampler.set_epoch(i)
            loss_epoch = 0
            self.optim.zero_grad()
            
            iterator = tqdm(self.train_loader, desc=f"Epoch {i}", leave=False) if self.rank == 0 else self.train_loader
            
            for step, batch in enumerate(iterator):
                batch = {k: v.to(self.local_rank) for k, v in batch.items() if v is not None and not isinstance(v, list)}
                
                loss, _, loss_list = self._step(self.model, batch)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    self.optim.zero_grad()
                    continue
                
                loss = loss / accumulation_steps
                loss.backward()
                
                if (step + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optim.step()
                    self.optim.zero_grad()
                    self.optim_steps += 1
                
                if self.rank == 0:
                    loss_val = loss.item() * accumulation_steps
                    loss_epoch += loss_val
                    info = f"Loss: {loss_val:.2f}"
                    if len(loss_list) > 0: info += f" Fp: {loss_list[0].item():.2f}"
                    if len(loss_list) > 1: info += f" FG: {loss_list[1].item():.2f}"
                    if len(loss_list) > 2: info += f" SP: {loss_list[2].item():.2f}"
                    if len(loss_list) > 3: info += f" Ang: {loss_list[3].item():.2f}"
                    if isinstance(iterator, tqdm): iterator.set_description(info)

            if self.rank == 0:
                avg_loss = loss_epoch / len(self.train_loader)
                print(f"Epoch {i}: Avg Loss {avg_loss:.4f}")
                write_record(self.txtfile, f"Epoch {i}: Avg Loss {avg_loss:.4f}")
                if i % 10 == 0: self.save_ckpt(i, avg_loss)

    def save_ckpt(self, epoch, loss):
        path = os.path.join(self.writer.log_dir, 'checkpoint')
        os.makedirs(path, exist_ok=True)
        model_to_save = self.model.module
        torch.save({'model': model_to_save.state_dict()}, os.path.join(path, f'model_{epoch}.pth'))

    def load_ckpt(self, path):
        ckpt = torch.load(path, map_location={'cuda:0': f'cuda:{self.local_rank}'})
        self.model.module.load_state_dict(ckpt['model'], strict=False)

def main():
    local_rank, rank, world_size = setup_ddp()
    
    parser = argparse.ArgumentParser()
    # [修改] 默认路径改为 mmap 文件夹
    parser.add_argument('--dataroot', type=str, default="/root/SCAGE-master1/UniSCAGE/processed_data1/pretrain/mmap_data")
    parser.add_argument('--ckpt', default=None, type=str)
    args = parser.parse_args()

    GlobalVar.parallel_train = False 

    # [关键修复] 动态获取真实类别数
    real_fg_num = nfg() + 1
    GlobalVar.fg_number = real_fg_num
    if rank == 0: print(f"FG Classes: {real_fg_num}")

    config = {
        'root': args.dataroot,
        'batch_size': 32, 
        'epochs': 100,
        'checkpoint': args.ckpt,
        'optim': {'init_lr': 1e-4}, 
        'dataloader_num_workers': 6,
        'model': {'atom_embed_dim': 512, 'hidden_size': 2048, 
                  'layer_num': 6, 'num_heads': 8, 'num_kernel': 128}
    }

    GlobalVar.pretrain_task = ['finger', 'fg', 'sp', 'angle']
    GlobalVar.loss_num = 4
    set_dist_bar_two(20, 50) 
    # GlobalVar.balanced_atom_fg_loss = False
    GlobalVar.use_calc_mt_loss = False

    if rank == 0: print("Start DDP Pretraining (Mmap Mode)...")
    trainer = PreTrainer(config, local_rank, rank)
    trainer.train()
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()