import argparse
import sys
import os
import shutil
import pickle
from datetime import datetime
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

def write_record(path, message):
    with open(path, 'a') as f:
        f.write(f'{message}\n')

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.train_loader, self.val_loader, self.test_loader = self.get_data_loaders()
        self.net = self._get_net()
        self.criterion = self._get_loss_fn() # 放到 device 在 train 里做
        self.optim = self._get_optim()
        
        self.start_epoch = 1
        self.best_metric = -np.inf if config['task'] == 'classification' else np.inf
        self.best_test_metric = 0
        
        # 独立的日志目录，防止覆盖
        run_name = f"{config['task_name']}_seed{config['seed']}_{datetime.now().strftime('%m%d_%H%M')}"
        self.writer = SummaryWriter(f'../train_result/finetune_result1/{run_name}')
        self.txtfile = os.path.join(self.writer.log_dir, 'record.txt')

    def get_data_loaders(self):
        dataset = FinetuneDataset_pkl(root=self.config['userconfig']['mpp']['dataset_dir'], 
                                      task_name=self.config['task_name'])
        # 关键：根据当前 seed 创建 splitter
        splitter = create_splitter(self.config['split_type'], self.config['seed'])
        
        train_dataset, val_dataset, test_dataset = splitter.split(dataset, self.config['task_name'])
        
        print(f"Data Split (Seed {self.config['seed']}): Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

        loader_args = dict(batch_size=self.config['batch_size'], num_workers=4, 
                           collate_fn=collator_finetune_pkl)
        
        return DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_args), \
               DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_args), \
               DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_args)

    def _get_net(self):
        model = GeoMRL(
            mode='finetune',
            atom_names=CompoundKit.atom_vocab_dict.keys(),
            atom_embed_dim=self.config['model']['atom_embed_dim'],
            num_kernel=self.config['model'].get('num_kernel', 128),
            layer_num=self.config['model']['layer_num'],
            num_heads=self.config['model']['num_heads'],
            atom_FG_class=nfg() + 1,
            hidden_size=self.config['model']['hidden_size'],
            num_tasks=self.config['num_tasks'],
            ablation=False # 如果你要跑消融，这里改为 True
        ).cuda()

        ckpt_path = self.config.get('checkpoint')
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"Loading Pretrained Backbone from: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location='cuda')
            state_dict = checkpoint if 'model' not in checkpoint else checkpoint['model']
            
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() 
                               if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
        return model

    def _get_loss_fn(self):
        if self.config['task'] == 'classification':
            return nn.BCEWithLogitsLoss(reduction='none')
        return nn.MSELoss()

    def _get_optim(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.config['optim']['init_lr'], 
                                weight_decay=self.config['optim']['weight_decay'])

    def _step(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor): batch[k] = v.cuda()

        pred_dict = self.net(batch)
        pred = pred_dict['graph_feature']
        label = batch['label'].float()
        
        if self.config['task'] == 'classification':
            is_valid = (label != -1)
            loss = self.criterion(pred.cuda(), label)
            loss = torch.where(is_valid, loss, torch.zeros_like(loss))
            loss = loss.sum() / (is_valid.sum() + 1e-6)
        else:
            loss = self.criterion(pred.cuda(), label)
            
        return loss, pred

    def _evaluate(self, loader):
        self.net.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor): batch[k] = v.cuda()
                _, pred = self._step(batch)
                preds.append(pred)
                targets.append(batch['label'])
        
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        
        if self.config['task'] == 'classification':
            return compute_cls_metric_tensor(targets, preds)
        else:
            mae, rmse = compute_reg_metric(targets, preds)
            return rmse

    def train(self):
        stopper = EarlyStopping(mode='lower' if self.config['task'] == 'regression' else 'higher', 
                                patience=self.config['patience'])
        
        print(f"Start Training Seed {self.config['seed']}...")
        
        for epoch in range(1, self.config['epochs'] + 1):
            self.net.train()
            loss_epoch = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
            for batch in pbar:
                self.optim.zero_grad()
                loss, _ = self._step(batch)
                loss.backward()
                self.optim.step()
                loss_epoch += loss.item()
                pbar.set_description(f"Loss: {loss.item():.4f}")

            # Validation
            val_metric = self._evaluate(self.val_loader)
            test_metric = self._evaluate(self.test_loader)
            
            # Record Best
            is_best = False
            if self.config['task'] == 'classification':
                if val_metric > self.best_metric:
                    self.best_metric = val_metric
                    self.best_test_metric = test_metric
                    is_best = True
            else:
                if val_metric < self.best_metric:
                    self.best_metric = val_metric
                    self.best_test_metric = test_metric
                    is_best = True
            
            msg = f"Epoch {epoch}: Val {val_metric:.4f} | Test {test_metric:.4f} | Best Val {self.best_metric:.4f}"
            if is_best: msg += " [BEST]"
            print(msg)
            write_record(self.txtfile, msg)
            
            if stopper.step(val_metric, self.net):
                print("Early Stopping!")
                break

        print(f"Seed {self.config['seed']} Finished. Best Test Result: {self.best_test_metric:.4f}")
        return self.best_test_metric

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='bbbp')
    parser.add_argument('--ckpt', type=str, default=None) 
    parser.add_argument('--seed', type=int, default=42, help="Starting seed")
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--config', type=str, default='./config/config_finetune.yaml')
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    GlobalVar.parallel_train = False
    GlobalVar.dist_bar = [20, 50] 

    if os.path.exists(args.config):
        print(f"Loading config from {args.config}")
        with open(args.config, 'r') as f:
            base_config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        print("Warning: Config file not found! Using hardcoded defaults.")
        base_config = {
            'batch_size': 32, 'epochs': 100, 'patience': 20, 'split_type': 'scaffold',
            'model': {'atom_embed_dim': 512, 'num_kernel': 128, 'layer_num': 6, 'num_heads': 8, 'hidden_size': 2048}, 
            'optim': {'init_lr': 0.0001, 'weight_decay': 1e-4},
            'userconfig': {'mpp': {'dataset_dir': './processed_data1/mpp/pkl', 'split_dir': './processed_data1/mpp/split/'}}
        }
    
    # 2. 命令行参数覆盖
    base_config['task_name'] = args.task
    if args.ckpt is not None:
        base_config['checkpoint'] = args.ckpt
    
    # [核心修复] 必须先定义 root，再调用 get_downstream_task_names
    base_config['root'] = './processed_data1/mpp/pkl' 
    
    # 3. 路径初始化
    base_config = get_downstream_task_names(base_config)
    user = 'mpp'
    base_config = config_current_user(user, base_config)
    base_config = config_dataset_form('pkl', base_config)

    # ==========================
    # 5 Seed Loop
    # ==========================
    seeds = [args.seed + i for i in range(5)]
    results = []

    print(f"\n{'='*60}")
    print(f"Starting 5-Seed Run for {args.task.upper()}")
    print(f"Seeds: {seeds}")
    print(f"{'='*60}\n")

    for seed in seeds:
        print(f"\n>>> Running Seed {seed}...")
        set_seed(seed)
        
        current_config = deepcopy(base_config)
        current_config['seed'] = seed
        
        trainer = Trainer(current_config)
        test_metric = trainer.train()
        results.append(test_metric)

    # ==========================
    # Final Statistics
    # ==========================
    mean_res = np.mean(results)
    std_res = np.std(results)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS SUMMARY ({args.task.upper()})")
    print(f"{'='*60}")
    for i, res in enumerate(results):
        print(f"Seed {seeds[i]}: {res:.4f}")
    print(f"{'-'*60}")
    print(f"Average: {mean_res:.4f} ± {std_res:.4f}")
    print(f"{'='*60}\n")

    with open('../train_result/finetune_result1/5seed_summary.txt', 'a') as f:
        f.write(f"Task: {args.task} | Seeds: {seeds} | Result: {mean_res:.4f} +/- {std_res:.4f}\n")

if __name__ == '__main__':
    main()