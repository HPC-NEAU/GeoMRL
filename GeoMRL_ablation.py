import torch
import torch.nn as nn
from torch import Tensor  
import torch.nn.functional as F
import math
from typing import Dict

# 引入层
from models.layers.projection import (AtomProjection, AtomFGProjection,
                                      AtomPairProjection, AtomAngleProjection,
                                      AtomPairProjectionBin, AtomAngleProjectionBin)
from utils.global_var_util import GlobalVar

# ==============================================================================
# 1. 基础组件
# ==============================================================================
class TensorAtomEmbedding(nn.Module):
    def __init__(self, num_features=7, embed_dim=512):
        super().__init__()
        # 增大词表到 512，防止越界
        self.embed_layers = nn.ModuleList([
            nn.Embedding(512, embed_dim, padding_idx=0) for _ in range(num_features)
        ])
    
    def forward(self, input_data):
        if isinstance(input_data, dict):
            if 'atom_feature' in input_data:
                x = input_data['atom_feature']
            elif 'atomic_num' in input_data:
                x = input_data['atomic_num']
                if x.dim() == 2: x = x.unsqueeze(-1)
            else:
                device = list(self.parameters())[0].device
                return torch.zeros((1, 1, self.embed_layers[0].embedding_dim), device=device)
        else:
            x = input_data

        x = x.long()
        x = x.clamp(min=0, max=511)

        out = 0
        valid = min(x.shape[-1], len(self.embed_layers))
        for i in range(valid):
            out += self.embed_layers[i](x[:, :, i])
            
        return out

class ResidueEmbedding(nn.Module):
    def __init__(self, vocab_size=25, embed_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    def forward(self, x):
        return self.embedding(x)

# RBF 距离编码
class GaussianLayer(nn.Module):
    def __init__(self, K=128):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)

        nn.init.uniform_(self.means.weight, 0, 30)
        nn.init.uniform_(self.stds.weight, 0, 10)
        self.bias = nn.Embedding(1, K)
        nn.init.constant_(self.bias.weight, 0)
        
    def forward(self, x):
        # x: [B, N, N]
        x = x.unsqueeze(-1) - self.means.weight # [B, N, N, K]
        x = -0.5 * (x / (self.stds.weight.abs() + 1e-5)) ** 2
        x = x.exp() + self.bias.weight
        return x

# ==============================================================================
# 2. 几何注意力层 (支持消融)
# ==============================================================================
class GeometricAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.5, use_geometry=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_geometry = use_geometry # [消融控制]
        
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        
        # 只有在启用几何信息时才初始化 RBF 投影层
        if self.use_geometry:
            self.rbf_proj = nn.Sequential(
                nn.Linear(128, num_heads),
            )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, rbf_feat=None, mask=None):
        B, N, C = x.shape
        
        # 1. Self Attention
        residual = x
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算 content score
        attn = (q @ k.transpose(-2, -1)) * self.scale # [B, Heads, N, N]
        
        # [核心消融点]
        # 如果 use_geometry=True 且 rbf_feat 存在，则注入几何 Bias
        # 如果 use_geometry=False (消融)，则不执行这一步，退化为普通 Self-Attention
        if self.use_geometry and rbf_feat is not None:
            geo_bias = self.rbf_proj(rbf_feat).permute(0, 3, 1, 2)
            attn = attn + geo_bias
        
        # Masking
        if mask is not None:
            mask_expanded = mask.view(B, 1, 1, N).expand(-1, self.num_heads, N, -1)
            attn = attn.masked_fill(~mask_expanded, -1e4)
            
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.layer_norm(x + residual)
        
        # 2. FFN
        x = self.ffn_norm(x + self.ffn(x))
        return x

# ==============================================================================
# 3. UniScage 主模型
# ==============================================================================
class UniScage(nn.Module):
    def __init__(self, mode, atom_names, atom_embed_dim, num_kernel,
                 layer_num, num_heads, atom_FG_class: int, hidden_size, num_tasks,
                 ablation=True): # [新增参数] ablation
        super().__init__()
        self.mode = mode
        
        # ablation=True 表示去除几何信息，即 use_geometry=False
        self.use_geometry = not ablation 
        if not self.use_geometry:
            print(f"!!! [ABLATION MODE] Geometric features are DISABLED !!!")
        
        # Embedding
        self.atom_feature = TensorAtomEmbedding(embed_dim=atom_embed_dim)
        self.res_feature = ResidueEmbedding(embed_dim=atom_embed_dim)
        self.type_embedding = nn.Embedding(2, atom_embed_dim)
        self.cls_embedding = nn.Embedding(1, atom_embed_dim)
        
        # 几何特征提取器 (RBF)
        # 如果消融，这层其实可以不初始化，但为了代码兼容性保留，Forward时不调用即可
        self.rbf = GaussianLayer(K=128)
        
        # Encoder Layers (传入控制参数)
        self.layers = nn.ModuleList([
            GeometricAttention(atom_embed_dim, num_heads, use_geometry=self.use_geometry) 
            for _ in range(layer_num)
        ])

        # Heads
        if self.mode in ["pretrain", "pretrain_bin"]:
            if 'fg' in GlobalVar.pretrain_task:
                self.head_FG_atom = AtomFGProjection(atom_embed_dim, atom_embed_dim // 2, atom_FG_class)
            if 'finger' in GlobalVar.pretrain_task:
                self.head_finger_keeping_atom = AtomProjection(atom_embed_dim, atom_embed_dim * 2, 2048)
            if 'sp' in GlobalVar.pretrain_task:
                HeadClass = AtomPairProjectionBin if self.mode == "pretrain_bin" else AtomPairProjection
                self.head_pair_distances = HeadClass(atom_embed_dim, atom_embed_dim // 8, 21)
            if 'angle' in GlobalVar.pretrain_task:
                HeadClass = AtomAngleProjectionBin if self.mode == "pretrain_bin" else AtomAngleProjection
                self.head_angle = HeadClass(atom_embed_dim, atom_embed_dim // 2, 1 if self.mode != "pretrain_bin" else 20)

        elif self.mode == "finetune":
            self.head_Graph = AtomProjection(atom_embed_dim, atom_embed_dim // 2, num_tasks)
            self.head_finger_keeping_atom = AtomProjection(atom_embed_dim, atom_embed_dim * 2, 2048)
            self.head_FG_atom = AtomFGProjection(atom_embed_dim, atom_embed_dim // 2, atom_FG_class)
            
        elif self.mode == "dti":
            # [修复] 确保输入维度是 atom_embed_dim
            self.dti_head = nn.Sequential(
                nn.Linear(atom_embed_dim, atom_embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(atom_embed_dim // 2, 1)
            )

    def forward(self, batched_data: Dict[str, Tensor]):
        # 1. Embedding
        x = self.atom_feature(batched_data)
        batch_size, n_atoms, _ = x.shape
        device = x.device
        
        # 2. 坐标处理 (构建几何特征)
        raw_pos = batched_data.get('atom_pos', batched_data.get('pos', None))
        if raw_pos is None: raw_pos = torch.zeros((batch_size, n_atoms, 3), device=device)
        
        mask_atoms = (x.sum(dim=-1) != 0).float().unsqueeze(-1)
        cls_pos = (raw_pos * mask_atoms).sum(1, keepdim=True) / (mask_atoms.sum(1, keepdim=True) + 1e-6)
        
        cls_token = self.cls_embedding(torch.zeros(batch_size, 1, dtype=torch.long, device=device))
        x = torch.cat([cls_token, x], dim=1) 
        pos = torch.cat([cls_pos, raw_pos], dim=1)
        
        if 'protein_x' in batched_data and self.mode == "dti":
            prot_emb = self.res_feature(batched_data['protein_x'])
            prot_pos = batched_data.get('protein_pos', torch.zeros((batch_size, prot_emb.shape[1], 3), device=device))
            
            x_lig = x + self.type_embedding(torch.zeros(1, device=device, dtype=torch.long))
            x_prot = prot_emb + self.type_embedding(torch.ones(1, device=device, dtype=torch.long))
            
            x = torch.cat([x_lig, x_prot], dim=1)
            pos = torch.cat([pos, prot_pos], dim=1)

        # 3. [消融控制] 计算 3D 距离特征
        rbf_feat = None
        if self.use_geometry:
            # 只有在非消融模式下才计算距离矩阵，节省显存和计算量
            delta = pos.unsqueeze(2) - pos.unsqueeze(1)
            dist_matrix = delta.norm(dim=-1)
            rbf_feat = self.rbf(dist_matrix) # [B, N, N, 128]

        # 4. Mask 准备
        attn_mask = (x.abs().sum(dim=-1) > 1e-9)

        # 5. Transformer Forward
        for layer in self.layers:
            # 传入 rbf_feat (如果是消融模式，这里是 None，Attention 层会自动忽略)
            x = layer(x, rbf_feat, mask=attn_mask)

        # 6. Output
        predictions = {}
        if self.mode in ["pretrain", "pretrain_bin"]:
            ligand_x = x[:, 1:n_atoms+1, :]
            cls_x = x[:, 0, :]
            
            if 'fg' in GlobalVar.pretrain_task:
                predictions['atom_fg'] = self.head_FG_atom(ligand_x) 
            if 'finger' in GlobalVar.pretrain_task:
                predictions['atom_finger_feature'] = self.head_finger_keeping_atom(cls_x)
            if 'sp' in GlobalVar.pretrain_task:
                spatial, dist_pred = self.head_pair_distances(ligand_x)
                predictions['spatial_pos_pred'] = spatial
                predictions['pair_distences_pred'] = dist_pred
            if 'angle' in GlobalVar.pretrain_task:
                angle_pred = self.head_angle(ligand_x, batched_data['angles_atom_index'])
                predictions['angle_pred'] = angle_pred

        elif self.mode == "finetune":
            predictions["graph_feature"] = self.head_Graph(x[:, 0, :])
            predictions["finger_feature"] = self.head_finger_keeping_atom(x[:, 0, :])
            predictions["atom_fg"] = self.head_FG_atom(x[:, 1:, :])
            
        elif self.mode == "dti":
            predictions["affinity"] = self.dti_head(x[:, 0, :])

        return predictions