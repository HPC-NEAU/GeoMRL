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

        x = x.long().clamp(min=0, max=511)
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

class PositionalEncoding(nn.Module):
    """蛋白质序列的位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

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
        x = x.unsqueeze(-1) - self.means.weight 
        x = -0.5 * (x / (self.stds.weight.abs() + 1e-5)) ** 2
        x = x.exp() + self.bias.weight
        return x

# ==============================================================================
# 2. 几何注意力 (Backbone)
# ==============================================================================
class GeometricAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1, use_geometry=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_geometry = use_geometry
        
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        
        if self.use_geometry:
            self.rbf_proj = nn.Linear(128, num_heads)
        
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
        residual = x
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.use_geometry and rbf_feat is not None:
            geo_bias = self.rbf_proj(rbf_feat).permute(0, 3, 1, 2)
            attn = attn + geo_bias
        
        if mask is not None:
            mask_expanded = mask.view(B, 1, 1, N).expand(-1, self.num_heads, N, -1)
            attn = attn.masked_fill(~mask_expanded, -1e4)
            
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.layer_norm(x + residual)
        x = self.ffn_norm(x + self.ffn(x))
        return x

# ==============================================================================
# 3. 强化版：几何门控交叉注意力 (DTI 专用 - 带有消融开关)
# ==============================================================================
class GeometricCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1, use_gating=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_gating = use_gating # 门控消融开关
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.rbf_proj = nn.Linear(128, num_heads)
        
        # 仅在开启门控时初始化门控网络
        if self.use_gating:
            self.geo_gate = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward(self, query, key, value, rbf_dist_feat=None):
        B, N_q, C = query.shape
        B, N_k, _ = key.shape
        residual = query
        
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if rbf_dist_feat is not None:
            geo_bias = self.rbf_proj(rbf_dist_feat).permute(0, 3, 1, 2)
            if self.use_gating:
                gate = self.geo_gate(rbf_dist_feat).permute(0, 3, 1, 2)
                attn = (attn + geo_bias) * gate
            else:
                attn = attn + geo_bias # 消融：仅几何偏置，不加门控
            
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.out_proj(x)
        x = self.layer_norm(x + residual)
        x = self.ffn_norm(x + self.ffn(x))
        return x

# ==============================================================================
# 4. GeoMRL 主模型
# ==============================================================================
class GeoMRL(nn.Module):
    def __init__(self, mode, atom_names, atom_embed_dim, num_kernel,
                 layer_num, num_heads, atom_FG_class: int, hidden_size, num_tasks,
                 ablation=False, use_gating=True): # 增加 use_gating 参数
        super().__init__()
        self.mode = mode
        self.use_geometry = not ablation 
        
        # 1. Embedding
        self.atom_feature = TensorAtomEmbedding(embed_dim=atom_embed_dim)
        self.res_feature = ResidueEmbedding(embed_dim=atom_embed_dim)
        self.type_embedding = nn.Embedding(2, atom_embed_dim)
        self.cls_embedding = nn.Embedding(1, atom_embed_dim)
        self.rbf = GaussianLayer(K=128)
        
        # 2. Backbone (Ligand)
        self.layers = nn.ModuleList([
            GeometricAttention(atom_embed_dim, num_heads, use_geometry=self.use_geometry) 
            for _ in range(layer_num)
        ])

        # 3. Heads 分支
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
            # DTI 专属增强模块
            self.prot_pos_enc = PositionalEncoding(atom_embed_dim)
            prot_layer = nn.TransformerEncoderLayer(
                d_model=atom_embed_dim, nhead=num_heads, dim_feedforward=atom_embed_dim*4, 
                dropout=0.2, batch_first=True
            )
            self.prot_encoder = nn.TransformerEncoder(prot_layer, num_layers=4)
            
            # 将 use_gating 传递给交叉注意力层
            self.cross_layers = nn.ModuleList([
                GeometricCrossAttention(hidden_size=atom_embed_dim, num_heads=num_heads, use_gating=use_gating)
                for _ in range(4)
            ])
            
            self.global_pool = nn.Sequential(
                nn.Linear(atom_embed_dim, 1),
                nn.Softmax(dim=1)
            )
            
            self.dti_head = nn.Sequential(
                nn.Linear(atom_embed_dim, atom_embed_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(atom_embed_dim, atom_embed_dim // 2),
                nn.GELU(),
                nn.Linear(atom_embed_dim // 2, 1)
            )

    def forward(self, batched_data: Dict[str, Tensor]):
        # 1. 基础配体/药物A Embedding & 3D RBF 准备
        # (这部分代码无论是 MPP, DTI 还是 DDI 都是公用的)
        x = self.atom_feature(batched_data)
        batch_size, n_atoms, _ = x.shape
        device = x.device
        
        raw_pos = batched_data.get('atom_pos', batched_data.get('pos', None))
        if raw_pos is None: raw_pos = torch.zeros((batch_size, n_atoms, 3), device=device)
        
        mask_atoms = (x.sum(dim=-1) != 0).float().unsqueeze(-1)
        cls_pos = (raw_pos * mask_atoms).sum(1, keepdim=True) / (mask_atoms.sum(1, keepdim=True) + 1e-6)
        
        cls_token = self.cls_embedding(torch.zeros(batch_size, 1, dtype=torch.long, device=device))
        x = torch.cat([cls_token, x], dim=1) 
        pos = torch.cat([cls_pos, raw_pos], dim=1)

        rbf_feat = None
        if self.use_geometry:
            dist_matrix = torch.norm(pos.unsqueeze(2) - pos.unsqueeze(1), dim=-1)
            rbf_feat = self.rbf(dist_matrix)

        attn_mask = (x.abs().sum(dim=-1) > 1e-9)

        # ============================================================
        # 模式 1: DTI (药物-靶标 交互)
        # ============================================================
        if self.mode == "dti" and 'protein_x' in batched_data:
            prot_x = self.res_feature(batched_data['protein_x'])
            prot_x = self.prot_pos_enc(prot_x)
            prot_x = self.prot_encoder(prot_x)
            
            ligand_feat = x
            for layer in self.layers:
                ligand_feat = layer(ligand_feat, rbf_feat, mask=attn_mask)
            
            prot_pos = batched_data.get('protein_pos', None)
            cross_rbf = None
            if self.use_geometry and prot_pos is not None:
                dist_cross = torch.norm(pos.unsqueeze(2) - prot_pos.unsqueeze(1), dim=-1)
                cross_rbf = self.rbf(dist_cross)
            
            curr_feat = ligand_feat
            for layer in self.cross_layers:
                curr_feat = layer(curr_feat, prot_x, prot_x, rbf_dist_feat=cross_rbf)
            
            weights = self.global_pool(curr_feat)
            final_feat = torch.sum(curr_feat * weights, dim=1) + curr_feat[:, 0, :]
            return {"affinity": self.dti_head(final_feat), "latent_feature": final_feat}

        # ============================================================
        # 模式 2: DDI (药物-药物 交互) —— 【新增修复点】
        # ============================================================
        elif self.mode == "ddi" and 'drug_b_tokens' in batched_data:
            # 药物 B 作为背景 Context
            drug_b_feat = self.drug_b_embedding(batched_data['drug_b_tokens'])
            drug_b_feat = self.drug_b_pos_enc(drug_b_feat)
            
            # 药物 A (主干) 提取几何特征
            ligand_a_feat = x
            for layer in self.layers:
                ligand_a_feat = layer(ligand_a_feat, rbf_feat, mask=attn_mask)
            
            # 交叉注意力交互
            curr_feat = ligand_a_feat
            for layer in self.ddi_cross_layers:
                # DDI 通常没有跨模态 RBF，传 None
                curr_feat = layer(curr_feat, drug_b_feat, drug_b_feat, rbf_dist_feat=None)
            
            weights = self.global_pool(curr_feat)
            final_feat = torch.sum(curr_feat * weights, dim=1) + curr_feat[:, 0, :]
            return {"affinity": self.ddi_head(final_feat), "latent_feature": final_feat}

        # ============================================================
        # 模式 3: PPI (蛋白-蛋白 交互) —— 【新增修复点】
        # ============================================================
        elif self.mode == "ppi" and 'protein_a_seq' in batched_data:
            # 蛋白 A 和 蛋白 B 均走 prot_encoder
            feat_a = self.res_feature(batched_data['protein_a_seq'])
            feat_a = self.prot_encoder(self.prot_pos_enc(feat_a))
            
            feat_b = self.res_feature(batched_data['protein_b_seq'])
            feat_b = self.prot_encoder(self.prot_pos_enc(feat_b))
            
            # 简单的全局池化后做交互
            rep_a = torch.mean(feat_a, dim=1)
            rep_b = torch.mean(feat_b, dim=1)
            
            # 拼接预测
            combined = torch.cat([rep_a, rep_b], dim=-1)
            # 假设你为 PPI 定义了一个特殊的 head，如果没有，我们复用 dti_head 的一部分逻辑
            # 这里为了不报错，统一指向一个能输出单个值的 MLP
            return {"affinity": self.dti_head(rep_a + rep_b), "latent_feature": rep_a + rep_b}
        # 4. MPP (Finetune) / Pretrain 逻辑
        for layer in self.layers:
            x = layer(x, rbf_feat, mask=attn_mask)

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
            # 同样提取用于预测的图级别特征用于画 t-SNE 聚类图
            graph_feat = x[:, 0, :]
            predictions["graph_feature"] = self.head_Graph(graph_feat)
            predictions["finger_feature"] = self.head_finger_keeping_atom(graph_feat)
            predictions["atom_fg"] = self.head_FG_atom(x[:, 1:, :])
            predictions["latent_feature"] = graph_feat # 这个输出对于 t-SNE 画图极其重要！
            
        return predictions# ==============================================================================
# 4. GeoMRL 主模型 (包含所有任务支持)
# ==============================================================================
class GeoMRL(nn.Module):
    def __init__(self, mode, atom_names, atom_embed_dim, num_kernel,
                 layer_num, num_heads, atom_FG_class: int, hidden_size, num_tasks,
                 ablation=False, use_gating=True):
        super().__init__()
        self.mode = mode
        self.use_geometry = not ablation 
        
        # 1. 基础 Embedding (配体 A 专用)
        self.atom_feature = TensorAtomEmbedding(embed_dim=atom_embed_dim)
        self.res_feature = ResidueEmbedding(embed_dim=atom_embed_dim)
        self.type_embedding = nn.Embedding(2, atom_embed_dim)
        self.cls_embedding = nn.Embedding(1, atom_embed_dim)
        
        self.rbf = GaussianLayer(K=128)
        
        # 2. 主干网络 Backbone (配体内部演化，所有任务共用)
        self.layers = nn.ModuleList([
            GeometricAttention(atom_embed_dim, num_heads, use_geometry=self.use_geometry) 
            for _ in range(layer_num)
        ])

        # 3. 任务头分支初始化
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

        elif self.mode == "finetune": # MPP 单分子属性预测
            self.head_Graph = AtomProjection(atom_embed_dim, atom_embed_dim // 2, num_tasks)
            self.head_finger_keeping_atom = AtomProjection(atom_embed_dim, atom_embed_dim * 2, 2048)
            self.head_FG_atom = AtomFGProjection(atom_embed_dim, atom_embed_dim // 2, atom_FG_class)
            
        elif self.mode == "dti": # DTI 药物靶标对接
            self.prot_pos_enc = PositionalEncoding(atom_embed_dim)
            prot_layer = nn.TransformerEncoderLayer(
                d_model=atom_embed_dim, nhead=num_heads, dim_feedforward=atom_embed_dim*4, 
                dropout=0.2, batch_first=True
            )
            self.prot_encoder = nn.TransformerEncoder(prot_layer, num_layers=4)
            
            self.cross_layers = nn.ModuleList([
                GeometricCrossAttention(hidden_size=atom_embed_dim, num_heads=num_heads, use_gating=use_gating)
                for _ in range(4)
            ])
            self.global_pool = nn.Sequential(nn.Linear(atom_embed_dim, 1), nn.Softmax(dim=1))
            self.dti_head = nn.Sequential(
                nn.Linear(atom_embed_dim, atom_embed_dim), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(atom_embed_dim, atom_embed_dim // 2), nn.GELU(),
                nn.Linear(atom_embed_dim // 2, 1)
            )
            
        elif self.mode == "ddi": # DDI 药物-药物相互作用
            # DDI 专属：药物 B 的轻量级序列编码器
            self.drug_b_embedding = nn.Embedding(200, atom_embed_dim, padding_idx=0)
            self.drug_b_pos_enc = PositionalEncoding(atom_embed_dim)
            
            # DDI 交叉交互：配体 A (Query) 查 药物 B (Key/Value)
            self.ddi_cross_layers = nn.ModuleList([
                GeometricCrossAttention(hidden_size=atom_embed_dim, num_heads=num_heads, use_gating=False) # 没坐标，所以不门控
                for _ in range(2)
            ])
            self.global_pool = nn.Sequential(nn.Linear(atom_embed_dim, 1), nn.Softmax(dim=1))
            self.ddi_head = nn.Sequential(
                nn.Linear(atom_embed_dim, atom_embed_dim // 2), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(atom_embed_dim // 2, 1)
            )

        elif self.mode == "ppi": # PPI 蛋白-蛋白相互作用
            self.prot_pos_enc = PositionalEncoding(atom_embed_dim)
            prot_layer = nn.TransformerEncoderLayer(d_model=atom_embed_dim, nhead=num_heads, batch_first=True)
            self.prot_encoder = nn.TransformerEncoder(prot_layer, num_layers=3)
            self.ppi_head = nn.Sequential(
                nn.Linear(atom_embed_dim * 2, atom_embed_dim), nn.GELU(),
                nn.Linear(atom_embed_dim, 1)
            )

    def forward(self, batched_data: Dict[str, Tensor]):
        # 1. 基础配体 A Embedding (除了 PPI，其他都需要走)
        if self.mode != "ppi":
            x = self.atom_feature(batched_data)
            batch_size, n_atoms, _ = x.shape
            device = x.device
            
            raw_pos = batched_data.get('atom_pos', batched_data.get('pos', None))
            if raw_pos is None: raw_pos = torch.zeros((batch_size, n_atoms, 3), device=device)
            
            mask_atoms = (x.sum(dim=-1) != 0).float().unsqueeze(-1)
            cls_pos = (raw_pos * mask_atoms).sum(1, keepdim=True) / (mask_atoms.sum(1, keepdim=True) + 1e-6)
            
            cls_token = self.cls_embedding(torch.zeros(batch_size, 1, dtype=torch.long, device=device))
            x = torch.cat([cls_token, x], dim=1) 
            pos = torch.cat([cls_pos, raw_pos], dim=1)

            rbf_feat = None
            if self.use_geometry:
                dist_matrix = torch.norm(pos.unsqueeze(2) - pos.unsqueeze(1), dim=-1)
                rbf_feat = self.rbf(dist_matrix)

            attn_mask = (x.abs().sum(dim=-1) > 1e-9)

        # ============================================================
        # 模式 1: DTI (药物-靶标 交互)
        # ============================================================
        if self.mode == "dti" and 'protein_x' in batched_data:
            prot_x = self.res_feature(batched_data['protein_x'])
            prot_x = self.prot_encoder(self.prot_pos_enc(prot_x))
            
            ligand_feat = x
            for layer in self.layers:
                ligand_feat = layer(ligand_feat, rbf_feat, mask=attn_mask)
            
            prot_pos = batched_data.get('protein_pos', None)
            cross_rbf = None
            if self.use_geometry and prot_pos is not None:
                dist_cross = torch.norm(pos.unsqueeze(2) - prot_pos.unsqueeze(1), dim=-1)
                cross_rbf = self.rbf(dist_cross)
            
            curr_feat = ligand_feat
            for layer in self.cross_layers:
                curr_feat = layer(curr_feat, prot_x, prot_x, rbf_dist_feat=cross_rbf)
            
            weights = self.global_pool(curr_feat)
            final_feat = torch.sum(curr_feat * weights, dim=1) + curr_feat[:, 0, :]
            return {"affinity": self.dti_head(final_feat), "latent_feature": final_feat}

        # ============================================================
        # 模式 2: DDI (药物-药物 交互)
        # ============================================================
        elif self.mode == "ddi" and 'drug_b_tokens' in batched_data:
            drug_b_feat = self.drug_b_pos_enc(self.drug_b_embedding(batched_data['drug_b_tokens']))
            
            ligand_a_feat = x
            for layer in self.layers:
                ligand_a_feat = layer(ligand_a_feat, rbf_feat, mask=attn_mask)
            
            curr_feat = ligand_a_feat
            for layer in self.ddi_cross_layers:
                curr_feat = layer(curr_feat, drug_b_feat, drug_b_feat, rbf_dist_feat=None)
            
            weights = self.global_pool(curr_feat)
            final_feat = torch.sum(curr_feat * weights, dim=1) + curr_feat[:, 0, :]
            return {"affinity": self.ddi_head(final_feat), "latent_feature": final_feat}

        # ============================================================
        # 模式 3: PPI (蛋白-蛋白 交互)
        # ============================================================
        elif self.mode == "ppi" and 'protein_a_seq' in batched_data:
            feat_a = self.prot_encoder(self.prot_pos_enc(self.res_feature(batched_data['protein_a_seq'])))
            feat_b = self.prot_encoder(self.prot_pos_enc(self.res_feature(batched_data['protein_b_seq'])))
            
            rep_a = torch.mean(feat_a, dim=1)
            rep_b = torch.mean(feat_b, dim=1)
            
            combined = torch.cat([rep_a, rep_b], dim=-1)
            return {"affinity": self.ppi_head(combined), "latent_feature": combined}

        # ============================================================
        # 模式 4: MPP (Finetune) / Pretrain
        # ============================================================
        if self.mode not in ["dti", "ddi", "ppi"]:
            for layer in self.layers:
                x = layer(x, rbf_feat, mask=attn_mask)

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
                    predictions['angle_pred'] = self.head_angle(ligand_x, batched_data['angles_atom_index'])

            elif self.mode == "finetune":
                graph_feat = x[:, 0, :]
                predictions["graph_feature"] = self.head_Graph(graph_feat)
                predictions["finger_feature"] = self.head_finger_keeping_atom(graph_feat)
                predictions["atom_fg"] = self.head_FG_atom(x[:, 1:, :])
                predictions["latent_feature"] = graph_feat
                
            return predictions