import torch
import torch.nn as nn
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

class AtomEmbedding(nn.Module):
    """
    SCAGE 风格的原子嵌入：融合多种化学特征。
    不仅仅是原子序数，还包括度、电荷、杂化等。
    """
    def __init__(self, embed_dim):
        super().__init__()
        # 定义每个特征的词表大小 (根据 RDKit 的范围设定)
        self.embed_dim = embed_dim
        
        # 1. 原子序数 (0-118)
        self.emb_atomic_num = nn.Embedding(128, embed_dim)
        # 2. 连接度 (0-10)
        self.emb_degree = nn.Embedding(16, embed_dim)
        # 3. 电荷 (-5 到 +5 -> 移位 0-10)
        self.emb_charge = nn.Embedding(16, embed_dim)
        # 4. 手性 (0-4)
        self.emb_chirality = nn.Embedding(8, embed_dim)
        # 5. 杂化 (0-8)
        self.emb_hybridization = nn.Embedding(8, embed_dim)
        # 6. 芳香性 (0-1)
        self.emb_aromaticity = nn.Embedding(2, embed_dim)
        # 7. 连接氢原子数 (0-8)
        self.emb_num_hs = nn.Embedding(8, embed_dim)

    def forward(self, x):
        """
        x: [N, 7]  (包含7种特征的整数索引)
        """
        # 将所有特征的 Embedding 相加 (Graphormer/SCAGE 常用做法)
        h = self.emb_atomic_num(x[:, 0]) + \
            self.emb_degree(x[:, 1]) + \
            self.emb_charge(x[:, 2]) + \
            self.emb_chirality(x[:, 3]) + \
            self.emb_hybridization(x[:, 4]) + \
            self.emb_aromaticity(x[:, 5]) + \
            self.emb_num_hs(x[:, 6])
        return h

class ResidueEmbedding(nn.Module):
    """
    蛋白质残基嵌入。
    """
    def __init__(self, embed_dim):
        super().__init__()
        # 20种标准氨基酸 + 未知 + Pad
        self.emb_aa = nn.Embedding(30, embed_dim)
        
    def forward(self, x):
        """
        x: [M, ] (氨基酸索引)
        """
        # 如果输入是 [M, 7] (为了和配体对齐形状)，我们只取第一列
        if x.dim() > 1:
            x = x[:, 0]
        return self.emb_aa(x)
class SE3AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 特征变换
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
        # SE(3) 坐标更新权重网络
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh() # 限制移动
        )
        
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()

    def forward(self, h, pos, edge_index, mcl_mask=None):
        row, col = edge_index
        
        # 1. 投影 Q, K, V
        Q = self.W_q(h).view(-1, self.num_heads, self.head_dim)
        K = self.W_k(h).view(-1, self.num_heads, self.head_dim)
        V = self.W_v(h).view(-1, self.num_heads, self.head_dim)
        
        # 2. 计算几何距离 bias
        diff_vec = pos[row] - pos[col]
        dist_sq = (diff_vec ** 2).sum(dim=-1, keepdim=True) # [E, 1]
        
        # 3. Attention Score
        # score = (Q*K) / sqrt(d) - DistanceBias
        score = (Q[row] * K[col]).sum(dim=-1) / (self.head_dim ** 0.5)
        
        # 几何 Bias: 距离越远，注意力越弱
        score = score - torch.clamp(dist_sq, max=100.0) * 0.05
        
        # 4. SCAGE MCL Mask (如果提供)
        if mcl_mask is not None:
            # mcl_mask: [E], 0 表示被 mask 掉
            score = score + (1.0 - mcl_mask.unsqueeze(-1)) * -1e9
            
        # 5. Softmax & Aggregate
        alpha = softmax(score, row) # [E, Heads]
        
        # 特征聚合
        v_out = (V[col] * alpha.unsqueeze(-1)).view(-1, self.hidden_dim)
        h_agg = scatter_add(v_out, row, dim=0, dim_size=h.size(0))
        
        # 6. 坐标更新 (SE(3) Equivariant Update)
        # 坐标变化量 = sum_j (x_i - x_j) * weight_ij
        # weight_ij 由特征决定
        coord_w = self.coord_mlp(v_out) # [E, 1]
        trans_vec = diff_vec * coord_w
        pos_update = scatter_add(trans_vec, row, dim=0, dim_size=h.size(0))
        
        # 残差连接
        h_new = self.norm(h + self.o_proj(h_agg))
        pos_new = pos + pos_update * 0.1 # 步长缩放
        
        return h_new, pos_new