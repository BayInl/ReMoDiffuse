import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.stylization_block import StylizationBlock
from ..builder import ATTENTIONS


@ATTENTIONS.register_module()
class EfficientSelfAttention(nn.Module):

    def __init__(self, latent_dim, num_heads, dropout, time_embed_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(latent_dim)
        self.qkv = nn.Linear(latent_dim, latent_dim * 3)
        self.dropout = nn.Dropout(dropout)
        self.time_embed_dim = time_embed_dim
        if time_embed_dim is not None:
            self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, src_mask, emb=None, **kwargs):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_heads
        # B, T, D
        query = self.query(self.norm(x))
        # B, T, D
        key = (self.key(self.norm(x)) + (1 - src_mask) * -1000000)
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, T, H, -1), dim=1)
        # B, T, H, HD
        value = (self.value(self.norm(x)) * src_mask).view(B, T, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        if self.time_embed_dim is None:
            y = x + y
        else:
            y = x + self.proj_out(y, emb)
        return y


@ATTENTIONS.register_module()
class EfficientCrossAttention(nn.Module):

    def __init__(self, latent_dim, text_latent_dim, num_heads, dropout, time_embed_dim):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, src_mask, emb=None, **kwargs):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_heads

        seq_lengths = src_mask.sum(dim=1).long()  # B,
        max_seq_len = seq_lengths.max().item()  # 批次中最长的有效序列长度

        # 如果所有序列都比最大长度T短，我们可以只处理到max_seq_len
        if max_seq_len < T:
            # 只对有效部分进行计算
            effective_x = x[:, :max_seq_len, :]
            effective_mask = src_mask[:, :max_seq_len]

            # 对有效部分进行标准化和投影
            x_norm = self.norm(effective_x)
            qkv = self.qkv(x_norm)
            query, key, value = qkv.chunk(3, dim=-1)

            # 应用掩码到key (只在有效长度内)
            key = (key + (1 - effective_mask) * -1000000)

            # 重塑并应用softmax
            query = F.softmax(query.view(B, max_seq_len, H, -1), dim=-1)
            key = F.softmax(key.view(B, max_seq_len, H, -1), dim=1)

            # 应用掩码到value
            value = (value * effective_mask).view(B, max_seq_len, H, -1)

            # 计算注意力
            attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
            y = torch.einsum('bnhd,bhdl->bnhl', query,
                             attention).reshape(B, max_seq_len, D)

            # 创建完整大小的输出张量
            full_y = torch.zeros_like(x)
            full_y[:, :max_seq_len, :] = y

            # 应用残差连接和可能的投影
            if self.time_embed_dim is None:
                return x + full_y
            else:
                return x + self.proj_out(full_y, emb)

        else:
            x_norm = self.norm(x)
            # B, T, 3*D
            qkv = self.qkv(x_norm)
            # Split into query, key, value
            query, key, value = qkv.chunk(3, dim=-1)
            # B, T, D
            key = (key + (1 - src_mask) * -1000000)
            query = F.softmax(query.view(B, T, H, -1), dim=-1)
            key = F.softmax(key.view(B, T, H, -1), dim=1)
            # B, T, H, HD
            value = (value * src_mask).view(B, T, H, -1)
            # B, H, HD, HD
            attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
            y = torch.einsum('bnhd,bhdl->bnhl', query,
                             attention).reshape(B, T, D)
            if self.time_embed_dim is None:
                y = x + y
            else:
                y = x + self.proj_out(y, emb)
            return y
