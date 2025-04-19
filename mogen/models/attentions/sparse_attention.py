import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LongformerConfig, LongformerSelfAttention
from ..utils.stylization_block import StylizationBlock
from ..builder import ATTENTIONS


@ATTENTIONS.register_module()
class SparseSelfAttention(nn.Module):
    def __init__(self, latent_dim, num_heads, dropout=0.0, time_embed_dim=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.time_embed_dim = time_embed_dim

        # Initialize Longformer attention
        config = LongformerConfig(
            hidden_size=latent_dim,
            num_attention_heads=num_heads,
            # Local attention window size, make sure this is appropriate
            attention_window=[14],
            attention_probs_dropout_prob=dropout,
        )
        # layer_id=0 is a dummy value, not used if config.attention_window is a list of size 1
        self.attention = LongformerSelfAttention(config, layer_id=0)

        self.norm = nn.LayerNorm(latent_dim)

        if time_embed_dim is not None:
            self.proj_out = StylizationBlock(
                latent_dim, time_embed_dim, dropout)

    def forward(self, x, src_mask, keyframe_mask, emb=None, **kwargs):
        """
        x: [B, T, D] - input sequence
        src_mask: [B, T, 1] - binary mask where 1 indicates valid sequence length positions.
        keyframe_mask: [B, T] - binary mask where 1 indicates keyframes within the valid length.
        emb: [B, E] - Optional time embedding.
        """
        B, T, D = x.shape
        H = self.num_heads
        device = x.device

        padding_needed = (14 - (T % 14)) % 14
        if padding_needed > 0:
            x = F.pad(x, (0, 0, 0, padding_needed), value=0)
            src_mask = F.pad(src_mask, (0, 0, 0, padding_needed), value=0)
            keyframe_mask = F.pad(keyframe_mask, (0, padding_needed), value=0)

        # [B, T], True for valid positions
        src_mask_squeezed = src_mask.squeeze(-1).bool()
        seq_lengths = src_mask_squeezed.sum(
            dim=1).long()  # [B], length of each sequence
        max_seq_len = seq_lengths.max().item()

        padding_needed = (14 - (max_seq_len % 14)) % 14
        if padding_needed > 0:
            max_seq_len = max_seq_len + padding_needed

        # Optimization: Process only up to max_seq_len
        x_eff = x[:, :max_seq_len, :]
        if x.size(1) < max_seq_len:
            x_eff = torch.cat([x_eff, torch.zeros(
                (B, max_seq_len - x_eff.size(1), D), device=x.device)], dim=1)

        # Ensure keyframe_mask_eff has the correct length with padding
        if keyframe_mask.size(1) < max_seq_len:
            # Create a new tensor with the required length, filled with False
            keyframe_mask_eff = torch.zeros(
                (B, max_seq_len), dtype=torch.bool, device=x.device)
            # Copy the original values
            keyframe_mask_eff[:, :keyframe_mask.size(1)] = keyframe_mask
        else:
            # [B, max_seq_len]
            keyframe_mask_eff = keyframe_mask[:, :max_seq_len]

        current_T = max_seq_len

        # Normalize effective input
        x_norm = self.norm(x_eff)  # [B, current_T, D]

        # --- Generate Longformer Masks (Size [B, current_T]) ---
        mask_range = torch.arange(current_T, device=device)[
            None, :]  # [1, current_T]
        # valid_token_mask is True for actual tokens within seq_lengths
        valid_token_mask = mask_range < seq_lengths[:, None]  # [B, current_T]

        longformer_attention_values = (
            valid_token_mask).long()  # 未掩码的令牌为 1 ，掩码的令牌为0
        # [B, current_T]
        longformer_attention_mask = longformer_attention_values

        # is_index_masked: True for padding positions (masked AFTER softmax)
        is_index_masked = ~valid_token_mask  # [B, current_T]

        # is_index_global_attn: True for keyframes within the valid length
        is_index_global_attn = keyframe_mask_eff.bool(
        ) & valid_token_mask  # [B, current_T]
        # --- End Mask Generation ---

        # Apply Longformer attention
        # Output is a tuple, first element is the hidden states
        attention_output_tuple = self.attention(
            x_norm,  # [B, current_T, D]
            attention_mask=longformer_attention_mask,
            # [B, current_T], True for padding
            is_index_masked=is_index_masked,
            # [B, current_T], True for global
            is_index_global_attn=is_index_global_attn,
            is_global_attn=True,
        )
        attention_output = attention_output_tuple[0]  # [B, current_T, D]

        # Project output (only the effective part)
        if self.time_embed_dim is not None:
            proj_out_eff = self.proj_out(
                attention_output, emb)  # [B, current_T, D]
        else:
            # If no time embed, assume no projection before residual
            proj_out_eff = attention_output

        # Place projected effective output into full tensor
        if max_seq_len < T:
            final_proj_out = torch.zeros_like(x)  # [B, T, D]
            final_proj_out[:, :max_seq_len, :] = proj_out_eff
        else:
            final_proj_out = proj_out_eff  # Already [B, T, D]

        # Add residual connection using original x
        final_out = x + final_proj_out  # [B, T, D]

        # remove the padding
        final_out = final_out[:, :T, :]

        return final_out
