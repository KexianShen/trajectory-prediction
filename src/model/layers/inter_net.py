from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.w


class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, num_heads=8, dim=128):
        """
        Initialize the Attention module.

        Args:
            num_heads
            dim

        Attributes:
            head_dim (int): Dimension size of each attention head.
            wq (nn.Linear): Linear transformation for queries.
            wk (nn.Linear): Linear transformation for keys.
            wv (nn.Linear): Linear transformation for values.
            wo (nn.Linear): Linear transformation for output.

        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.wq = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_heads * self.head_dim, dim, bias=False)

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = xq.shape
        xq, xk, xv = self.wq(xq), self.wk(xk), self.wv(xv)

        xq = xq.view(bsz, seqlen, self.num_heads, self.head_dim)
        xk = xk.view(bsz, -1, self.num_heads, self.head_dim)
        xv = xv.view(bsz, -1, self.num_heads, self.head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / (self.head_dim**0.5)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.

        Attributes:
            w1 (nn.Linear): Linear transformation for the first layer.
            w2 (nn.Linear): Linear transformation for the second layer.
            w3 (nn.Linear): Linear transformation for the third layer.

        """
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(
        self, layer_id: int, num_heads: int = 8, dim: int = 128, norm_eps: float = 1e-5
    ):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            num_heads
            dim
            norm_eps

        Attributes:
            num_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.attention = Attention(num_heads, dim)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=4 * dim)
        self.layer_id = layer_id
        self.attention_q_norm = RMSNorm(dim, eps=norm_eps)
        self.attention_k_norm = RMSNorm(dim, eps=norm_eps)
        self.attention_v_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = xq + self.attention.forward(
            self.attention_q_norm(xq),
            self.attention_k_norm(xk),
            self.attention_v_norm(xv),
            mask,
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class InterNet(nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        num_heads: int = 8,
        dim: int = 128,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.self_atten_layers = torch.nn.ModuleList()
        self.cross_atten_layers = torch.nn.ModuleList()
        for layer_id in range(num_layers):
            self.self_atten_layers.append(
                TransformerBlock(layer_id, num_heads, dim, norm_eps)
            )
            self.cross_atten_layers.append(
                TransformerBlock(layer_id, num_heads, dim, norm_eps)
            )
        self.norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        agent_emb: torch.Tensor,
        modal_query: torch.Tensor,
        scene_emb: torch.Tensor,
    ):
        mask = None

        for layer_id in range(self.num_layers):
            q = k = agent_emb + modal_query
            agent_emb = self.self_atten_layers[layer_id](q, k, agent_emb, mask)
            agent_emb = self.cross_atten_layers[layer_id](
                agent_emb, scene_emb, scene_emb, mask
            )
        agent_emb = self.norm(agent_emb)
        return agent_emb


class SceneNet(nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        num_heads: int = 8,
        dim: int = 128,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.self_atten_layers = torch.nn.ModuleList()
        for layer_id in range(num_layers):
            self.self_atten_layers.append(
                TransformerBlock(layer_id, num_heads, dim, norm_eps)
            )
        self.norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        scene_feat: torch.Tensor,
    ):
        mask = None

        for layer_id in range(self.num_layers):
            scene_feat = self.self_atten_layers[layer_id](
                scene_feat, scene_feat, scene_feat, mask
            )
        scene_feat = self.norm(scene_feat)
        return scene_feat
