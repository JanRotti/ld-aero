import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    __doc__ = r"""Applies QKV self-attention with a residual connection.
    
    Input:
        x: tensor of shape (N, in_channels, H, W)
        norm (None or nn.Module): which normalization to use. Default: None
    Output:
        tensor of shape (N, in_channels, H, W)
    Args:
        in_channels (int): number of input channels
    """
    def __init__(self, in_channels, norm=None):
        super().__init__()
        
        self.in_channels = in_channels
        self.norm = norm if norm is not None else nn.Identity()
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.to_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x, time=None, y=None):
        b, c, h, w = x.shape
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)

        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        k = k.view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)

        dot_products = torch.bmm(q, k) * (c ** (-0.5))
        assert dot_products.shape == (b, h * w, h * w)

        attention = torch.softmax(dot_products, dim=-1)
        out = torch.bmm(attention, v)
        assert out.shape == (b, h * w, c)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)

        return self.to_out(out) + x


class MultiHeadAttentionBlock(nn.Module):
    __doc__ = r"""Applies multi-head QKV self-attention with a residual connection.
    
    Input:
        x: tensor of shape (N, in_channels, H, W)
        norm (None or nn.Module): which normalization to use. Default: None
        num_heads (int): number of attention heads. Default: 4
    Output:
        tensor of shape (N, in_channels, H, W)
    Args:
        in_channels (int): number of input channels
    """
    def __init__(self, in_channels, norm=None, num_heads=4):
        super().__init__()
        
        self.in_channels = in_channels
        self.norm = norm if norm is not None else nn.Identity()
        self.num_heads = num_heads
        
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3 * num_heads, 1)
        self.to_out = nn.Conv2d(in_channels * num_heads, in_channels, 1)

    def forward(self, x, time=None, y=None):
        b, c, h, w = x.shape
        qkv = self.to_qkv(self.norm(x))
        qkv = qkv.view(b, self.num_heads, -1, h, w)
        q, k, v = torch.split(qkv, c, dim=2)

        q = q.permute(0, 1, 3, 4, 2).view(b * self.num_heads, h * w, c)
        k = k.view(b * self.num_heads, c, h * w)
        v = v.permute(0, 1, 3, 4, 2).view(b * self.num_heads, h * w, c)

        dot_products = torch.bmm(q, k) * (c ** (-0.5))
        assert dot_products.shape == (b * self.num_heads, h * w, h * w)

        attention = torch.softmax(dot_products, dim=-1)
        out = torch.bmm(attention, v)
        assert out.shape == (b * self.num_heads, h * w, c)
        out = out.view(b, self.num_heads, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
        out = out.view(b, self.num_heads * c, h, w)

        return self.to_out(out) + x