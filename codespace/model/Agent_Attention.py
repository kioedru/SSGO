import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


class AgentAttention(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        **kwargs,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        agent_num = seq_len
        self.dim = dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.agent_num = agent_num
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, seq_len))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, seq_len))
        trunc_normal_(self.an_bias, std=0.02)
        trunc_normal_(self.na_bias, std=0.02)
        self.pool = nn.AdaptiveAvgPool1d(output_size=agent_num)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads
        q = self.q(x)

        kv = self.kv(x).reshape(b, n, 2, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        agent_tokens = self.pool(q.permute(0, 2, 1)).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, num_heads, n, head_dim)
        v = v.reshape(b, num_heads, n, head_dim)
        agent_tokens = agent_tokens.reshape(b, num_heads, self.agent_num, head_dim)

        position_bias1 = self.an_bias.repeat(b, 1, 1, 1)
        position_bias2 = self.na_bias.repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2

        agent_attn = self.softmax(
            (agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias
        )
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        agent_bias = position_bias1 + position_bias2
        q_attn = self.softmax(
            (q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias
        )
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == "__main__":
    dim = 512
    seq_len = 3

    block = AgentAttention(dim=dim, seq_len=seq_len)

    x = torch.rand(32, seq_len, dim)

    # Forward pass
    output = block(x)
    print(f"Input size: {x.size()}")
    print(f"Output size: {output.size()}")
