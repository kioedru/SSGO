import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x[32,512,3]
        avg_x = self.avg_pool(x)  # avg_x[32,512,1]
        avg_x = self.fc1(avg_x)  # avg_x[32,64,1]
        avg_x = self.relu1(avg_x)
        avg_out = self.fc2(avg_x)  # avg_out[32,512,1]
        max_x = self.max_pool(x)  # max_x[32,512,1]
        max_x = self.fc1(max_x)  # max_x[32,64,1]
        max_x = self.relu1(max_x)
        max_out = self.fc2(max_x)  # max_out[32,512,1]
        out = avg_out + max_out  # out[32,512,1]
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x[32,512,3]
        avg_out = torch.mean(x, dim=1, keepdim=True)  # avg_out[32,1,3]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # max_out[32,1,3]
        x = torch.cat([avg_out, max_out], dim=1)  # x[32,2,3]
        x = self.conv1(x)  # x[32,1,3]
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # Apply channel attention
        out = x * self.channel_attention(x)
        # Apply spatial attention
        out = out * self.spatial_attention(out)
        return out


if __name__ == "__main__":
    block = CBAM(512)
    input = torch.rand(32, 512, 3)
    output = block(input)
    print(input.size(), output.size())
