# ---------------------------------------
# 论文:CBAM: Convolutional Block Attention Module. ArXiv, abs/1807.06521.
# ---------------------------------------


import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):  # in=64,ratio=16
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(
            in_planes, in_planes // ratio, 1, bias=False
        )  # in=64,out=64/16=4,kernel=1
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x[3,64,32,32]
        avg_x = self.avg_pool(x)  # avg_x[3,64,1,1]
        avg_x = self.fc1(avg_x)  # avg_x[3,4,1,1]
        avg_x = self.relu1(avg_x)
        avg_out = self.fc2(avg_x)  # avg_out[3,64,1,1]
        max_x = self.max_pool(x)  # max_x[3,64,1,1]
        max_x = self.fc1(max_x)  # max_x[3,4,1,1]
        max_x = self.relu1(max_x)
        max_out = self.fc2(max_x)  # max_out[3,64,1,1]
        out = avg_out + max_out  # out[3,64,1,1]
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(
            2, 1, kernel_size, padding=padding, bias=False
        )  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x[3,64,32,32]
        avg_out = torch.mean(x, dim=1, keepdim=True)  # avg_out[3,1,32,32]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # max_out[3,1,32,32]
        x = torch.cat([avg_out, max_out], dim=1)  # x[3,2,32,32]
        x = self.conv1(x)  # x[3,1,32,32]
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


# 输入 N C H W,  输出 N C H W
if __name__ == "__main__":
    block = CBAM(64)
    input = torch.rand(3, 64, 32, 32)
    output = block(input)
    print(input.size(), output.size())
