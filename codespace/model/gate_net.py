import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os


def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class GateNet(nn.Module):
    def __init__(self, input_dim, branch_num):
        super(GateNet, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, branch_num)
        )

    def forward(self, x):
        logits = self.MLP(x)
        weight = DiffSoftmax(logits, tau=1.0, hard=True)  # 硬门控
        return weight


# output = weight[:, 0:1] * self.branch1(inputs[0]) + weight[:, 1:2] * self.branch2(inputs[1])
if __name__ == "__main__":
    block = GateNet(512 * 3, 2)
    input = torch.rand(32, 512 * 3)
    output = block(input)
    print(input.size(), output.size())
    print(output)