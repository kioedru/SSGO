import torch
import torch.nn as nn
import math


class Predictor(nn.Module):
    def __init__(
        self,
        dropout,
        num_class=45,
    ):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_class)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src):
        src = self.fc1(src)
        src = self.activation(src)
        src = self.dropout(src)
        src = self.fc2(src)
        return src
