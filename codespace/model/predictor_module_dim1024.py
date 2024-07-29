import torch
import torch.nn as nn
import math

from codespace.model.multihead_attention_transformer import _get_activation_fn


class FC_Decoder(nn.Module):
    def __init__(self, num_class, dim_feedforward, activation, dropout, input_num=3):
        super().__init__()
        self.num_class = num_class

        self.output_layer1 = nn.Linear(dim_feedforward * input_num, dim_feedforward)
        self.activation1 = torch.nn.Sigmoid()
        self.dropout1 = torch.nn.Dropout(dropout)
        self.output_layer2 = nn.Linear(dim_feedforward, dim_feedforward // input_num)
        self.activation2 = torch.nn.Sigmoid()
        self.dropout2 = torch.nn.Dropout(dropout)

        self.output_layer3 = nn.Linear(dim_feedforward // input_num, num_class)

    def forward(self, hs):  # hs[32,3,512*2]
        hs = hs.flatten(1)  # [32,3072]

        hs = self.output_layer1(hs)
        hs = self.activation1(hs)
        hs = self.dropout1(hs)

        hs = self.output_layer2(hs)
        hs = self.activation2(hs)
        hs = self.dropout2(hs)

        # (512//2,GO标签数)
        out = self.output_layer3(hs)
        # 后面还需要一个sigmoid，在测试输出后面直接加了
        return out


class Predictor(nn.Module):
    def __init__(
        self, pre_model, num_class, dim_feedforward, activation, dropout, input_num=3
    ):
        super().__init__()
        self.pre_model = pre_model
        self.fc_decoder = FC_Decoder(
            num_class=num_class,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout,
            input_num=input_num,
        )

    def forward(self, src):
        rec, hs = self.pre_model(src)  # hs[32,3,512*2]
        out = self.fc_decoder(hs)
        return rec, out


def build_predictor(pre_model, args):
    predictor = Predictor(
        pre_model=pre_model,
        num_class=args.num_class,
        dim_feedforward=args.dim_feedforward,
        activation=_get_activation_fn(args.activation),
        dropout=args.dropout,
        input_num=args.input_num,
    )

    return predictor
