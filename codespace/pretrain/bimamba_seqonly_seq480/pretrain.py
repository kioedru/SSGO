import numpy as np
import pandas as pd
import argparse
import time
import random
import torch
import torch.nn.parallel
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import os
import copy
import csv
from sklearn.preprocessing import minmax_scale
from transformers import AutoTokenizer, AutoModel

from codespace.pretrain.bimamba_seqonly_seq480.pretrain_model import (
    build_Pre_Train_Model,
)
from codespace.model import aslloss_adaptive
from codespace.utils.read_pretrain_data import (
    read_ppi,
    read_feature,
    # read_seq_embed_avgpool_esm2_480,
    read_seq,
)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", val_only=False):
        self.name = name
        self.fmt = fmt
        self.val_only = val_only
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if self.val_only:
            fmtstr = "{name} {val" + self.fmt + "}"
        else:
            fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class multimodesFullDataset(torch.utils.data.Dataset):
    def __init__(self, num_modes, modes_features):
        self.modes_features = modes_features
        self.num_modes = num_modes

    def __getitem__(self, index):
        modes_features = []
        for i in range(self.num_modes):
            if self.modes_features[i][index] is None:
                modes_features.append("None")
            else:
                modes_features.append(self.modes_features[i][index])
        return modes_features

    def __len__(self):
        return self.modes_features[0].size(0)


def get_ssl_datasets(organism_num):
    ppi_matrix, ppi_id = read_ppi(organism_num)
    feature = read_feature(organism_num)
    # 得到以ppi_id顺序排列的蛋白质氨基酸序列
    seq = read_seq(organism_num)

    # 归一化
    ppi_matrix = minmax_scale(ppi_matrix)
    feature = minmax_scale(feature)

    ppi_matrix = torch.from_numpy(ppi_matrix).float()
    feature = torch.from_numpy(feature).float()

    full_dataset = multimodesFullDataset(3, [ppi_matrix, feature, seq])
    return full_dataset, [ppi_matrix.shape[1], feature.shape[1]]


def parser_args():
    parser = argparse.ArgumentParser(description="CFAGO self-supervised Training")
    parser.add_argument("--org", help="organism")
    parser.add_argument(
        "--dataset_dir", help="dir of dataset", default="DANE/Database/human"
    )
    parser.add_argument("--aspect", type=str, choices=["P", "F", "C"], help="GO aspect")

    parser.add_argument("--output", metavar="DIR", help="path to output folder")
    parser.add_argument(
        "--num_class", default=45, type=int, help="Number of class labels"
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="use pre-trained model. default is False. ",
    )
    parser.add_argument(
        "--optim",
        default="AdamW",
        type=str,
        choices=["AdamW", "Adam_twd"],
        help="which optim to use",
    )

    # loss
    parser.add_argument(
        "--eps", default=1e-5, type=float, help="eps for focal loss (default: 1e-5)"
    )
    parser.add_argument(
        "--gamma_pos",
        default=0,
        type=float,
        metavar="gamma_pos",
        help="gamma pos for simplified asl loss",
    )
    parser.add_argument(
        "--gamma_neg",
        default=2,
        type=float,
        metavar="gamma_neg",
        help="gamma neg for simplified asl loss",
    )
    parser.add_argument(
        "--loss_dev", default=-1, type=float, help="scale factor for loss"
    )
    parser.add_argument(
        "--loss_clip", default=0.0, type=float, help="scale factor for clip"
    )

    parser.add_argument(
        "-j",
        "--workers",
        default=32,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        default=500,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )

    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=32,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs",
    )

    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=1e-5,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-2,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-2)",
        dest="weight_decay",
    )

    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument("--resume_omit", default=[], type=str, nargs="*")
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )

    parser.add_argument(
        "--ema-decay",
        default=0.9997,
        type=float,
        metavar="M",
        help="decay of model ema",
    )
    parser.add_argument(
        "--ema-epoch", default=0, type=int, metavar="M", help="start ema epoch"
    )

    # distribution training
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--local_rank", type=int, help="local rank for DistributedDataParallel"
    )

    # data aug
    parser.add_argument(
        "--cutout", action="store_true", default=False, help="apply cutout"
    )
    parser.add_argument(
        "--n_holes", type=int, default=1, help="number of holes to cut out from image"
    )
    parser.add_argument(
        "--length",
        type=int,
        default=-1,
        help="length of the holes. suggest to use default setting -1.",
    )
    parser.add_argument(
        "--cut_fact", type=float, default=0.5, help="mutual exclusion with length. "
    )

    parser.add_argument(
        "--norm_norm",
        action="store_true",
        default=False,
        help="using mormal scale to normalize input features",
    )

    # * Transformer
    parser.add_argument(
        "--attention_layers",
        default=6,
        type=int,
        help="Number of layers of each multi-head attention module",
    )

    parser.add_argument(
        "--dim_feedforward",
        default=512,
        type=int,
        help="Intermediate size of the feedforward layers in the multi-head attention blocks",
    )
    parser.add_argument(
        "--activation",
        default="gelu",
        type=str,
        choices=["relu", "gelu", "lrelu", "sigmoid"],
        help="Number of attention heads inside the multi-head attention module's attentions",
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="Dropout applied in the multi-head attention module",
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the multi-head attention module's attentions",
    )
    parser.add_argument("--pre_norm", action="store_true")

    # * raining
    parser.add_argument("--amp", action="store_true", default=False, help="apply amp")
    parser.add_argument(
        "--early-stop", action="store_true", default=False, help="apply early stop"
    )
    parser.add_argument(
        "--kill-stop", action="store_true", default=False, help="apply early stop"
    )
    args = parser.parse_args()
    return args


def get_args():
    args = parser_args()
    return args


# in kioedru:
# nohup python /home/kioedru/code/SSGO/codespace/pretrain/mamba3_seq480_residue/pretrain.py> /home/kioedru/code/SSGO/codespace/pretrain/mamba3_seq480_residue/pretrain.log 2>&1 &
# in Kioedru:
# nohup python /home/Kioedru/code/SSGO/codespace/pretrain/mamba3_seq480_residue/pretrain.py> /home/Kioedru/code/SSGO/codespace/pretrain/mamba3_seq480_residue/pretrain.log 2>&1 &
def main():
    args = get_args()
    args.residue_dim = 480
    args.model_name = f"bimamba_seqonly_seq480"
    pretrain_path_in_kioedru = f"/home/kioedru/code/SSGO/codespace/pretrain"
    pretrain_path_in_Kioedru = f"/home/Kioedru/code/SSGO/codespace/pretrain"
    if os.path.exists(pretrain_path_in_kioedru):
        args.pretrain_path = pretrain_path_in_kioedru
    else:
        args.pretrain_path = pretrain_path_in_Kioedru

    args.pretrain_model = os.path.join(
        args.pretrain_path, args.model_name, f"{args.model_name}.pkl"
    )
    args.performance_path = os.path.join(
        args.pretrain_path, args.model_name, f"pretrain_loss.csv"
    )
    args.org = "human"
    args.seed = int(1329765522)
    args.dim_feedforward = int(512)
    args.nheads = int(8)
    args.dropout = float(0.1)
    args.attention_layers = int(6)
    args.batch_size = int(32)
    args.activation = "gelu"
    args.epochs = int(5000)
    args.lr = float(1e-5)
    args.device = "cuda:1"

    # # 指定随机种子初始化随机数生成器（保证实验的可复现性）
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    # 使用一个隐藏层
    args.h_n = 1
    return main_worker(args)


def main_worker(args):
    esm_path = "/home/Kioedru/code/SSGO/data/LLM/esm2"

    tokenizer = AutoTokenizer.from_pretrained(esm_path)
    model = AutoModel.from_pretrained(esm_path)
    # print(model)
    embedding_model = model.embeddings.to(args.device)

    full_dataset, args.modesfeature_len = get_ssl_datasets("9606")
    args.modesfeature_len.append(args.residue_dim)
    args.encode_structure = [1024]
    # build model
    pre_model = build_Pre_Train_Model(args)

    # 使用对称损失函数
    pretrain_loss = aslloss_adaptive.pretrainLossOptimized(
        clip=args.loss_clip,
        eps=args.eps,
    )

    # optimizer
    args.lr_mult = 1
    pre_model_param_dicts = [
        {"params": [p for n, p in pre_model.named_parameters() if p.requires_grad]},
    ]
    # 使用AdamW
    optimizer = getattr(torch.optim, "AdamW")(
        pre_model_param_dicts,
        args.lr_mult * args.lr,
        betas=(0.9, 0.999),
        eps=1e-09,
        weight_decay=0,
    )

    full_sampler = None
    full_loader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=args.batch_size,  # 32，每次取32个进行训练
        shuffle=(full_sampler is not None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=full_sampler,
        drop_last=False,
    )

    # 每隔2500个epoch就把学习率乘0.01
    steplr = lr_scheduler.StepLR(optimizer, 2500)
    pre_train(
        tokenizer,
        embedding_model,
        args,
        full_loader,
        pre_model,
        pretrain_loss,
        optimizer,
        steplr,
        args.epochs,
        device=args.device,
    )

    torch.save(pre_model, args.pretrain_model)


def create_log(args):
    # 定义csv文件的路径
    csv_path = args.performance_path

    with open(csv_path, "w") as f:
        csv.writer(f).writerow(["Epoch", "Loss", "lr", "Time"])

    return csv_path


def pre_train(
    tokenizer,
    embedding_model,
    args,
    full_loader,
    pretrain_model,
    pretrain_loss,
    optimizer,
    steplr,
    num_epochs,
    device,
):
    csv_path = create_log(args)

    net = pretrain_model.to(device)
    print("training on", device)
    for epoch in range(num_epochs):
        start = time.time()
        batch_count = 0
        train_l_sum = 0.0
        for protein_data in full_loader:
            torch.cuda.empty_cache()
            protein_data[0] = protein_data[0].to(device)
            protein_data[1] = protein_data[1].to(device)
            # protein_data[2] = protein_data[2].to(device)
            seq_emb = Embed_Seq(tokenizer, embedding_model, protein_data[2], args)
            ori = copy.deepcopy(protein_data[:2])
            rec, hs, ori_3 = net(protein_data, seq_emb)
            ori.append(ori_3)

            l = pretrain_loss(ori, rec, hs)
            optimizer.zero_grad()
            l.backward()
            train_l_sum += l.cpu().item()
            # record loss
            optimizer.step()  # 优化方法
            batch_count += 1
        steplr.step()

        # 添加数据到csv
        with open(csv_path, "a") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    train_l_sum / batch_count,
                    optimizer.param_groups[0]["lr"],
                    time.time() - start,
                ]
            )


def Embed_Seq(tokenizer, embedding_model, seqs, args):
    all_seq_embeds = []
    for seq in seqs:
        if seq == "None":
            seq_embed = torch.zeros(1, 2000, args.residue_dim).to(args.device)

        else:

            inputs = tokenizer(
                seq,
                padding="max_length",
                truncation=True,
                max_length=2000,
                return_tensors="pt",
            )
            inputs = inputs.to(args.device)
            # 获取嵌入
            with torch.no_grad():
                seq_embed = embedding_model(**inputs)
        all_seq_embeds.append(seq_embed)
    all_seq_embeds = torch.stack(all_seq_embeds, dim=0)
    all_seq_embeds = all_seq_embeds.squeeze(dim=1)
    # 找到最小和最大值
    min_val = torch.min(all_seq_embeds)
    max_val = torch.max(all_seq_embeds)
    # 进行归一化
    normalized_tensor = (all_seq_embeds - min_val) / (max_val - min_val)
    # print(all_seq_embeds.shape)
    return normalized_tensor


if __name__ == "__main__":
    main()
