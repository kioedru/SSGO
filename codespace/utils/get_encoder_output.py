import numpy as np
import pandas as pd
import argparse
import time
import random
import torch
import torch.nn.parallel
from torch.optim import lr_scheduler
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import os
import sys

sys.path.append("/home/Kioedru/code/SSGO")

from codespace.model import aslloss_adaptive

from sklearn.preprocessing import minmax_scale
import csv
from codespace.q2l.q2l_3.predictor_module_q2l import build_predictor


class multimodesDataset(torch.utils.data.Dataset):
    def __init__(self, num_modes, modes_features, labels):
        self.modes_features = modes_features
        self.labels = labels
        self.num_modes = num_modes

    def __getitem__(self, index):
        modes_features = []
        for i in range(self.num_modes):
            modes_features.append(self.modes_features[i][index])
        return modes_features, self.labels[index]

    def __len__(self):
        return self.modes_features[0].size(0)


from codespace.utils.read_finetune_data import (
    read_feature_by_index,
    read_labels,
    read_ppi_by_index,
    read_seq_embed_avgpool_esm2_2000_by_index,
    read_seq_embed_avgpool_esm2_480_by_index,
    read_seq_embed_avgpool_prott5_1024_by_index,
)


def perf_write_to_csv(args, epoch, perf, loss, time, lr):
    if not os.path.exists(args.epoch_performance_path):
        with open(args.epoch_performance_path, "w") as f:
            csv.writer(f).writerow(
                ["epoch", "loss", "time", "lr", "m-aupr", "Fmax", "M-aupr", "F1", "acc"]
            )

    with open(args.epoch_performance_path, "a") as f:
        csv.writer(f).writerow(
            [
                epoch,
                loss,
                time,
                lr,
                perf["m-aupr"],
                perf["Fmax"],
                perf["M-aupr"],
                perf["F1"],
                perf["acc"],
            ]
        )


# 检查并创建文件夹
def check_and_create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 已创建。")
    else:
        print(f"文件夹 '{folder_path}' 已存在。")


# prott5:[num,1024]
def get_finetune_data(args, usefor, aspect, organism_num):
    feature = read_feature_by_index(usefor, aspect, organism_num)
    ppi_matrix = read_ppi_by_index(usefor, aspect, organism_num)
    if args.seq_feature == "seq480":
        seq = read_seq_embed_avgpool_esm2_480_by_index(usefor, aspect, organism_num)
    elif args.seq_feature == "seq1024":
        seq = read_seq_embed_avgpool_prott5_1024_by_index(usefor, aspect, organism_num)
    labels = read_labels(usefor, aspect, organism_num)
    return feature, seq, ppi_matrix, labels


def get_dataset(args, aspect, organism_num):
    train_feature, train_seq, train_ppi_matrix, train_labels = get_finetune_data(
        args, "train", aspect, organism_num
    )
    valid_feature, valid_seq, valid_ppi_matrix, valid_labels = get_finetune_data(
        args, "valid", aspect, organism_num
    )
    test_feature, test_seq, test_ppi_matrix, test_labels = get_finetune_data(
        args, "test", aspect, organism_num
    )

    combine_feature = np.concatenate((train_feature, valid_feature), axis=0)
    combine_seq = np.concatenate((train_seq, valid_seq), axis=0)
    combine_ppi_matrix = np.concatenate((train_ppi_matrix, valid_ppi_matrix), axis=0)
    combine_labels = np.concatenate((train_labels, valid_labels), axis=0)

    combine_feature = torch.from_numpy(combine_feature).float()
    combine_seq = torch.from_numpy(combine_seq).float()
    combine_ppi_matrix = torch.from_numpy(combine_ppi_matrix).float()
    combine_labels = torch.from_numpy(combine_labels).float()
    test_feature = torch.from_numpy(test_feature).float()
    test_seq = torch.from_numpy(test_seq).float()
    test_ppi_matrix = torch.from_numpy(test_ppi_matrix).float()
    test_labels = torch.from_numpy(test_labels).float()

    train_dataset = multimodesDataset(
        3, [combine_ppi_matrix, combine_feature, combine_seq], combine_labels
    )
    test_dataset = multimodesDataset(
        3, [test_ppi_matrix, test_feature, test_seq], test_labels
    )
    modefeature_lens = [
        combine_ppi_matrix.shape[1],
        combine_feature.shape[1],
        combine_seq.shape[1],
    ]
    print("combine_ppi_matrix = ", combine_ppi_matrix.shape)

    return train_dataset, test_dataset, modefeature_lens


def parser_args():
    parser = argparse.ArgumentParser(description="CFAGO main")
    parser.add_argument("--org", help="organism")
    parser.add_argument(
        "--aspect", type=str, default="P", choices=["P", "F", "C"], help="GO aspect"
    )
    parser.add_argument("--num_class", default=45, type=int, help="标签数")
    parser.add_argument("--pretrained_model", type=str, help="输入的预训练模型的路径")
    parser.add_argument("--finetune_model", type=str, help="输出的微调模型的路径")
    parser.add_argument("--performance_path", type=str, help="输出的指标的路径")

    parser.add_argument("--dataset_dir", help="dir of dataset")
    parser.add_argument("--output", metavar="DIR", help="path to output folder")

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
        "--dtgfl",
        action="store_true",
        default=False,
        help="disable_torch_grad_focal_loss in asl",
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
        default=200,
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
        "--seed", default=1329765522, type=int, help="seed for initializing training. "
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
    parser.add_argument(
        "--pretrain-update",
        default=2,
        type=int,
        help="参数更新方式",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
    )
    parser.add_argument(
        "--nni",
        default=False,
        type=bool,
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
        "--pre_lr",
        default=3e-3,
        type=float,
    )
    parser.add_argument(
        "--seq_pre_lr",
        default=5e-3,
        type=float,
    )
    parser.add_argument(
        "--fusion_lr",
        default=1e-4,
        type=float,
    )
    parser.add_argument(
        "--param",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--fusion",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--seq_feature",
        default="seq1024",  # seq1024
        type=str,
    )
    args = parser.parse_args()
    return args


def get_args():
    args = parser_args()
    return args


import nni


# nohup python -u /home/Kioedru/code/SSGO/codespace/utils/get_encoder_output.py --device cuda:1 --seed 1329765522 --aspect P --num_class 45 &
# nohup python -u /home/Kioedru/code/SSGO/codespace/utils/get_encoder_output.py --device cuda:1 --seed 1329765522 --aspect F --num_class 38 &
# nohup python -u /home/Kioedru/code/SSGO/codespace/utils/get_encoder_output.py --device cuda:1 --seed 1329765522 --aspect C --num_class 35 &


def main():
    args = get_args()

    args.org = "9606"

    # /home/Kioedru/code/SSGO/codespace/pretrain/one_feature_only/9606/transformer_seq480_only.pkl
    args.seq_model_backbone = "bimamba"
    args.seq_model_name = f"{args.seq_model_backbone}_{args.seq_feature}_only"
    # /home/Kioedru/code/SSGO/codespace/pretrain/bimamba/9606/bimamba.pkl
    args.ppi_feature_model_name = f"transformer"
    path_in_kioedru = f"/home/kioedru/code/SSGO/codespace"
    path_in_Kioedru = f"/home/Kioedru/code/SSGO/codespace"
    if os.path.exists(path_in_kioedru):
        args.path = path_in_kioedru
    else:
        args.path = path_in_Kioedru

    # 预训练模型：Sequence的路径
    args.seq_pretrained_model = os.path.join(
        args.path,
        "pretrain",
        "one_feature_only",
        args.org,
        f"{args.seq_model_name}.pkl",
    )
    # 预训练模型：ppi+亚细胞+结构域的路径
    args.ppi_feature_pretrained_model = os.path.join(
        args.path,
        "pretrain",
        args.ppi_feature_model_name,
        args.org,
        f"{args.ppi_feature_model_name}.pkl",
    )

    return main_worker(args)


def main_worker(args):

    # 准备数据集,esm2+prott5时 seq_2=True
    train_dataset, test_dataset, args.modesfeature_len = get_dataset(
        args, args.aspect, args.org
    )
    args.encode_structure = [1024]

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        # 指定用多少个子进程来加载数据，CFAGO中默认为32
        num_workers=args.workers,
        # 是否将加载的数据保存在锁页内存中（以占用更多内存的代价，加快数据从CPU到GPU的转移速度）
        pin_memory=True,
        sampler=train_sampler,
        drop_last=False,
    )
    test_sampler = None
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=test_sampler,
    )

    torch.cuda.empty_cache()

    ppi_feature_pre_model = torch.load(
        args.ppi_feature_pretrained_model, map_location=args.device
    )
    seq_pre_model = torch.load(args.seq_pretrained_model, map_location=args.device)

    finetune(
        args,
        train_loader,
        test_loader,
        ppi_feature_pre_model,
        seq_pre_model,
        args.device,
    )


@torch.no_grad()
def finetune(
    args,
    data_loader,
    test_loader,
    ppi_feature_model,
    seq_model,
    device,
):
    ppi_feature_model = ppi_feature_model.to(device)
    seq_model = seq_model.to(device)
    print("training on", device)

    all_ppi_feature_embed = []
    all_seq_embed = []
    for protein_data, label in data_loader:

        protein_data[0] = protein_data[0].to(device)
        protein_data[1] = protein_data[1].to(device)
        protein_data[2] = protein_data[2].to(device)
        label = label.to(device)
        ppi_feature_src = protein_data  # 2,32,512
        seq_src = protein_data[2].unsqueeze(0)  # 1,32,512
        _, hs_ppi_feature = ppi_feature_model(ppi_feature_src)  # 2,32,512
        _, hs_seq = seq_model(seq_src)
        all_ppi_feature_embed.append(hs_ppi_feature.detach().cpu())
        all_seq_embed.append(hs_seq.detach().cpu())

    all_ppi_feature_embed = torch.cat(all_ppi_feature_embed, 1).numpy()
    all_seq_embed = torch.cat(all_seq_embed, 1).numpy()
    embed_path = "/home/Kioedru/code/SSGO/data/encoder"
    pd.to_pickle(
        all_ppi_feature_embed,
        os.path.join(embed_path, f"train_{args.aspect}_ppi_feature_embed.pkl"),
    )
    pd.to_pickle(
        all_seq_embed, os.path.join(embed_path, f"train_{args.aspect}_seq_embed.pkl")
    )

    all_ppi_feature_embed = []
    all_seq_embed = []
    for protein_data, label in test_loader:

        protein_data[0] = protein_data[0].to(device)
        protein_data[1] = protein_data[1].to(device)
        protein_data[2] = protein_data[2].to(device)
        label = label.to(device)
        ppi_feature_src = protein_data  # 2,32,512
        seq_src = protein_data[2].unsqueeze(0)  # 1,32,512
        _, hs_ppi_feature = ppi_feature_model(ppi_feature_src)  # 2,32,512
        _, hs_seq = seq_model(seq_src)
        all_ppi_feature_embed.append(hs_ppi_feature.detach().cpu())
        all_seq_embed.append(hs_seq.detach().cpu())

    all_ppi_feature_embed = torch.cat(all_ppi_feature_embed, 1).numpy()
    all_seq_embed = torch.cat(all_seq_embed, 1).numpy()
    embed_path = "/home/Kioedru/code/SSGO/data/encoder"
    pd.to_pickle(
        all_ppi_feature_embed,
        os.path.join(embed_path, f"test_{args.aspect}_ppi_feature_embed.pkl"),
    )
    pd.to_pickle(
        all_seq_embed, os.path.join(embed_path, f"test_{args.aspect}_seq_embed.pkl")
    )
    print("ok")


if __name__ == "__main__":
    main()
