# --------------------------------------------------------
# part of code borrowed from Quert2Label
# Written by Zhourun Wu
# --------------------------------------------------------
import numpy as np
import pandas as pd
import torch.distributed as dist
import argparse
import time
from copy import deepcopy
import random
import torch
import torch.nn.parallel
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import os

# from pretrain_model import build_Pre_Train_Model
import aslloss

import aslloss_4
import copy

# from read_pretrain_data import (
#     read_feature,
#     read_ppi,
#     read_seq_embedding_avgpooling,
# )
from sklearn.preprocessing import minmax_scale
import csv
from predictor_module import build_predictor


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


from read_finetune_data import (
    read_feature_by_index,
    read_labels,
    read_ppi_by_index,
    read_seq_embedding_avgpooling_by_index,
    read_seq_embedding_avgpooling_esm2_prott5_by_index,
    read_seq_embedding_avgpooling_esm2_480_prott5_1024_by_index,
    read_seq_esm2_480_by_index,
    read_seq_prott5_1024_by_index,
    read_esm2_480_and_prott5_1024_respectively_by_index,
    read_seq_onehot,
)


def get_finetune_data(usefor, aspect, organism_num, seq_2, onehot):
    feature = read_feature_by_index(usefor, aspect, organism_num)
    if seq_2 == 1:  # esm2+prott5 (2000,1024)
        seq = read_seq_embedding_avgpooling_esm2_prott5_by_index(
            usefor, aspect, organism_num
        )
    elif seq_2 == 0:  # esm2 (2000)
        seq = read_seq_embedding_avgpooling_by_index(usefor, aspect, organism_num)
    elif seq_2 == 2:  # esm2+prott5 (480,1024)
        seq = read_seq_embedding_avgpooling_esm2_480_prott5_1024_by_index(
            usefor, aspect, organism_num
        )
    elif seq_2 == 3:  # esm2(480)
        seq = read_seq_esm2_480_by_index(usefor, aspect, organism_num)
    elif seq_2 == 4:  # prott5(1024)
        seq = read_seq_prott5_1024_by_index(usefor, aspect, organism_num)
    ppi_matrix = read_ppi_by_index(usefor, aspect, organism_num)
    labels = read_labels(usefor, aspect, organism_num)
    if seq_2 == 5:  # esm2(480)+prott5(1024) （分别）
        esm2, prott5 = read_esm2_480_and_prott5_1024_respectively_by_index(
            usefor, aspect, organism_num
        )
        return feature, esm2, prott5, ppi_matrix, labels
    if onehot:
        seq_onehot = read_seq_onehot(usefor, aspect, organism_num)
        seq = np.concatenate((seq, seq_onehot), axis=1)  # 横向拼接onehot
    return feature, seq, ppi_matrix, labels


def get_4features_dataset(aspect, organism_num, seq_2):
    train_feature, train_esm2, train_prott5, train_ppi_matrix, train_labels = (
        get_finetune_data("train", aspect, organism_num, seq_2)
    )
    valid_feature, valid_esm2, valid_prott5, valid_ppi_matrix, valid_labels = (
        get_finetune_data("valid", aspect, organism_num, seq_2)
    )
    test_feature, test_esm2, test_prott5, test_ppi_matrix, test_labels = (
        get_finetune_data("test", aspect, organism_num, seq_2)
    )

    combine_feature = np.concatenate((train_feature, valid_feature), axis=0)
    combine_esm2 = np.concatenate((train_esm2, valid_esm2), axis=0)
    combine_prott5 = np.concatenate((train_prott5, valid_prott5), axis=0)
    combine_ppi_matrix = np.concatenate((train_ppi_matrix, valid_ppi_matrix), axis=0)
    combine_labels = np.concatenate((train_labels, valid_labels), axis=0)

    combine_feature = torch.from_numpy(combine_feature).float()
    combine_esm2 = torch.from_numpy(combine_esm2).float()
    combine_prott5 = torch.from_numpy(combine_prott5).float()
    combine_ppi_matrix = torch.from_numpy(combine_ppi_matrix).float()
    combine_labels = torch.from_numpy(combine_labels).float()
    test_feature = torch.from_numpy(test_feature).float()
    test_esm2 = torch.from_numpy(test_esm2).float()
    test_prott5 = torch.from_numpy(test_prott5).float()
    test_ppi_matrix = torch.from_numpy(test_ppi_matrix).float()
    test_labels = torch.from_numpy(test_labels).float()

    train_dataset = multimodesDataset(
        4,
        [combine_ppi_matrix, combine_feature, combine_esm2, combine_prott5],
        combine_labels,
    )
    test_dataset = multimodesDataset(
        4, [test_ppi_matrix, test_feature, test_esm2, test_prott5], test_labels
    )
    modefeature_lens = [
        combine_ppi_matrix.shape[1],
        combine_feature.shape[1],
        combine_esm2.shape[1],
        combine_prott5.shape[1],
    ]
    print("combine_ppi_matrix = ", combine_ppi_matrix.shape)

    return train_dataset, test_dataset, modefeature_lens


def get_dataset(aspect, organism_num, seq_2, onehot):
    train_feature, train_seq, train_ppi_matrix, train_labels = get_finetune_data(
        "train", aspect, organism_num, seq_2, onehot
    )
    valid_feature, valid_seq, valid_ppi_matrix, valid_labels = get_finetune_data(
        "valid", aspect, organism_num, seq_2, onehot
    )
    test_feature, test_seq, test_ppi_matrix, test_labels = get_finetune_data(
        "test", aspect, organism_num, seq_2, onehot
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
    parser.add_argument("--aspect", type=str, choices=["P", "F", "C"], help="GO aspect")
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


# python CFAGO-code/self_supervised_leaning.py --org human --dataset_dir Dataset/human --output human_result --dist-url tcp://127.0.0.1:3723 --seed 1329765522 --dim_feedforward 512 --nheads 8 --dropout 0.1 --attention_layers 6 --batch-size 32 --activation gelu --epochs 5000 --lr 1e-5


# nohup python CFAGO-code/self_supervised_leaning.py --dist-url tcp://127.0.0.1:3732 --aspect P> log/pre_train/P.log 2>&1 &
def main():
    args = get_args()
    args.epochs = 200
    args.se = False
    args.pretrain_update = 0  # 0全更新，1不更新，2更新一半
    args.org = "9606"
    # args.aspect = "P"
    # args.num_class = int(45)
    # args.seed = int(
    #     1329765519
    # )  #  1329765522  132976111  1329765525    1329765529  1329765519
    model_name = f"mamba3_seq480"

    args.pretrained_model = f"/home/kioedru/code/CFAGO/CFAGO_seq/result/model/pretrain_model_{model_name}.pkl"
    args.finetune_model = f"/home/kioedru/code/CFAGO/CFAGO_seq/result/model/finetune_model_{args.aspect}_{model_name}.pkl"
    args.performance_path = f"/home/kioedru/code/CFAGO/CFAGO_seq/result/log/finetune_performance_{model_name}.csv"
    args.device = "cuda:0"

    args.dist_url = "tcp://127.0.0.1:3723"
    args.dim_feedforward = int(512)
    args.nheads = int(8)
    args.dropout = float(0.3)
    args.attention_layers = int(6)
    args.gamma_pos = int(0)
    args.gamma_neg = int(2)
    args.batch_size = int(32)
    args.lr = float(1e-4)

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

    # 准备数据集,esm2+prott5时 seq_2=True
    train_dataset, test_dataset, args.modesfeature_len = get_dataset(
        args.aspect, "9606", seq_2=3, onehot=False
    )
    args.encode_structure = [1024]

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        # (train_sampler is None)输出为true，因此会打乱数据
        shuffle=(train_sampler is None),
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

    # 定义损失函数
    loss = aslloss.AsymmetricLossOptimized(
        gamma_neg=args.gamma_neg,
        gamma_pos=args.gamma_pos,
        clip=args.loss_clip,
        disable_torch_grad_focal_loss=args.dtgfl,
        eps=args.eps,
    )

    # optimizer
    args.lr_mult = args.batch_size / 32
    # pre_model_param_dicts = [
    #     {"params": [p for n, p in pre_model.named_parameters() if p.requires_grad]},
    # ]
    torch.cuda.empty_cache()

    # 载入微调模型
    finetune_pre_model = torch.load(args.pretrained_model, map_location="cuda:0")
    # 创建预测模型
    predictor_model = build_predictor(finetune_pre_model, args)

    # if args.optim == 'AdamW':
    # 参数字典列表，存储预训练模型和fc_decoder层的参数
    predictor_model_param_dicts = [
        # 预训练模型的参数使用较低的学习率1e-5（因为已经训练好了，无需大幅度调整）
        {
            "params": [
                p
                for n, p in predictor_model.pre_model.named_parameters()
                if p.requires_grad
            ],
            "lr": 1e-5,
        },
        # fc_decoder层的参数使用默认学习率
        {
            "params": [
                p
                for n, p in predictor_model.fc_decoder.named_parameters()
                if p.requires_grad
            ]
        },
    ]

    # 优化器，使用AdamW算法
    predictor_model_optimizer = getattr(torch.optim, "AdamW")(
        predictor_model_param_dicts,
        lr=args.lr_mult * args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
    )
    # 学习率调度器 指定优化器，step_size=50，默认gamma=0.1，每隔step_size个周期就将每个参数组的学习率*gamma
    steplr = lr_scheduler.StepLR(predictor_model_optimizer, 50)
    patience = 10
    changed_lr = False

    # 每隔2500个epoch就把学习率乘0.01
    finetune(
        args,
        train_loader,
        predictor_model,
        loss,
        predictor_model_optimizer,
        steplr,
        args.epochs,
        args.device,
    )
    pd.to_pickle(
        predictor_model,
        args.finetune_model,
    )
    # 导入已微调好的模型
    # predictor_model = pd.read_pickle(
    #     "/home/Kioedru/code/CFAGO_seq/result/model/finetune_model_{args.aspect}.pkl"
    # ).to(args.device)
    perf = evaluate(test_loader, predictor_model, args.device)

    if not os.path.exists(args.performance_path):
        with open(args.performance_path, "w") as f:
            csv.writer(f).writerow(
                ["features", "aspect", "m-aupr", "M-aupr", "F1", "acc", "Fmax"]
            )

    with open(args.performance_path, "a") as f:
        csv.writer(f).writerow(
            [
                "ppi+feature+seq",
                args.aspect,
                perf["m-aupr"],
                perf["M-aupr"],
                perf["F1"],
                perf["acc"],
                perf["Fmax"],
            ]
        )


def create_log():
    # 定义csv文件的路径
    csv_path = "/home/kioedru/code/CFAGO/CFAGO_seq/result/log/finetune_log.csv"

    with open(csv_path, "w") as f:
        csv.writer(f).writerow(["Epoch", "Loss", "lr", "Time"])

    return csv_path


def finetune(
    args,
    data_loader,
    model,
    loss,
    optimizer,
    steplr,
    num_epochs,
    device,
):
    csv_path = create_log()

    net = model.to(device)
    net.train()
    print("training on", device)
    for epoch in range(num_epochs):
        if args.pretrain_update == 1:  # 不更新参数
            for p in model.pre_model.parameters():
                p.requires_grad = False
        if args.pretrain_update == 2:  # 更新后半部分参数
            if epoch >= (args.epochs / 2):
                for p in model.pre_model.parameters():
                    p.requires_grad = True
            else:
                for p in model.pre_model.parameters():
                    p.requires_grad = False
        if args.pretrain_update == 0:  # 更新全部参数
            for p in model.pre_model.parameters():
                p.requires_grad = True
        start = time.time()
        batch_count = 0
        train_l_sum = 0.0
        for protein_data, label in data_loader:

            protein_data[0] = protein_data[0].to(device)
            protein_data[1] = protein_data[1].to(device)
            protein_data[2] = protein_data[2].to(device)
            label = label.to(device)

            rec, output = net(protein_data)
            l = loss(rec, output, label)
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
                    optimizer.param_groups[1]["lr"],
                    time.time() - start,
                ]
            )


from evaluate_performance import evaluate_performance


@torch.no_grad()
def evaluate(test_loader, predictor_model, device):

    # switch to evaluate mode
    predictor_model.eval()
    all_output_sm = []
    all_label = []
    predictor_model = predictor_model.to(device)
    for proteins, label in test_loader:
        proteins[0] = proteins[0].to(device)
        proteins[1] = proteins[1].to(device)
        proteins[2] = proteins[2].to(device)
        label = label.to(device)

        # compute output
        rec, output = predictor_model(proteins)
        output_sm = torch.nn.functional.sigmoid(output)

        # collect output and label for metric calculation
        all_output_sm.append(output_sm.detach().cpu())
        all_label.append(label.detach().cpu())

    all_output_sm = torch.cat(all_output_sm, 0).numpy()
    all_label = torch.cat(all_label, 0).numpy()

    # calculate metrics
    perf = evaluate_performance(
        all_label, all_output_sm, (all_output_sm > 0.5).astype(int)
    )

    return perf


if __name__ == "__main__":
    main()
