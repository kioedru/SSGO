import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
from loguru import logger
import sys
import os


from CLIProt.utils.get_dataset import get_dataset
from CLIProt.utils.evaluate_performance import evaluate_performance
from CLIProt.utils.summary import perf_write_to_csv
from CLIProt.cliprot import CLIProt


# ===================== 参数设置 =====================
def setting_args():
    parser = argparse.ArgumentParser(description="CLIProt")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        choices=["cuda:0", "cuda:1", "cpu"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/home/Kioedru/code/SSGO/CLIProt/best_model.pth",
        help="Path to save the best model",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="/home/Kioedru/code/SSGO/CLIProt/train.log",
        help="Path to save the log file",
    )

    parser.add_argument(
        "--step_size",
        type=int,
        default=50,
        help="Step size for learning rate scheduler",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.1, help="Learning rate decay factor"
    )
    parser.add_argument(
        "--aspect",
        type=str,
        default="P",
        choices=["P", "F", "C"],
        help="Aspect of GO terms to predict: 'P' for Biological Process, 'F' for Molecular Function, 'C' for Cellular Component.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization",
    )
    args = parser.parse_args()
    return args


# ===================== 随机种子设置 =====================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


# ===================== 日志配置 =====================
def set_logger(log_path):
    logger.remove()  # 移除默认的logger
    logger.add(
        log_path, format="{time} | {level} | {message}", level="INFO", encoding="utf-8"
    )
    logger.add(sys.stdout, format="{time} | {level} | {message}", level="INFO")


# ===================== 数据加载 =====================
def get_dataloader(aspect, batch_size, workers):
    train_dataset, test_dataset = get_dataset(aspect, "9606")
    # dataloders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        sampler=None,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        sampler=None,
        pin_memory=True,
    )

    return train_loader, test_loader


def get_go_embed(aspect):
    aspcet2namespace = {"P": "bpo", "F": "mfo", "C": "cco"}
    data_path = "/home/Kioedru/code/CLIProt/data_cp"
    terms_file_name = f"{aspcet2namespace[aspect]}_terms_embeddings.pkl"
    terms = pd.read_pickle(os.path.join(data_path, terms_file_name))
    embeddings = np.concatenate(
        [np.array(embedding, ndmin=2) for embedding in terms.embeddings.values]
    )  # F:[38,256]
    terms_embedding = torch.from_numpy(embeddings)
    return terms_embedding


# ===================== 训练和验证函数 =====================
def Encoder(protein_features, Seq_Encoder, PPI_Feature_Encoder):
    ppi_feature_src = protein_features  # 2,32,512
    seq_src = protein_features[2].unsqueeze(0)  # 1,32,512
    _, hs_ppi_feature = PPI_Feature_Encoder(ppi_feature_src)  # 2,32,512
    _, hs_seq = Seq_Encoder(seq_src)
    hs = torch.cat([hs_ppi_feature, hs_seq], dim=0)
    hs = torch.einsum("LBD->BLD", hs)  # 32,3,512
    return hs


def train_one_epoch(model, loader, optimizer, device, Seq_Encoder, PPI_Feature_Encoder):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        inputs = [tensor.to(device) for tensor in batch["protein_features"]]
        targets = batch["labels"].to(device)
        optimizer.zero_grad()
        align_features = Encoder(inputs, Seq_Encoder, PPI_Feature_Encoder)
        logits = model(align_features)
        loss = model.compute_loss(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, device, Seq_Encoder, PPI_Feature_Encoder):
    model.eval()
    all_outputs_sm = []
    all_labels = []

    for batch in tqdm(loader, desc="Evaluate", leave=False):
        inputs = [tensor.to(device) for tensor in batch["protein_features"]]
        targets = batch["labels"].to(device)
        align_features = Encoder(inputs, Seq_Encoder, PPI_Feature_Encoder)
        logits = model(align_features)
        outputs_sm = torch.nn.functional.sigmoid(logits)
        all_outputs_sm.append(outputs_sm.detach().cpu())
        all_labels.append(targets.detach().cpu())

    all_outputs_sm = torch.cat(all_outputs_sm, 0).numpy()
    all_labels = torch.cat(all_labels, 0).numpy()
    metrics = evaluate_performance(
        all_labels, all_outputs_sm, (all_outputs_sm > 0.5).astype(int)
    )
    # metrics={"M-aupr": M-aupr, "M-aupr-labels": M-aupr-labels, "m-aupr": m-aupr, "acc": acc, "F1": F1, "Fmax": Fmax}

    return metrics


def main():
    args = setting_args()
    set_seed(args.seed)
    set_logger(args.log_path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloader(
        args.aspect, args.batch_size, args.workers
    )
    go_feature = get_go_embed(args.aspect).to(device)
    model = CLIProt(go_feature, protein_dim=512, go_dim=256, latent_dim=512)
    model = model.to(device)

    Seq_Encoder = torch.load(
        "/home/Kioedru/code/SSGO/codespace/pretrain/one_feature_only/9606/transformer_seq1024_only.pkl",
        map_location=args.device,
    )
    # PPI_Feature_Encoder = torch.load(
    #     "/home/Kioedru/code/SSGO/codespace/pretrain/transformer/9606/transformer.pkl",
    #     map_location=args.device,
    # )
    sys.path.append("/home/Kioedru/code/SSGO/CFAGO-code")
    model_path = "/home/Kioedru/code/SSGO/human_attention_layers_6_lr_1e-05_seed_1329765522_activation_gelu_model.pkl"
    PPI_Feature_Encoder = torch.load(model_path)
    # ===================== 损失函数 & 优化器 =====================

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    # 按轮数衰减
    # scheduler = optim.lr_scheduler.StepLR(
    #     optimizer, step_size=args.step_size, gamma=args.gamma
    # )
    # # 余弦退火
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=args.epochs, eta_min=0
    # )

    # 指数衰减
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # ===================== 训练主循环 =====================
    best_Fmax = 0  # 保存最佳模型时的最高验证准确率
    logger.info(f"Starting training on {device}...")

    for epoch in range(args.epochs):
        logger.info(f"Epoch [{epoch + 1}/{args.epochs}]")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, Seq_Encoder, PPI_Feature_Encoder
        )
        metrics = evaluate(model, test_loader, device, Seq_Encoder, PPI_Feature_Encoder)

        logger.info(f"Train Loss: {train_loss:.3f}")
        filtered_metrics = {k: v for k, v in metrics.items() if k != "M-aupr-labels"}
        logger.info(
            f"Test metrics: {filtered_metrics}, Learning Rate: {scheduler.get_last_lr()[0]}"
        )

        # 保存最佳模型
        if metrics["Fmax"] > best_Fmax:
            best_Fmax = metrics["Fmax"]
            torch.save(model.state_dict(), args.save_path)
            logger.info(f"Best model saved to {args.save_path}")

        # 更新学习率
        scheduler.step()

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
