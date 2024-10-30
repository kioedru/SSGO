import os
import csv


def perf_write_to_csv(epoch_performance_path, epoch, perf, loss, time, lr):
    if not os.path.exists(epoch_performance_path):
        with open(epoch_performance_path, "w") as f:
            csv.writer(f).writerow(
                ["epoch", "loss", "time", "lr", "Fmax", "m-aupr", "M-aupr", "F1", "acc"]
            )

    with open(epoch_performance_path, "a") as f:
        csv.writer(f).writerow(
            [
                epoch,
                loss,
                time,
                lr,
                perf["Fmax"],
                perf["m-aupr"],
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
