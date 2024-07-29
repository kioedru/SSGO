import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

aspects = ["P", "F", "C"]
num_class = {"P": 45, "F": 38, "C": 35}
pretrain_updates = [2]
seeds = [132976111, 1329765519, 1329765522, 1329765525, 1329765529]
epochs = [100]
device = "cuda:0"


def run_finetune(aspect, epoch, pretrain_update, seed):
    print(
        f"finetune for {aspect} in epochs={epoch} seed={seed} pretrain_update={pretrain_update}"
    )
    command = f"nohup python -u /home/Kioedru/code/SSGO/codespace/finetune/transformer_bimamba_fusion_seq480/finetune.py --device {device}  --seed {seed}  --aspect {aspect} --num_class {num_class[aspect]} &"
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    return stderr


# 使用 ThreadPoolExecutor 并行运行三个 aspect 的任务
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = []
    for epoch in epochs:
        for pretrain_update in pretrain_updates:
            for seed in seeds:
                for aspect in aspects:
                    futures.append(
                        executor.submit(
                            run_finetune, aspect, epoch, pretrain_update, seed
                        )
                    )

    # 打印错误信息
    for future in as_completed(futures):
        stderr = future.result()
        print(stderr)

# nohup python -u /home/Kioedru/code/SSGO/codespace/finetune/transformer_bimamba_fusion_seq480/run_for_five.py &
