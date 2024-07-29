import subprocess

aspects = ["P", "F", "C"]
num_class = {"P": 45, "F": 38, "C": 35}
# pretrain_updates = [2, 1, 0]
pretrain_updates = [2]

seeds = [1329765519]
# seeds = [132976111, 1329765519, 1329765522, 1329765525, 1329765529]
epochs = [100]
device = "cuda:1"
backbones = ["transformer"]
# backbones = ["transformer", "bimamba"]
for epoch in epochs:
    for pretrain_update in pretrain_updates:
        for backbone in backbones:
            for aspect in aspects:
                for seed in seeds:
                    print(
                        f"finetune for {aspect} in epochs={epoch} seed={seed} pretrain_update={pretrain_update}"
                    )

                    command = f"nohup python -u /home/Kioedru/code/SSGO/codespace/finetune/one_feature_only/finetune.py --feature_name seq1024 --backbone {backbone} --device {device} --pretrain-update {pretrain_update} --epochs {epoch} --seed {seed}  --aspect {aspect} --num_class {num_class[aspect]} &"
                    process = subprocess.Popen(
                        command,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    stdout, stderr = process.communicate()
                    print(stderr)
# nohup python -u /home/Kioedru/code/SSGO/codespace/finetune/one_feature_only/run_for_five_seq1024.py &
