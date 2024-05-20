import subprocess

aspects = ["P", "F", "C"]
num_class = {"P": 45, "F": 38, "C": 35}
seeds = [1329765522, 132976111, 1329765525, 1329765529, 1329765519]
for aspect in aspects:
    for seed in seeds:
        print(f"finetune for {aspect} in seed={seed} ")

        command = f"nohup python /home/kioedru/code/SSGO/codespace/finetune/mamba3_seq480/finetune.py --seed {seed}  --aspect {aspect} --num_class {num_class[aspect]} &"
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        print(stderr)