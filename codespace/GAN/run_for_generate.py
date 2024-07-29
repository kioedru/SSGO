import subprocess
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_command(command):
    process = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.stdout, process.stderr
    return command, stdout, stderr


# aspects = ["P", "F", "C"]
aspects = ["C"]
num_class = {"P": 45, "F": 38, "C": 35}
organism_num = "9606"
FeatureName = "esm2-480"
device = "cuda:1"
commands = []

for aspect in aspects:

    terms = pd.read_pickle(
        f"/home/Kioedru/code/SSGO/data/finetune/{organism_num}/terms_{aspect}.pkl"
    )["terms"].tolist()

    for GOTermID in terms:
        command = f"python -u /home/Kioedru/code/SSGO/codespace/GAN/Generating_Synthetic_Positive_Samples_FFPred-GAN.py --device {device} --aspect {aspect} --organism_num {organism_num} --FeatureName {FeatureName} --GOTermID {GOTermID} > /home/Kioedru/code/SSGO/data/synthetic/{FeatureName}/{organism_num}/{aspect}/{GOTermID}/generating.log 2>&1"
        commands.append(command)

# 使用ThreadPoolExecutor来并行运行命令，每次并行运行3个命令
with ThreadPoolExecutor(max_workers=2) as executor:
    future_to_command = {executor.submit(run_command, cmd): cmd for cmd in commands}
    for future in as_completed(future_to_command):
        cmd = future_to_command[future]
        try:
            command, stdout, stderr = future.result()
            if stderr:
                print(f"Error for command {command}: {stderr.decode()}")
            else:
                print(f"Completed command {command}: {stdout.decode()}")
        except Exception as e:
            print(f"Exception for command {cmd}: {e}")

# nohup python -u /home/Kioedru/code/SSGO/codespace/GAN/run_for_generate.py> /home/Kioedru/code/SSGO/codespace/GAN/esm2-480_C.log 2>&1 &
