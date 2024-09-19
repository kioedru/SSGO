import random
import numpy as np
import torch

seed = 1329765519
if seed is not None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
search_space = {
    "lr": {"_type": "loguniform", "_value": [1e-5, 0.1]},
    "pre_lr": {"_type": "loguniform", "_value": [1e-5, 0.1]},
    "seq_pre_lr": {"_type": "loguniform", "_value": [1e-5, 0.1]},
    "dropout": {"_type": "uniform", "_value": [0.1, 0.5]},
}
# search_space = {
#     "lr": {"_type": "choice", "_value": [0.002834232201390033]},
#     "pre_lr": {"_type": "choice", "_value": [0.002842738151730068]},
#     "seq_pre_lr": {"_type": "choice", "_value": [0.005539218325185251]},
#     "dropout": {"_type": "choice", "_value": [0.4625400261289042]},
# }
from nni.experiment import Experiment

experiment = Experiment("local")

# 配置 trial
experiment.config.trial_command = f"python -u /home/Kioedru/code/SSGO/codespace/finetune/2_1_inner_frozen/finetune_same.py --model_num 62 --seed {seed} --seq_feature seq1024 --aspect P --num_class 45 --device cuda:1 --nni True"
experiment.config.trial_code_directory = "."

# 配置搜索空间
experiment.config.search_space = search_space

# 配置调优算法
# experiment.config.tuner.name = "Random"
# experiment.config.tuner.class_args["seed"] = 100
experiment.config.tuner.name = "Random"
experiment.config.tuner.class_args["seed"] = 100
# 共尝试10组超参，并且每次并行地评估2组超参。
experiment.config.max_trial_number = 200
experiment.config.trial_concurrency = 1

experiment.run(8085)

# nohup python -u /home/Kioedru/code/SSGO/codespace/finetune/2_1_inner_frozen/nni_P.py &
