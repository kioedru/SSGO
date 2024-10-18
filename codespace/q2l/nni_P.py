import random
import numpy as np
import torch

seed = 1329765522
if seed is not None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from nni.experiment import Experiment


search_space = {
    "lr": {"_type": "loguniform", "_value": [1e-5, 0.1]},
    "pre_lr": {"_type": "loguniform", "_value": [1e-5, 0.1]},
    # "seq_pre_lr": {"_type": "loguniform", "_value": [1e-5, 0.1]},
    # "pre_lr_model17": {"_type": "loguniform", "_value": [1e-5, 0.1]},
    # "seq_pre_lr_model17": {"_type": "loguniform", "_value": [1e-5, 0.1]},
    "dropout": {"_type": "uniform", "_value": [0.1, 0.5]},
    # "lr_model17": {"_type": "loguniform", "_value": [1e-5, 0.1]},
    # "lr_model39": {"_type": "loguniform", "_value": [1e-5, 0.1]},
}
experiment = Experiment("local")
# python -u /home/Kioedru/code/SSGO/codespace/q2l/finetune0.py --device cuda:0 --seed 1329765522 --seq_feature seq1024 --aspect C --num_class 35
# 配置 trial
experiment.config.trial_command = f"python -u /home/Kioedru/code/SSGO/codespace/q2l/finetune0.py --device cuda:0 --seed 1329765522 --seq_feature seq1024 --aspect P --num_class 45 --nni True"
experiment.config.trial_code_directory = "."

# 配置搜索空间
experiment.config.search_space = search_space

# 配置调优算法
experiment.config.tuner.name = "Random"
experiment.config.tuner.class_args["seed"] = 1329765522
# experiment.config.tuner.name = "TPE"
# experiment.config.tuner.class_args["optimize_mode"] = "maximize"
# 共尝试10组超参，并且每次并行地评估2组超参。
experiment.config.max_trial_number = 200
experiment.config.trial_concurrency = 2

experiment.run(8101)

# nohup python -u /home/Kioedru/code/SSGO/codespace/q2l/nni_P.py &
