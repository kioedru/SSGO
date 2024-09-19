# {
#     "lr": 0.00025913394093475507,
#     "pre_lr": 0.00249355105687278,
#     "seq_pre_lr": 0.01970014885597919,
#     "pre_lr_model17": 0.010308145533619225,
#     "seq_pre_lr_model17": 0.00021205159495776921,
#     "dropout": 0.12404460735928378,
#     "lr_model17": 0.005369189892977804,
#     "lr_model39": 0.0032432752939293118
# }
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
    "seq_pre_lr": {"_type": "loguniform", "_value": [1e-5, 0.1]},
    # "pre_lr_model17": {"_type": "loguniform", "_value": [1e-5, 0.1]},
    # "seq_pre_lr_model17": {"_type": "loguniform", "_value": [1e-5, 0.1]},
    "dropout": {"_type": "uniform", "_value": [0.1, 0.5]},
    # "lr_model17": {"_type": "loguniform", "_value": [1e-5, 0.1]},
    # "lr_model39": {"_type": "loguniform", "_value": [1e-5, 0.1]},
}
# search_space = {
#     "lr": {"_type": "choice", "_value": [0.021873915093390775]},
#     "pre_lr": {"_type": "choice", "_value": [0.0024334145914109615]},
#     "seq_pre_lr": {"_type": "choice", "_value": [0.00014303850645524388]},
#     "pre_lr_model17": {"_type": "choice", "_value": [0.000014852729879134244]},
#     "seq_pre_lr_model17": {"_type": "choice", "_value": [0.07845444968715394]},
#     "dropout": {"_type": "choice", "_value": [0.3385886816258754]},
#     "lr_model17": {"_type": "choice", "_value": [0.01448947526010299]},
#     "lr_model39": {"_type": "choice", "_value": [0.04378824357208102]},
# }
experiment = Experiment("local")

# 配置 trial
experiment.config.trial_command = f"python -u /home/Kioedru/code/SSGO/codespace/finetune/2_1_final/finetune82.py --model_num 82 --seed 1329765522 --seq_feature seq1024 --aspect C --num_class 35 --device cuda:0 --nni True"
experiment.config.trial_code_directory = "."

# 配置搜索空间
experiment.config.search_space = search_space

# 配置调优算法
experiment.config.tuner.name = "Random"
experiment.config.tuner.class_args["seed"] = 100
# experiment.config.tuner.name = "TPE"
# experiment.config.tuner.class_args["optimize_mode"] = "maximize"
# 共尝试10组超参，并且每次并行地评估2组超参。
experiment.config.max_trial_number = 200
experiment.config.trial_concurrency = 4

experiment.run(8102)

# nohup python -u /home/Kioedru/code/SSGO/codespace/finetune/2_1_final/nni_P.py &
