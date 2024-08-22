search_space = {
    "lr": {"_type": "loguniform", "_value": [1e-5, 0.1]},
    "pre_lr": {"_type": "loguniform", "_value": [1e-5, 0.1]},
    "seq_pre_lr": {"_type": "loguniform", "_value": [1e-5, 0.1]},
    "dropout": {"_type": "uniform", "_value": [0.1, 0.5]},
}

from nni.experiment import Experiment

experiment = Experiment("local")

# 配置 trial
experiment.config.trial_command = f"python -u /home/Kioedru/code/SSGO/codespace/finetune/2_1_simple/finetune.py --fusion transformer --seq_feature seq1024 --aspect C --num_class 35 --device cuda:1 --nni True"
experiment.config.trial_code_directory = "."

# 配置搜索空间
experiment.config.search_space = search_space

# 配置调优算法
experiment.config.tuner.name = "Random"
experiment.config.tuner.class_args["seed"] = 100

# 共尝试10组超参，并且每次并行地评估2组超参。
experiment.config.max_trial_number = 100
experiment.config.trial_concurrency = 4

experiment.run(8091)
# nohup python -u /home/Kioedru/code/SSGO/codespace/finetune/2_1_simple/nni_C.py &
# 461963