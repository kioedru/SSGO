search_space = {
    "lr": {"_type": "loguniform", "_value": [1e-5, 0.1]},
    "pre_lr": {"_type": "loguniform", "_value": [1e-5, 0.1]},
    "seq_pre_lr": {"_type": "loguniform", "_value": [1e-5, 0.1]},
    "dropout": {"_type": "uniform", "_value": [0.1, 0.5]},
}

from nni.experiment import Experiment

experiment = Experiment("local")

# 配置 trial
experiment.config.trial_command = f"python -u /home/Kioedru/code/SSGO/codespace/finetune/2_1_inner_fusion2/finetune.py --model_num 42 --seed 1329765522 --seq_feature seq1024 --aspect F --num_class 38 --device cuda:0 --nni True"
experiment.config.trial_code_directory = "."

# 配置搜索空间
experiment.config.search_space = search_space

# 配置调优算法
# experiment.config.tuner.name = "Random"
# experiment.config.tuner.class_args["seed"] = 100
experiment.config.tuner.name = "TPE"
experiment.config.tuner.class_args["optimize_mode"] = "maximize"

# 共尝试10组超参，并且每次并行地评估2组超参。
experiment.config.max_trial_number = 100
experiment.config.trial_concurrency = 2

experiment.run(8094)
# nohup python -u /home/Kioedru/code/SSGO/codespace/finetune/2_1_inner_fusion2/nni_F.py &