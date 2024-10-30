环境：python3.8  pytorch 1.12.0



This is the repository for CFAGO, a protein function prediction method based on cross-fusion of network structure and attributes via attention mechanism 


annotation_preprocess.py is used to prepocess Gene Ontology annotation file in terms of STRING PPI network data. Run python annotation_preprocess.py -h for options and use instructions.

network_data_preprocess.py is used to prepocess STRING PPI network data to weighted adjacency matrix. Run python network_data_preprocess.py -h for options and use instructions.


attribute_data_preprocess.py is used to prepocess uniprot protein domain (pfam) and subcellular location data to binary vectors. Run python attribute_data_preprocess.py -h for options and use instructions.

self_supervised_leaning.py is used to pre-train CFAGO in terms of reconstruct protein features. Run python self_supervised_leaning.py -h for options and use instructions.

CFAGO.py is used to fine-tune CFAGO and predict functions for testing proteins. Run python CFAGO.py -h for options and use instructions.


Uses pytorch 1.12.0.

---------- Sample Use Case ----------

Let's say you want to conduct experiments on Dataset/human.

Here is what you need to run:

With a normal GPU node, in CFAGO directory:

Step 1:

Preprocess STRING PPI file, annotation file and uniprot file.

python annotation_preprocess.py -data_path Dataset(自己定一个文件路径) -af goa_human.gaf -pf 9606.protein.info.v11.5.txt -ppif 9606.protein.links.detailed.v11.5.txt -org human -stl 41

python network_data_preprocess.py -data_path Dataset -snf 9606.protein.links.detailed.v11.5.txt -org human

python attribute_data_preprocess.py -data_path Dataset -pf 9606.protein.info.v11.5.txt -ppif 9606.protein.links.detailed.v11.5.txt -org human -uniprot uniprot-filtered-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[96--.tab

Step 2:

Pre-train CFAGO by self-supervised learning:

python self_supervised_leaning.py --org human --dataset_dir Dataset/human --output human_result --dist-url tcp://127.0.0.1:3723 --seed 1329765522 --dim_feedforward 512 --nheads 8 --dropout 0.1 --attention_layers 6 --batch-size 32 --activation gelu --epochs 5000 --lr 1e-5

Step 3:
#
fine-tuning CFAGO with annotations as labels, and output predictions for test proteins in terms of one GO branch: P for biological process ontology, F for moleculer function ontology, C for cellular component  ontology.
#最终预测
#--dist-url tcp://127.0.0.1:3723 分布式训练用，跑两个进程，改一下端口号就可以
python CFAGO.py --org human --dataset_dir Dataset/human --output human_result --aspect P --num_class 45 --dist-url tcp://127.0.0.1:3723 --seed 1329765522 --dim_feedforward 512 --nheads 8 --dropout 0.3 --attention_layers 6 --gamma_pos 0 --gamma_neg 2 --batch-size 32 --activation gelu --lr 1e-4 --pretrained_model human_result/human_attention_layers_6_lr_1e-05_seed_1329765522_activation_gelu_model.pkl

The results file that this step produces will be found at this path: ./human_result/human_attention_layers_6_aspect_P_fintune_seed_1329765522_act_gelu.csv. This file contains the five evaluation matrices' values on the Biological Process Ontology.






# aslloss.py 损失函数 训练过程的 多标签模型的损失函数

# encoder.py 特征编码 和 解码

# get_dataset.py 数据集读取   训练模型和预测 

# logger.py 保持日志，不用管

# multihead_attention.py 多头注意力 是论文图1d， 里面的注意力部分

# predictor_module.py 多标签模型 

# self_supervised_leaning.py 预训练模型

# validation.py 最终评估指标的代码
