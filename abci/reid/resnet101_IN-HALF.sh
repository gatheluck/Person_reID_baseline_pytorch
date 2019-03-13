#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -N reid_resnet101_IN-HALF
#$ -o /home/gatheluck/user/waseda/abci_log/reid_resnet101_IN-HALF.o


source /etc/profile.d/modules.sh
module load cuda/9.0/9.0.176.4
export PATH="~/anaconda3/bin:${PATH}"
source activate 

cd /home/aaa10329ah/user/waseda
python run_all.py 			-a resnet101 			-l logs/reid_resnet101_IN-HALF 			--bb_weight data/models/resnet101_IN-HALF.pth 			--logger_dir logs/reid_resnet101_IN-HALF/logger_out 			--data_dir data/Market/pytorch/ 			--train_all 			--batch_size 32 			--num_epochs 100
