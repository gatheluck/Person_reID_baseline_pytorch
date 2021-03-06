#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -N reid_res50_fgsm-l-1-IN_market
#$ -o /home/aaa10329ah/user/waseda/abci_log/reid_resnet50_fgsm-l-1-IN_market.o

source /etc/profile.d/modules.sh
module load cuda/9.0/9.0.176.4
export PATH="/home/aaa10329ah/anaconda3/bin:${PATH}"
source activate reid
cd /home/aaa10329ah/user/waseda/person-reid
# script

python run_all.py --gpu_ids 0,1,2,3 \
									--data_dir data/Market/pytorch/ \
									--train_all \
									--num_epochs 100 \
									--batchsize 32 \
									-a resnet50 \
									--bb_weight data/models/resnet50_fgsm-l-1-IN.pth \
									-l logs/resnet50_fgsm-l-1-IN_market