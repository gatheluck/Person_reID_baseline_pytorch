#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -N reid_prepare_dataset
#$ -o /home/aaa10329ah/user/waseda/abci_log/reid_prepare_dataset.o

source /etc/profile.d/modules.sh
module load cuda/9.0/9.0.176.4
export PATH="/home/aaa10329ah/anaconda3/bin:${PATH}"
source activate reid
cd /home/aaa10329ah/user/waseda/person-reid
# script

python prepare.py