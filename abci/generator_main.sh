models=(resnet50 resnet101)
train_types=(IN IN-FGSM IN-HALF IN-IN FGSM FGSM-IN FGSM-HALF FGSM-FGSM)
where=mlab  #(aist mlab)

if [ ${where} = aist ]; then
	venv=anaconda3
	logdir=~/user/waseda/abci_log
	projectdir=/home/aaa10329ah/user/waseda/person-reid
	gpu_ids=0,1,2,3
elif [ ${where} = mlab ]; then
	venv=miniconda3
	logdir=~/iccv2019/abci_log
	projectdir=~/iccv2019/person-reid
	gpu_ids=0
else
	echo 'Invalid' 1>&2
  exit 1
fi

project=reid #name of the projects
env_name=reid
suffix=main
mkdir -p ${project}
for model in ${models[@]}; do
	for train_type in ${train_types[@]}; do
		filename=${project}/${model}_${train_type}.sh

		name=${project}_${model}_${train_type}
		logpath=${logdir}/${name}.o

		if [ ${model} = resnet50 ]; then
			batch_size=24
		elif [ ${model} = resnet101 ]; then
			batch_size=24
		else
			echo 'Invalid' 1>&2
			exit 1
		fi

		echo -e "#!/bin/bash\n\n#$ -l rt_G.large=1\n#$ -l h_rt=24:00:00\n#$ -j y\n#$ -N ${name}\n#$ -o ${logpath}\n\n" > ${filename}
		echo -e "source /etc/profile.d/modules.sh\nmodule load cuda/9.0/9.0.176.4\nexport PATH=\"~/${venv}/bin:\${PATH}\"\nsource activate ${env_name}\n" >> ${filename}

		echo -e "cd ${projectdir}" >>  ${filename}
		echo -e "python run_all.py \
			-a ${model} \
			-l logs/${name} \
			--bb_weight data/models/${model}_${train_type}.pth \
			--logger_dir logs/${name}/logger_out \
			--data_dir data/Market/pytorch/ \
			--batchsize ${batch_size} \
			--num_workers 20 \
			--num_epochs 10" >> ${filename}
	done
done