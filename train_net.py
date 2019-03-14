# -*- coding: utf-8 -*-

# from __future__ import print_function, division

import argparse
import torch
torch.backends.cudnn.benchmark=True
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
#from PIL import Image
import time
import os
import sys
from model import ft_net, ft_net_dense, PCB
from random_erasing import RandomErasing
import yaml
from shutil import copyfile
import logging
from tqdm import tqdm

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

model_names = ['resnet34', 'resnet50', 'resnet101']

def train(opt):
	data_dir = opt.data_dir
	log_dir = opt.log_dir 

	transform_train_list = [
					#transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
					transforms.Resize((256,128), interpolation=3),
					transforms.Pad(10),
					transforms.RandomCrop((256,128)),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
					]

	transform_val_list = [
					transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
					transforms.ToTensor(),
					transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
					]

	print(transform_train_list)
	data_transforms = {
		'train': transforms.Compose( transform_train_list ),
		'val': transforms.Compose(transform_val_list),
	}

	train_all = ''
	if opt.train_all: train_all = '_all'

	image_datasets = {}
	image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all), data_transforms['train'])
	image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])

	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
								shuffle=True, num_workers=opt.num_workers, pin_memory=True) # 8 workers may work faster
								for x in ['train', 'val']}
	dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

	opt.train_dataset_size = dataset_sizes['train']
	opt.val_dataset_size = dataset_sizes['val']
	print("train_dataset_size: ", opt.train_dataset_size)
	print("val_dataset_size: ", opt.val_dataset_size)

	opt.val_freq = int((opt.train_dataset_size/opt.batchsize) * 0.1)
	assert opt.val_freq > 0 

	class_names = image_datasets['train'].classes
	
	model = ft_net(len(class_names), opt.arch, opt.bb_weight, opt.droprate, opt.stride)

	ignored_params = list(map(id, model.model.fc.parameters() )) + list(map(id, model.classifier.parameters() ))
	base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
	optimizer_ft = optim.SGD([
							{'params': base_params, 'lr': 0.1*opt.lr},
							{'params': model.model.fc.parameters(), 'lr': opt.lr},
							{'params': model.classifier.parameters(), 'lr': opt.lr}
					], weight_decay=5e-4, momentum=0.9, nesterov=True)

	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

	dir_name = os.path.join(opt.log_dir)
	if not os.path.isdir(dir_name):
		os.mkdir(dir_name)
	#record every run
	copyfile('./train.py', dir_name+'/train.py')
	copyfile('./model.py', dir_name+'/model.py')

	# save opts
	with open('%s/opts.yaml'%dir_name,'w') as fp:
		yaml.dump(vars(opt), fp, default_flow_style=False)

	model = model.cuda()
	criterion = nn.CrossEntropyLoss()
	train_model(model, dataloaders, criterion, optimizer_ft, exp_lr_scheduler, opt.num_epochs, opt.loggers, opt)

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, loggers, opt):
	since = time.time() 

	num_itr = 1
	num_log = 1
	running_loss = 0.0
	running_acc  = 0.0
	ep_loss = 0.0
	ep_acc  = 0.0 
	best_val_acc = 0.0

	for epoch in range(1, num_epochs+1):
		print('Epoch {}/{}'.format(epoch, num_epochs))
		print('-' * 10)

		# training #################################
		for data_train in tqdm(dataloaders["train"]):
			inputs_train, labels_train = data_train

			train_batch_size, c, h, w = inputs_train.shape
			if train_batch_size < opt.batchsize: continue

			if opt.cuda:
				inputs_train = Variable(inputs_train.cuda().detach())
				labels_train = Variable(labels_train.cuda().detach())
			else:
				inputs_train, labels_train = Variable(inputs_train), Variable(labels_train)

			optimizer.zero_grad()
			outputs_train = model(inputs_train)

			_, preds_train = torch.max(outputs_train.data, 1)
			loss_train = criterion(outputs_train, labels_train)
			loss_train.backward()
			optimizer.step()

			running_loss += loss_train.item() * train_batch_size
			running_acc  += float(torch.sum(preds_train == labels_train.data))
			ep_loss += loss_train.item() * train_batch_size
			ep_acc  += float(torch.sum(preds_train == labels_train.data))

			# validation ################################
			if num_itr % opt.val_freq == 0:
				print("validation") 
				val_loss = 0.0
				val_acc  = 0.0

				for data_val in dataloaders["val"]:
					inputs_val, labels_val = data_val

					val_batch_size, c, h, w = inputs_val.shape
					if val_batch_size < opt.batchsize: continue

					if opt.cuda:
						inputs_val = Variable(inputs_val.cuda().detach())
						labels_val = Variable(labels_val.cuda().detach())
					else:
						inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)

					optimizer.zero_grad()
					with torch.no_grad():
						outputs_val = model(inputs_val)

					_, preds_val = torch.max(outputs_val.data, 1)
					loss_val = criterion(outputs_val, labels_val)

					val_loss += loss_val.item() * val_batch_size
					val_acc  += float(torch.sum(preds_val == labels_val.data))

				val_loss /= opt.val_dataset_size
				val_acc  /= opt.val_dataset_size

				running_loss /= opt.val_freq * train_batch_size
				running_acc  /= opt.val_freq * train_batch_size

				# logging
				loggers['loss_train'].set(num_log, running_loss)
				loggers['loss_val'].set(num_log, val_loss)
				loggers['acc_train'].set(num_log, running_acc)
				loggers['acc_val'].set(num_log, val_acc)

				# save best model
				if val_acc > best_val_acc: best_val_acc = val_acc
				save_network(model, 'best', opt)

				running_loss = 0.0
				running_acc  = 0.0
				
				num_log += 1

			# end of validation
			num_itr += 1
			#print("num_itr: ", num_itr)

		# end of one epoch
		ep_loss /= opt.train_dataset_size 
		ep_acc  /= opt.train_dataset_size
			
	print("Training complete")
	save_network(model, 'last', opt)



def save_network(network, epoch_label, opt):
	save_filename = 'weight_%s.pth'% epoch_label
	save_path = os.path.join(opt.log_dir, save_filename)
	torch.save(network.cpu().state_dict(), save_path)
	if torch.cuda.is_available(): network.cuda()