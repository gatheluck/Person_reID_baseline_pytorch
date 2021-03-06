import os
import sys
import os.path
import argparse

import torch
from torchvision import models
import logger

model_names = ['resnet34', 'resnet50', 'resnet101']

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

class Options():
	def __init__(self):
		self.initialized = False

	def initialize(self, parser):
		parser = argparse.ArgumentParser(description='Run_All')
		parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
		parser.add_argument('-l', '--log_dir', required=True, type=str, help='log_dir')
		parser.add_argument('--logger_dir', required=True, type=str, help='logger directory')
		parser.add_argument('--data_dir',default='./data/Market/pytorch',type=str, help='training dir path')
		parser.add_argument('--train_all', action='store_true', help='use all training data' )
		parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
		parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
		parser.add_argument('--stride', default=2, type=int, help='stride')
		parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
		parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
		parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
		parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
		parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50' )
		parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )
		parser.add_argument('--num_epochs', default=30, type=int, help='number of training epochs')
		parser.add_argument('-a', '--arch', type=str, choices=model_names, required=True, help='name of architechure')
		parser.add_argument('--bb_weight', type=str, required=True, help='path to backbone weight')
		# parser.add_argument('--num_retry', type=int, default=1, help='number of retry')

		#parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
		parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
		# parser.add_argument('--test_dir',default='../Market/pytorch',type=str, help='./test_data')
		# parser.add_argument('-l', '--log_dir', required=True, type=str, help='save model path')
		# parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
		# parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
		# parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
		# parser.add_argument('--PCB', action='store_true', help='use PCB' )
		parser.add_argument('--multi', action='store_true', default=False, help='use multiple query' )
		# parser.add_argument('--fp16', action='store_true', help='use fp16.' )

		parser.add_argument('-r', '--result_suffix', default='', type=str, help='suffix of result file')

		self.initialized = True
		return parser

	def gather_options(self):
		if not self.initialized:
			parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
			parser = self.initialize(parser)

		self.parser = parser
		return parser.parse_args()

	def print_options(self, opt):
		message = ''
		message += '---------------------------- Options --------------------------\n'
		for k, v in sorted(vars(opt).items()):
			comment = ''
			default = self.parser.get_default(k)
			if v != default:
				comment = '\t[default: {}]'.format(str(default))
			message += '{:>15}: {:<25}{}\n'.format(str(k), str(v), comment)
		message += '---------------------------- End ------------------------------'
		print(message)

		os.makedirs(opt.log_dir, exist_ok=True)
		with open(os.path.join(opt.log_dir, 'options.txt'), 'wt') as f:
			command = ''
			for k, v in sorted(vars(opt).items()):
				command += '--{} {} '.format(k, str(v))
			command += '\n'
			f.write(command)
			f.write(message)
			f.write('\n')
	
	def parse(self):
		opt = self.gather_options()

		opt.test_dir = opt.data_dir
		opt.result_suffix = opt.log_dir.split('/')[-1]

		# logging
		if not os.path.exists(opt.logger_dir): os.makedirs(opt.logger_dir, exist_ok=True)
		loggers = {}
		loss_train_logger = logger.Logger(os.path.join(opt.logger_dir, 'loss_train.csv'), opt.num_epochs)
		loss_val_logger   = logger.Logger(os.path.join(opt.logger_dir, 'loss_val.csv'), opt.num_epochs)
		acc_train_logger = logger.Logger(os.path.join(opt.logger_dir, 'acc_train.csv'), opt.num_epochs)
		acc_val_logger   = logger.Logger(os.path.join(opt.logger_dir, 'acc_val.csv'), opt.num_epochs)
		loggers['loss_train'] = loss_train_logger
		loggers['loss_val']   = loss_val_logger
		loggers['acc_train'] = acc_train_logger
		loggers['acc_val']   = acc_val_logger
		opt.loggers = loggers

		self.opt = opt
		self.print_options(opt)
		return self.opt
		