import os
import sys
import argparse

import torch
from torchvision import models

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

class Options():
	def __init__(self):
		self.initialized = False

	def initialize(self, parser):
		parser = argparse.ArgumentParser(description='Run_All')
		parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
		parser.add_argument('-l', '--log_dir', required=True, type=str, help='log_dir')
		parser.add_argument('--data_dir',default='../Market/pytorch',type=str, help='training dir path')
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

		self.opt = opt
		self.print_options(opt)
		return self.opt
		