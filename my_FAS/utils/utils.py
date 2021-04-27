from __future__ import division
import json
import math
import pandas as pd
import torch
import os
import sys
import shutil
from torchvision.utils import make_grid


def adjust_learning_rate(optimizer, epoch, init_param_lr, lr_epoch_1, lr_epoch_2):
	i = 0
	for param_group in optimizer.param_groups:
		init_lr = init_param_lr[i]
		i += 1
		if (epoch <= lr_epoch_1):
			param_group['lr'] = init_lr * 0.1 ** 0
		elif (epoch <= lr_epoch_2):
			param_group['lr'] = init_lr * 0.1 ** 1
		else:
			param_group['lr'] = init_lr * 0.1 ** 2


import matplotlib.pyplot as plt


def draw_far_frr_roc(frr_list, far_list, roc_auc, id=''):
	plt.switch_backend('agg')
	plt.rcParams['figure.figsize'] = (6.0, 6.0)
	plt.title('FAR_FRR_ROC')
	plt.plot(far_list, frr_list, 'b', label='AUC = %0.4f' % roc_auc)
	plt.legend(loc='upper right')
	plt.plot([0, 1], [1, 0], 'r--')
	plt.grid(ls='--')
	plt.ylabel('False Negative Rate')
	plt.xlabel('False Positive Rate')
	save_dir = './save_results/ROC/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	plt.savefig('./save_results/ROC/' + id + '_FAR_FRR_ROC.png')
	file = open('./save_results/ROC/' + id + '_FAR_FRR.txt', 'w')
	save_json = []
	dict = {}
	dict['FAR'] = far_list
	dict['FRR'] = frr_list
	save_json.append(dict)
	json.dump(save_json, file, indent=4)


def draw_roc(tpr_list, far_list, roc_auc, id=''):
	plt.switch_backend('agg')
	plt.rcParams['figure.figsize'] = (6.0, 6.0)
	plt.title('ROC')
	plt.plot(far_list, tpr_list, 'b', label='AUC = %0.4f' % roc_auc)
	plt.legend(loc='upper right')
	plt.plot([0, 1], [1, 0], 'r--')
	plt.plot([0, 1], [0, 1], 'g--')
	plt.grid(ls='--')
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	save_dir = './save_results/ROC/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	plt.savefig('./save_results/ROC/' + id + '_ROC.png')
	file = open('./save_results/ROC/' + id + '_FAR_TPR.txt', 'w')
	save_json = []
	dict = {}
	dict['FAR'] = far_list
	dict['TPR'] = tpr_list
	save_json.append(dict)
	json.dump(save_json, file, indent=4)


def sample_frames(flag, num_frames, dataset_name):
	'''
		from every video (frames) to sample num_frames to test
		return: the choosen frames' path and label
	'''
	# The process is a litter cumbersome, you can change to your way for convenience
	sub_title = ''
	if '_' in dataset_name:
		sub_title = dataset_name.split('_')[-1]

	root_path = '/DISK3/houjyuf/workspace/Adv_FAS_v2/data_label/' + dataset_name
	if (flag == 0):  # select the fake images of training set
		label_path = root_path + '/train_fake.json'

	elif (flag == 1):  # select the real images of training set
		label_path = root_path + '/train_real.json'

	elif (flag == 2):  # select the devel images
		label_path = root_path + '/devel_label.json'

	elif (flag == 3):  # select the test images
		label_path = root_path + '/test_label.json'

	elif(flag == 4):  # select the train images
		label_path = root_path + '/train_label.json'

	elif(flag == 50):  # select the train and val images
		label_path = [root_path + '/train_label.json', root_path + '/devel_label.json']
	elif (flag == 51):  # select the train and val images of fake
		label_path = [root_path + '/train_fake.json', root_path + '/devel_fake.json']
	elif (flag == 52):  # select the train and val images of real
		label_path = [root_path + '/train_real.json', root_path + '/devel_real.json']

	else:  # select all the real and fake images
		label_path = root_path + '/all_label.json'
	sample_data_pd = []
	if isinstance(label_path, list):
		assert len(label_path) == 2, 'length of label_path should be 2'
		print('opening:', label_path)
		f_json1 = open(label_path[0], 'r', encoding='utf8')
		json_dict1 = json.load(f_json1)
		f_json2 = open(label_path[1], 'r', encoding='utf8')
		json_dict2 = json.load(f_json2)
		json_dict = json_dict1+json_dict2
		# print(len(json_dict1),len(json_dict2),len(json_dict))
		res_json = json.dumps(json_dict)
		sample_data_pd = pd.read_json(res_json)

	else:
		print('opening', label_path)
		f_json = open(label_path)
		sample_data_pd = pd.read_json(f_json)
	return sample_data_pd

def sample_frames_self(flag, dataset_name, sub_flag=1):
	'''
		from every video (frames) to sample num_frames to test
		return: the choosen frames' path and label
	'''
	# The process is a litter cumbersome, you can change to your way for convenience

	root_path = '/DISK3/houjyuf/workspace/Adv_FAS_v2/data_label/' + dataset_name
	flag_dict = {
		0: root_path + '/train_fake.json', 1: root_path + '/train_real.json',
		2: root_path + '/devel_label.json',
		3: root_path + '/test_label.json',
		4: root_path + '/train_label.json',
		50: [root_path + '/train_label.json', root_path + '/devel_label.json'],
		51: [root_path + '/train_fake.json', root_path + '/devel_fake.json'],
		52: [root_path + '/train_real.json', root_path + '/devel_real.json'],
		100: root_path + '/all_label.json'
	}
	label_path = flag_dict[flag]

	if isinstance(label_path, list):
		assert len(label_path) == 2, 'length of label_path should be 2'
		print('opening:', label_path)
		f_json1 = open(label_path[0], 'r', encoding='utf8')
		json_dict1 = json.load(f_json1)
		f_json2 = open(label_path[1], 'r', encoding='utf8')
		json_dict2 = json.load(f_json2)
		json_dict = json_dict1+json_dict2
		# print(len(json_dict1),len(json_dict2),len(json_dict))
		res_json = json.dumps(json_dict)
		sample_data_pd = pd.read_json(res_json)
	else:
		print('opening', label_path)
		f_json = open(label_path)
		if(sub_flag == 1):
			sample_data_pd = pd.read_json(f_json)
		else:
			data_pd = pd.read_json(f_json)
			sample_data_pd = data_pd[data_pd.sub_label == sub_flag]

	return sample_data_pd

def sample_frames_multi(flag, dataset_name, sub_flag=0):
	'''
		from every video (frames) to sample num_frames to test
		return: the choosen frames' path and label
	'''
	# The process is a litter cumbersome, you can change to your way for convenience

	root_path = '/DISK3/houjyuf/workspace/Adv_FAS_v2/data_label/' + dataset_name
	flag_dict = {
		0: root_path + '/train_fake.json', 1: root_path + '/train_real.json',
		2: root_path + '/devel_label.json',
		3: root_path + '/test_label.json',
		4: root_path + '/train_label.json',
		50: [root_path + '/train_label.json', root_path + '/devel_label.json'],
		51: [root_path + '/train_fake.json', root_path + '/devel_fake.json'],
		52: [root_path + '/train_real.json', root_path + '/devel_real.json'],
		100: root_path + '/all_label.json'
	}
	label_path = flag_dict[flag]

	if isinstance(label_path, list):
		assert len(label_path) == 2, 'length of label_path should be 2'
		print('opening:', label_path)
		f_json1 = open(label_path[0], 'r', encoding='utf8')
		json_dict1 = json.load(f_json1)
		f_json2 = open(label_path[1], 'r', encoding='utf8')
		json_dict2 = json.load(f_json2)
		json_dict = json_dict1+json_dict2
		# print(len(json_dict1),len(json_dict2),len(json_dict))
		res_json = json.dumps(json_dict)
		sample_data_pd = pd.read_json(res_json)
	else:
		print('opening', label_path)
		f_json = open(label_path)

		if (sub_flag == 0):
			sample_data_pd = pd.read_json(f_json)
		else:
			data_pd = pd.read_json(f_json)
			sample_data_pd = data_pd[data_pd.sub_label == sub_flag]

	return sample_data_pd

class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res


def mkdirs(checkpoint_path, best_model_path, logs):
	if not os.path.exists(checkpoint_path):
		os.makedirs(checkpoint_path)
	if not os.path.exists(best_model_path):
		os.makedirs(best_model_path)
	if not os.path.exists(logs):
		os.mkdir(logs)


def time_to_str(t, mode='min'):
	if mode == 'min':
		t = int(t) / 60
		hr = t // 60
		min = t % 60
		return '%2d hr %02d min' % (hr, min)
	elif mode == 'sec':
		t = int(t)
		min = t // 60
		sec = t % 60
		return '%2d min %02d sec' % (min, sec)
	else:
		raise NotImplementedError


class Logger(object):
	def __init__(self):
		self.terminal = sys.stdout
		self.file = None

	def open(self, file, mode=None):
		if mode is None:
			mode = 'w'
		self.file = open(file, mode)

	def write(self, message, is_terminal=1, is_file=1):
		if '\r' in message:
			is_file = 0
		if is_terminal == 1:
			self.terminal.write(message)
			self.terminal.flush()
		if is_file == 1:
			self.file.write(message)
			self.file.flush()

	def flush(self):
		# this flush method is needed for python 3 compatibility.
		# this handles the flush command by doing nothing.
		# you might want to specify some extra behavior here.
		pass


def save_checkpoint(save_list, is_best, model, gpus, checkpoint_path, best_model_path, filename='_checkpoint.pth.tar'):
	epoch = save_list[0]
	valid_args = save_list[1]
	best_model_HTER = round(save_list[2], 5)
	best_model_ACC = save_list[3]
	best_model_ACER = save_list[4]
	threshold = save_list[5]
	if (len(gpus) > 1):
		old_state_dict = model.state_dict()
		from collections import OrderedDict
		new_state_dict = OrderedDict()
		for k, v in old_state_dict.items():
			flag = k.find('.module.')
			if (flag != -1):
				k = k.replace('.module.', '.')
			new_state_dict[k] = v
		state = {
			"epoch": epoch,
			"state_dict": new_state_dict,
			"valid_arg": valid_args,
			"best_model_EER": best_model_HTER,
			"best_model_ACER": best_model_ACER,
			"best_model_ACC": best_model_ACC,
			"threshold": threshold
		}
	else:
		state = {
			"epoch": epoch,
			"state_dict": model.state_dict(),
			"valid_arg": valid_args,
			"best_model_EER": best_model_HTER,
			"best_model_ACER": best_model_ACER,
			"best_model_ACC": best_model_ACC,
			"threshold": threshold
		}
	filepath = checkpoint_path + filename
	torch.save(state, filepath)
	# just save best model
	if is_best == True:
		shutil.copy(filepath, best_model_path + 'model_best_' + str(best_model_HTER) + '_' + str(epoch) + '.pth.tar')


def zero_param_grad(params):
	for p in params:
		if p.grad is not None:
			p.grad.zero_()

class GridMaker:
	def __init__(self):
		pass

	def __call__(self, images, cues):
		b, c, h, w = images.shape
		images_min = images.view(b, -1).min(axis=1)[0][:, None]
		images_max = images.view(b, -1).max(axis=1)[0][:, None]
		images = (images.view(b, -1) - images_min) / (images_max - images_min)
		images = images.reshape(b, c, h, w)

		b, c, h, w = cues.shape
		cues_min = cues.view(b, -1).min(axis=1)[0][:, None]
		cues_max = cues.view(b, -1).max(axis=1)[0][:, None]
		cues = (cues.view(b, -1) - cues_min) / (cues_max - cues_min)
		cues = cues.reshape(b, c, h, w)

		return make_grid(images), make_grid(cues)

if __name__ == "__main__":
	# sample_frames_self(flag=0, dataset_name='Replay_rosplgf_3c', sub_flag=2)
	sample_frames_multi(flag=0, dataset_name='CASIA_ros', sub_flag=1)
