import os
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.dataset import YunpeiDataset, HoujyufDataset, HoujyufDataset_v2, DepthDataset, RsgbDataset, grgDataset, ContrasiveDataset
from utils.utils import sample_frames, sample_frames_self, sample_frames_multi

# DepthDataset use depth map for SSDG dataset to optimize
# three source domains, one target domain
def get_dataset(src1_data, src1_train_num_frames, src2_data, src2_train_num_frames, src3_data, src3_train_num_frames,
				tgt_data, tgt_test_num_frames, batch_size):
	print('Load Source Data')
	print('Source Data: ', src1_data)
	src1_train_data_fake = sample_frames(flag=0, num_frames=src1_train_num_frames, dataset_name=src1_data)
	src1_train_data_real = sample_frames(flag=1, num_frames=src1_train_num_frames, dataset_name=src1_data)
	print('Source Data: ', src2_data)
	src2_train_data_fake = sample_frames(flag=0, num_frames=src2_train_num_frames, dataset_name=src2_data)
	src2_train_data_real = sample_frames(flag=1, num_frames=src2_train_num_frames, dataset_name=src2_data)
	print('Source Data: ', src3_data)
	src3_train_data_fake = sample_frames(flag=0, num_frames=src3_train_num_frames, dataset_name=src3_data)
	src3_train_data_real = sample_frames(flag=1, num_frames=src3_train_num_frames, dataset_name=src3_data)

	print('Load Target Data')
	print('Target Data: ', tgt_data)
	tgt_test_data = sample_frames(flag=2, num_frames=tgt_test_num_frames, dataset_name=tgt_data)

	src1_train_dataloader_fake = DataLoader(YunpeiDataset(src1_train_data_fake, train=True),
											batch_size=batch_size, shuffle=True)
	src1_train_dataloader_real = DataLoader(YunpeiDataset(src1_train_data_real, train=True),
											batch_size=batch_size, shuffle=True)
	src2_train_dataloader_fake = DataLoader(YunpeiDataset(src2_train_data_fake, train=True),
											batch_size=batch_size, shuffle=True)
	src2_train_dataloader_real = DataLoader(YunpeiDataset(src2_train_data_real, train=True),
											batch_size=batch_size, shuffle=True)
	src3_train_dataloader_fake = DataLoader(YunpeiDataset(src3_train_data_fake, train=True),
											batch_size=batch_size, shuffle=True)
	src3_train_dataloader_real = DataLoader(YunpeiDataset(src3_train_data_real, train=True),
											batch_size=batch_size, shuffle=True)
	tgt_dataloader = DataLoader(YunpeiDataset(tgt_test_data, train=False), batch_size=batch_size, shuffle=False)
	return src1_train_dataloader_fake, src1_train_dataloader_real, \
		   src2_train_dataloader_fake, src2_train_dataloader_real, \
		   src3_train_dataloader_fake, src3_train_dataloader_real, \
		   tgt_dataloader

def get_mask_dataset(src1_data, src1_train_num_frames, tgt_data, tgt_test_num_frames, batch_size):
	if 'Oulu' not in src1_data and 'Replay' not in src1_data:
		tr_fake_flag, tr_real_flag, src_val, src_test, tgt_test = 51, 52, 3, 3, 3
	else:
		tr_fake_flag, tr_real_flag, src_val, src_test, tgt_test = 0, 1, 2, 3, 3
	print(src1_data, tr_fake_flag, tr_real_flag, src_val, src_test, tgt_test)
	print('Load Source Data')
	print('Source Data: ', src1_data)
	src1_train_data_fake = sample_frames(flag=tr_fake_flag, num_frames=src1_train_num_frames, dataset_name=src1_data)
	src1_train_data_real = sample_frames(flag=tr_real_flag, num_frames=src1_train_num_frames, dataset_name=src1_data)
	print('src val Data: ', src1_data)
	src1_val_data = sample_frames(flag=src_val, num_frames=tgt_test_num_frames, dataset_name=src1_data)
	print('src test Data: ', src1_data)
	src1_test_data = sample_frames(flag=src_test, num_frames=tgt_test_num_frames, dataset_name=src1_data)

	print('Load Target Data')
	print('Target Data: ', tgt_data)
	tgt_test_data = sample_frames(flag=tgt_test, num_frames=tgt_test_num_frames, dataset_name=tgt_data)

	src1_train_dataloader_fake = DataLoader(HoujyufDataset(src1_train_data_fake, train=True),
											batch_size=batch_size, shuffle=True)
	src1_train_dataloader_real = DataLoader(HoujyufDataset(src1_train_data_real, train=True),
											batch_size=batch_size, shuffle=True)
	src1_val_dataloader = DataLoader(HoujyufDataset(src1_val_data, train=False), batch_size=batch_size, shuffle=False)
	src1_test_dataloader = DataLoader(HoujyufDataset(src1_test_data, train=False), batch_size=batch_size, shuffle=False)

	tgt_test_dataloader = DataLoader(HoujyufDataset(tgt_test_data, train=False), batch_size=batch_size,
									 shuffle=False)
	return src1_train_dataloader_fake, src1_train_dataloader_real, \
		   src1_val_dataloader, src1_test_dataloader, tgt_test_dataloader

def get_new_dataset(src1_data, src1_train_num_frames, tgt_data, tgt_test_num_frames, batch_size):
	if 'Oulu' not in src1_data and 'Replay' not in src1_data:
		tr_fake_flag, tr_real_flag, src_val, src_test, tgt_test = 51, 52, 3, 3, 3
	else:
		tr_fake_flag, tr_real_flag, src_val, src_test, tgt_test = 0, 1, 2, 3, 3
	print(src1_data, tr_fake_flag, tr_real_flag, src_val, src_test, tgt_test)
	print('Load Source Data')
	print('Source Data: ', src1_data)
	src1_train_data_fake = sample_frames(flag=tr_fake_flag, num_frames=src1_train_num_frames, dataset_name=src1_data)
	src1_train_data_real = sample_frames(flag=tr_real_flag, num_frames=src1_train_num_frames, dataset_name=src1_data)
	print('src val Data: ', src1_data)
	src1_val_data = sample_frames(flag=src_val, num_frames=tgt_test_num_frames, dataset_name=src1_data)
	print('src test Data: ', src1_data)
	src1_test_data = sample_frames(flag=src_test, num_frames=tgt_test_num_frames, dataset_name=src1_data)

	print('Load Target Data')
	print('Target Data: ', tgt_data)
	tgt_test_data = sample_frames(flag=tgt_test, num_frames=tgt_test_num_frames, dataset_name=tgt_data)

	src1_train_dataloader_fake = DataLoader(HoujyufDataset_v2(src1_train_data_fake, train=True),
											batch_size=batch_size, shuffle=True)
	src1_train_dataloader_real = DataLoader(HoujyufDataset_v2(src1_train_data_real, train=True),
											batch_size=batch_size, shuffle=True)
	src1_val_dataloader = DataLoader(HoujyufDataset_v2(src1_val_data, train=False), batch_size=batch_size, shuffle=False)
	src1_test_dataloader = DataLoader(HoujyufDataset_v2(src1_test_data, train=False), batch_size=batch_size, shuffle=False)

	tgt_test_dataloader = DataLoader(HoujyufDataset_v2(tgt_test_data, train=False), batch_size=batch_size,
									 shuffle=False)
	return src1_train_dataloader_fake, src1_train_dataloader_real, \
		   src1_val_dataloader, src1_test_dataloader, tgt_test_dataloader


def get_rsgb_dataset(src1_data, src1_train_num_frames, tgt_data, tgt_test_num_frames, batch_size):
	if 'Oulu' not in src1_data and 'Replay' not in src1_data:
		tr_fake_flag, tr_real_flag, src_val, src_test, tgt_test = 51, 52, 3, 3, 3
	else:
		tr_fake_flag, tr_real_flag, src_val, src_test, tgt_test = 0, 1, 2, 3, 3
	print(src1_data, tr_fake_flag, tr_real_flag, src_val, src_test, tgt_test)
	print('Load Source Data')
	print('Source Data: ', src1_data)
	src1_train_data_fake = sample_frames(flag=tr_fake_flag, num_frames=src1_train_num_frames, dataset_name=src1_data)
	src1_train_data_real = sample_frames(flag=tr_real_flag, num_frames=src1_train_num_frames, dataset_name=src1_data)
	print('src val Data: ', src1_data)
	src1_val_data = sample_frames(flag=src_val, num_frames=tgt_test_num_frames, dataset_name=src1_data)
	print('src test Data: ', src1_data)
	src1_test_data = sample_frames(flag=src_test, num_frames=tgt_test_num_frames, dataset_name=src1_data)

	print('Load Target Data')
	print('Target Data: ', tgt_data)
	tgt_test_data = sample_frames(flag=tgt_test, num_frames=tgt_test_num_frames, dataset_name=tgt_data)

	src1_train_dataloader_fake = DataLoader(DepthDataset(src1_train_data_fake, train=True),
											batch_size=batch_size, shuffle=True)
	src1_train_dataloader_real = DataLoader(DepthDataset(src1_train_data_real, train=True),
											batch_size=batch_size, shuffle=True)
	src1_val_dataloader = DataLoader(DepthDataset(src1_val_data, train=False), batch_size=batch_size, shuffle=False)
	src1_test_dataloader = DataLoader(DepthDataset(src1_test_data, train=False), batch_size=batch_size, shuffle=False)

	tgt_test_dataloader = DataLoader(DepthDataset(tgt_test_data, train=False), batch_size=batch_size,
									 shuffle=False)
	return src1_train_dataloader_fake, src1_train_dataloader_real, \
		   src1_val_dataloader, src1_test_dataloader, tgt_test_dataloader


def get_grg_dataset(src1_data, src1_train_num_frames, tgt_data, tgt_test_num_frames, batch_size, size):
	tr_fake_flag, tr_real_flag, src_val, src_test, tgt_test = 0, 1, 2, 3, 3

	print(src1_data, tr_fake_flag, tr_real_flag, src_val, src_test, tgt_test)
	print('Load Source Data')
	print('Source Data: ', src1_data)
	src1_train_data_fake = sample_frames(flag=tr_fake_flag, num_frames=src1_train_num_frames, dataset_name=src1_data)
	src1_train_data_real = sample_frames(flag=tr_real_flag, num_frames=src1_train_num_frames, dataset_name=src1_data)
	print('src val Data: ', src1_data)
	src1_val_data = sample_frames(flag=src_val, num_frames=tgt_test_num_frames, dataset_name=src1_data)
	print('src test Data: ', src1_data)
	src1_test_data = sample_frames(flag=src_test, num_frames=tgt_test_num_frames, dataset_name=src1_data)

	print('Load Target Data')
	print('Target Data: ', tgt_data)
	tgt_test_data = sample_frames(flag=tgt_test, num_frames=tgt_test_num_frames, dataset_name=tgt_data)

	src1_train_dataloader_fake = DataLoader(grgDataset(src1_train_data_fake, train=True, size=size),
											batch_size=batch_size, shuffle=True)
	src1_train_dataloader_real = DataLoader(grgDataset(src1_train_data_real, train=True, size=size),
											batch_size=batch_size, shuffle=True)
	src1_val_dataloader = DataLoader(grgDataset(src1_val_data, train=False, size=size), batch_size=batch_size, shuffle=False)
	src1_test_dataloader = DataLoader(grgDataset(src1_test_data, train=False, size=size), batch_size=batch_size, shuffle=False)

	tgt_test_dataloader = DataLoader(grgDataset(tgt_test_data, train=False, size=size), batch_size=batch_size,
									 shuffle=False)
	return src1_train_dataloader_fake, src1_train_dataloader_real, \
		   src1_val_dataloader, src1_test_dataloader, tgt_test_dataloader


def get_sgtd_dataset(src1_data, src1_train_num_frames, tgt_data, tgt_test_num_frames, batch_size, test_batch):
	if 'Oulu' not in src1_data and 'Replay' not in src1_data:
		tr_fake_flag, tr_real_flag, src_val, src_test, tgt_test = 51, 52, 3, 3, 3
	else:
		tr_fake_flag, tr_real_flag, src_val, src_test, tgt_test = 0, 1, 2, 3, 3
	print(src1_data, tr_fake_flag, tr_real_flag, src_val, src_test, tgt_test)
	print('Load Source Data')
	print('Source Data: ', src1_data)
	src1_train_data_fake = sample_frames(flag=tr_fake_flag, num_frames=src1_train_num_frames, dataset_name=src1_data)
	src1_train_data_real = sample_frames(flag=tr_real_flag, num_frames=src1_train_num_frames, dataset_name=src1_data)
	print('src val Data: ', src1_data)
	src1_val_data = sample_frames(flag=src_val, num_frames=tgt_test_num_frames, dataset_name=src1_data)
	print('src test Data: ', src1_data)
	src1_test_data = sample_frames(flag=src_test, num_frames=tgt_test_num_frames, dataset_name=src1_data)

	print('Load Target Data')
	print('Target Data: ', tgt_data)
	tgt_test_data = sample_frames(flag=tgt_test, num_frames=tgt_test_num_frames, dataset_name=tgt_data)

	src1_train_dataloader_fake = DataLoader(RsgbDataset(src1_train_data_fake, train=True),
											batch_size=batch_size, shuffle=True)
	src1_train_dataloader_real = DataLoader(RsgbDataset(src1_train_data_real, train=True),
											batch_size=batch_size, shuffle=True)
	src1_val_dataloader = DataLoader(RsgbDataset(src1_val_data, train=False), batch_size=test_batch, shuffle=False)
	src1_test_dataloader = DataLoader(RsgbDataset(src1_test_data, train=False), batch_size=test_batch, shuffle=False)

	tgt_test_dataloader = DataLoader(RsgbDataset(tgt_test_data, train=False), batch_size=test_batch, shuffle=False)
	return src1_train_dataloader_fake, src1_train_dataloader_real, \
		   src1_val_dataloader, src1_test_dataloader, tgt_test_dataloader


def get_contrasive_dataset(src1_data, tgt_data, batch_size):
	if 'CASIA' in src1_data:
		tr_fake_flag, tr_real_flag, src_val, src_test, tgt_test = 0, 1, 3, 3, 3
		sub_num = 3
	if 'Replay' in src1_data:
		tr_fake_flag, tr_real_flag, src_val, src_test, tgt_test = 0, 1, 2, 3, 3
		sub_num = 2
	if 'msu' in src1_data:
		tr_fake_flag, tr_real_flag, src_val, src_test, tgt_test = 0, 1, 3, 3, 3
		sub_num = 2
	src1_train_dataloader_real_list = []
	src1_train_dataloader_fake_list = []
	print('Load Source Data')
	print('Source Data: ', src1_data)
	for i in range(1, sub_num + 1):
		src1_train_data_fake = sample_frames_multi(flag=0, dataset_name=src1_data, sub_flag=i)
		src1_train_data_real = sample_frames_multi(flag=1, dataset_name=src1_data, sub_flag=i)
		src1_train_dataloader_fake = DataLoader(ContrasiveDataset(src1_train_data_fake, train=True),
												batch_size=batch_size, shuffle=True)
		src1_train_dataloader_real = DataLoader(ContrasiveDataset(src1_train_data_real, train=True),
												batch_size=batch_size, shuffle=True)
		src1_train_dataloader_fake_list.append(src1_train_dataloader_fake)
		src1_train_dataloader_real_list.append(src1_train_dataloader_real)

	print('src val Data: ', src1_data)
	src1_val_data = sample_frames_multi(flag=src_val, dataset_name=src1_data)

	print('src test Data: ', src1_data)
	src1_test_data = sample_frames_multi(flag=src_test, dataset_name=src1_data)

	print('Load Target Data')
	print('Target Data: ', tgt_data)
	tgt_test_data = sample_frames_multi(flag=tgt_test, dataset_name=tgt_data)

	src1_val_dataloader = DataLoader(ContrasiveDataset(src1_val_data, train=False), batch_size=batch_size,
									 shuffle=False)
	src1_test_dataloader = DataLoader(ContrasiveDataset(src1_test_data, train=False), batch_size=batch_size,
									  shuffle=False)

	tgt_test_dataloader = DataLoader(ContrasiveDataset(tgt_test_data, train=False), batch_size=batch_size,
									 shuffle=False)
	return src1_train_dataloader_fake_list, src1_train_dataloader_real_list, \
		   src1_val_dataloader, src1_test_dataloader, tgt_test_dataloader


def get_multi_dataset(src1_data, tgt_data, batch_size):
	if 'CASIA' in src1_data:
		tr_fake_flag, tr_real_flag, src_val, src_test, tgt_test = 0, 1, 3, 3, 3
		sub_num = 3
	if 'Replay' in src1_data:
		tr_fake_flag, tr_real_flag, src_val, src_test, tgt_test = 0, 1, 2, 3, 3
		sub_num = 2
	if 'msu' in src1_data:
		tr_fake_flag, tr_real_flag, src_val, src_test, tgt_test = 0, 1, 3, 3, 3
		sub_num = 2
	src1_train_dataloader_real_list = []
	src1_train_dataloader_fake_list = []
	print('Load Source Data')
	print('Source Data: ', src1_data)
	for i in range(1, sub_num+1):
		src1_train_data_fake = sample_frames_multi(flag=0, dataset_name=src1_data, sub_flag=i)
		src1_train_data_real = sample_frames_multi(flag=1, dataset_name=src1_data, sub_flag=i)
		src1_train_dataloader_fake = DataLoader(HoujyufDataset(src1_train_data_fake, train=True),
											batch_size=batch_size, shuffle=True)
		src1_train_dataloader_real = DataLoader(HoujyufDataset(src1_train_data_real, train=True),
											batch_size=batch_size, shuffle=True)
		src1_train_dataloader_fake_list.append(src1_train_dataloader_fake)
		src1_train_dataloader_real_list.append(src1_train_dataloader_real)

	print('src val Data: ', src1_data)
	src1_val_data = sample_frames_multi(flag=src_val, dataset_name=src1_data)

	print('src test Data: ', src1_data)
	src1_test_data = sample_frames_multi(flag=src_test, dataset_name=src1_data)

	print('Load Target Data')
	print('Target Data: ', tgt_data)
	tgt_test_data = sample_frames_multi(flag=tgt_test, dataset_name=tgt_data)

	src1_val_dataloader = DataLoader(HoujyufDataset(src1_val_data, train=False), batch_size=batch_size,
									 shuffle=False)
	src1_test_dataloader = DataLoader(HoujyufDataset(src1_test_data, train=False), batch_size=batch_size,
									  shuffle=False)

	tgt_test_dataloader = DataLoader(HoujyufDataset(tgt_test_data, train=False), batch_size=batch_size,
									 shuffle=False)
	return src1_train_dataloader_fake_list, src1_train_dataloader_real_list, \
		   src1_val_dataloader, src1_test_dataloader, tgt_test_dataloader

def get_dataset_2(src1_data, src1_train_num_frames, src2_data, src2_train_num_frames,
				  tgt_data, tgt_test_num_frames, batch_size):
	src_test, tgt_test = 3, 3
	if 'Replay' in src1_data:
		tr1_fake_flag, tr1_real_flag, tr2_fake_flag, tr2_real_flag, src_val = 0, 1, 51, 52, 2
	elif 'Replay' in src2_data:
		tr1_fake_flag, tr1_real_flag, tr2_fake_flag, tr2_real_flag, src_val = 51, 52, 0, 1, 3
	else:
		tr1_fake_flag, tr1_real_flag, tr2_fake_flag, tr2_real_flag, src_val = 51, 52, 51, 52, 3
	print('Load Source Data')
	print('Source Data: ', src1_data)
	src1_train_data_fake = sample_frames(flag=tr1_fake_flag, num_frames=src1_train_num_frames, dataset_name=src1_data)
	src1_train_data_real = sample_frames(flag=tr1_real_flag, num_frames=src1_train_num_frames, dataset_name=src1_data)
	print('Source Data: ', src2_data)
	src2_train_data_fake = sample_frames(flag=tr2_fake_flag, num_frames=src2_train_num_frames, dataset_name=src2_data)
	src2_train_data_real = sample_frames(flag=tr2_real_flag, num_frames=src2_train_num_frames, dataset_name=src2_data)

	print('Load Target Data')
	print('Target Data: ', tgt_data)
	print('src val Data: ', src1_data)
	src1_val_data = sample_frames(flag=src_val, num_frames=tgt_test_num_frames, dataset_name=src1_data)
	print('src test Data: ', src1_data)
	src1_test_data = sample_frames(flag=src_test, num_frames=tgt_test_num_frames, dataset_name=src1_data)

	print('Load Target Data')
	print('Target Data: ', tgt_data)
	tgt_test_data = sample_frames(flag=tgt_test, num_frames=tgt_test_num_frames, dataset_name=tgt_data)

	src1_train_dataloader_fake = DataLoader(DepthDataset(src1_train_data_fake, train=True),
											batch_size=batch_size, shuffle=True)
	src1_train_dataloader_real = DataLoader(DepthDataset(src1_train_data_real, train=True),
											batch_size=batch_size, shuffle=True)
	src2_train_dataloader_fake = DataLoader(DepthDataset(src2_train_data_fake, train=True),
											batch_size=batch_size, shuffle=True)
	src2_train_dataloader_real = DataLoader(DepthDataset(src2_train_data_real, train=True),
											batch_size=batch_size, shuffle=True)

	src1_val_dataloader = DataLoader(DepthDataset(src1_val_data, train=False), batch_size=batch_size,
									 shuffle=False)
	src1_test_dataloader = DataLoader(DepthDataset(src1_test_data, train=False), batch_size=batch_size,
									  shuffle=False)
	tgt_dataloader = DataLoader(DepthDataset(tgt_test_data, train=False), batch_size=batch_size, shuffle=False)
	return src1_train_dataloader_fake, src1_train_dataloader_real, \
		   src2_train_dataloader_fake, src2_train_dataloader_real, \
		   src1_val_dataloader, src1_test_dataloader, tgt_dataloader

if __name__ == '__main__':
	test_list = '/DISK3/rosine/reproduction/data/train_txt/train_casia_video.txt'

	# train_data = Spoofing_train(test_list, transform=transforms.Compose(
	#     [RandomHorizontalFlip(), ToTensor(), Cutout(), Normaliztion()]), input_type='multi', color_aj='True')
	# dataloader_train = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=1)
	# count = 0
	# for i, sample_batched in enumerate(dataloader_train):
	#     # get the input
	#     inputs, spoof_label = sample_batched['image_x'].cuda(7), sample_batched['spoofing_label'].cuda(7)
	#
	#     print(inputs.size())


