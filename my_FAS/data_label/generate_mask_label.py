import os
import glob
import json
import shutil
import argparse


def generate_sublabel(dataset, tmp):
	video_index = 0
	sub_label = 1

	if dataset == 'CASIA':
		video_index = '_'.join(tmp[-2].split('_')[2:])
		# 1,3,5,7 domain1 2,4,6,8 domain2 HR_1, HR_2, HR_3, HR_4 domain3
		if tmp[-2] in ['1', '3', '5', '7']:
			sub_label = 1
		elif tmp[-2] in ['2', '4', '6', '8']:
			sub_label = 2
		elif tmp[-2] in ['HR_1', 'HR_2', 'HR_3', 'HR_4']:
			sub_label = 3
		# if tmp[-2] in ['1', '2']:
		#     sub_label = 1
		# elif tmp[-2] == 'HR_1':
		#     sub_label = 2
		# elif tmp[-2] in ['3', '4']:
		#     sub_label = 3
		# elif tmp[-2] == 'HR_2':
		#     sub_label = 4
		# elif tmp[-2] in ['5', '6']:
		#     sub_label = 5
		# elif tmp[-2] == 'HR_3':
		#     sub_label = 6
		# elif tmp[-2] in ['7', '8']:
		#     sub_label = 7
		# elif tmp[-2] == 'HR_4':
		#     sub_label = 8

		# if tmp[-2] in ['1', '2']:
		# 	sub_label = 1
		# elif tmp[-2] in ['HR_1']:
		# 	sub_label = 2

		# elif tmp[-2] in ['3', '4', 'HR_2']:
		# 	sub_label = 3

		# elif tmp[-2] in ['5', '6', 'HR_3']:
		# 	sub_label = 4

		# elif tmp[-2] in ['7', '8', 'HR_4']:
		# 	sub_label = 5

	elif dataset == 'Replay':
		# tmp[-2]  real  client110_session01_webcam_authenticate_adverse_1
		# attack  attack_highdef_client110_session01_highdef_photo_adverse
		tmp_list = tmp[-2].split('_')
		# if tmp_list[1] == 'real':
		#     sub_label = 1 if tmp_list[-2] == 'adverse' else 2
		# else:
		#     if tmp_list[-2] == 'photo':
		#         sub_label = 3 if tmp_list[-1] == 'adverse' else 4
		#     else:
		#         sub_label = 5 if tmp_list[-1] == 'adverse' else 6

		if tmp[1] == 'real':
			if tmp_list[-2] == 'adverse':
				sub_label = 1
			else:
				sub_label = 2
		if tmp[1] == 'attack':
			if tmp_list[-1] == 'adverse':
				sub_label = 1
			else:
				sub_label = 2

		# if tmp[1] == 'real':
		# 	if tmp_list[-2] == 'adverse':
		# 		sub_label = 1
		# 	else:
		# 		sub_label = 2
		# if tmp[1] == 'attack':
		# 	if tmp_list[-2] == 'photo':
		# 		sub_label = 3
		# 	else:
		# 		sub_label = 4

		video_index = tmp[-2][6:9]
	elif 'Oulu' in dataset:
		tmp2 = tmp[-2]
		video_index = tmp2[2]
		sub_label = '_'.join(tmp2[0], tmp2[1], tmp2[3])

	elif 'msu' in dataset:
		tmp_list = tmp[-2].split('_')
		sub_label = 1 if tmp_list[2] == 'laptop' else 2

	return video_index, sub_label


def generate_multi_sublabel(dataset, tmp):
	video_index = 0
	sub_label = 1

	if dataset == 'CASIA':
		video_index = tmp[-3]
		if tmp[-2] in ['1', '2']:
			sub_label = 1
		elif tmp[-2] == 'HR_1':
			sub_label = 2
		elif tmp[-2] in ['3', '4']:
			sub_label = 3
		elif tmp[-2] == 'HR_2':
			sub_label = 4
		elif tmp[-2] in ['5', '6']:
			sub_label = 5
		elif tmp[-2] == 'HR_3':
			sub_label = 6
		elif tmp[-2] in ['7', '8']:
			sub_label = 7
		elif tmp[-2] == 'HR_4':
			sub_label = 8

	elif dataset == 'Replay':
		# tmp[-2]  real  client110_session01_webcam_authenticate_adverse_1
		# attack  attack_highdef_client110_session01_highdef_photo_adverse
		tmp_list = tmp[-2].split('_')
		if tmp[-3] == 'real':
			sub_label = 1 if tmp_list[-2] == 'adverse' else 2
		else:
			if tmp_list[-2] == 'photo':
				sub_label = 3 if tmp_list[-1] == 'adverse' else 4
			else:
				sub_label = 5 if tmp_list[-1] == 'adverse' else 6
		video_index = tmp[-2][6:9]
	elif 'Oulu' in dataset:
		tmp2 = tmp[-2]
		video_index = tmp2[2]
		sub_label = '_'.join(tmp2[0], tmp2[1], tmp2[3])

	return video_index, sub_label


def process(dataset, data_dir, save_dir):
	all_final_json = []
	real_final_json = []
	fake_final_json = []
	train_final_json = []
	test_final_json = []
	devel_final_json = []

	train_real_json = []
	train_fake_json = []
	test_real_json = []
	test_fake_json = []
	devel_real_json = []
	devel_fake_json = []

	label_save_dir = save_dir + '/'
	if not os.path.exists(label_save_dir):
		os.makedirs(label_save_dir)

	f_all = open(label_save_dir + 'all_label.json', 'w')
	f_real = open(label_save_dir + 'real_label.json', 'w')
	f_fake = open(label_save_dir + 'fake_label.json', 'w')
	f_train = open(label_save_dir + 'train_label.json', 'w')
	f_test = open(label_save_dir + 'test_label.json', 'w')
	f_devel = open(label_save_dir + 'devel_label.json', 'w')

	f_train_real = open(label_save_dir + 'train_real.json', 'w')
	f_train_fake = open(label_save_dir + 'train_fake.json', 'w')
	f_test_real = open(label_save_dir + 'test_real.json', 'w')
	f_test_fake = open(label_save_dir + 'test_fake.json', 'w')
	f_devel_real = open(label_save_dir + 'devel_real.json', 'w')
	f_devel_fake = open(label_save_dir + 'devel_fake.json', 'w')

	dataset_path = data_dir + '/'
	frame_number = 0
	if dataset == 'CASIA':
		video_list = glob.glob(dataset_path + '/**/**/**/**')
	elif dataset == 'Replay':
		video_list1 = glob.glob(dataset_path + '/**/real/**')
		video_list2 = glob.glob(dataset_path + '/**/attack/**/**')
		video_list = video_list1 + video_list2
	elif dataset == 'msu':
		video_list = glob.glob(dataset_path + '/**/**/**')
	elif 'Oulu' in dataset:
		video_list = glob.glob(dataset_path + '/**/**')

	# Casia: /TrainType/label/subjects/videoType/
	# Replay  real:  /TrainType/real/videoType/
	##        attack:  /TrainType/attack/Method/videoType/
	# msu: /TrainType/label/videoType/
	# Oulu /TrainType/videoType/
	# print('handling:', len(video_list))
	test_num = 0
	devel_num = 0
	for video in video_list:

		for img in glob.glob(video + '/*jpg'):
			frame_number += 1
			tmp = img.split('/')
			# depth_name = tmp[5]+'_depth'
			# depth_path = '/'.join(tmp[:5])+'/'+depth_name+'/'+'/'.join(tmp[6:])
			depth_path = mask_dir+'/'.join(tmp[6:])
			TrainType = tmp[6]  # devel/train/test/
			if 'Oulu' not in dataset:
				label = 1 if tmp[7] == 'real' else 0  # real/attack
			else:
				label = 1 if tmp[-2].split('_')[3] == '1' else 0 # real/attack
			video_index, sub_label = generate_sublabel(dataset, tmp[6:])
			dict = {}
			dict['photo_path'] = img
			dict['photo_label'] = label
			dict['photo_belong_to_video_ID'] = frame_number
			dict['sub_label'] = sub_label
			dict['mask_path'] = depth_path
			#frame_number += 1
			if 'Oulu' not in dataset:
				if TrainType == 'train':
					train_final_json.append(dict)
					if label == 1:
						train_real_json.append(dict)
						real_final_json.append(dict)
					else:
						train_fake_json.append(dict)
						fake_final_json.append(dict)
				elif TrainType == 'test':
					if SAMPLE_TEST > 1 and test_num != SAMPLE_TEST:
						test_num += 1
						continue
					test_final_json.append(dict)
					test_num = 0
					if label == 1:
						test_real_json.append(dict)
						real_final_json.append(dict)
					else:
						test_fake_json.append(dict)
						fake_final_json.append(dict)
				else:
					if SAMPLE_DEVEL > 1 and devel_num != SAMPLE_DEVEL:
						devel_num += 1
						continue
					devel_final_json.append(dict)
					devel_num = 0
					if label == 1:
						devel_real_json.append(dict)
						real_final_json.append(dict)
					else:
						devel_fake_json.append(dict)
						fake_final_json.append(dict)
			else:
				protocol = dataset.split('_')[-1]
				videoType = tmp[-2].split('_')
				device, session, spooftype = videoType[0], videoType[1], videoType[3]
				if protocol == 'P1':
					if session == '1':
						pass
				if protocol == 'P1':
					pass
				if protocol == 'P1':
					pass
				if protocol == 'P1':
					pass
			all_final_json.append(dict)

	print('(train): ', len(train_final_json))
	print('(test): ', len(test_final_json))
	print('(devel): ', len(devel_final_json))
	print('(all): ', len(all_final_json))
	print('(real): ', len(real_final_json))
	print('(fake): ', len(fake_final_json))
	json.dump(all_final_json, f_all, indent=4)
	f_all.close()
	json.dump(real_final_json, f_real, indent=4)
	f_real.close()
	json.dump(fake_final_json, f_fake, indent=4)
	f_fake.close()
	json.dump(train_final_json, f_train, indent=4)
	f_train.close()
	json.dump(test_final_json, f_test, indent=4)
	f_test.close()
	json.dump(devel_final_json, f_devel, indent=4)
	f_devel.close()

	json.dump(train_real_json, f_train_real, indent=4)
	f_train_real.close()
	json.dump(train_fake_json, f_train_fake, indent=4)
	f_train_fake.close()
	json.dump(test_real_json, f_test_real, indent=4)
	f_test_real.close()
	json.dump(test_fake_json, f_test_fake, indent=4)
	f_test_fake.close()
	json.dump(devel_real_json, f_devel_real, indent=4)
	f_devel_real.close()
	json.dump(devel_fake_json, f_devel_fake, indent=4)
	f_devel_fake.close()


def process_multi(dataset, data_dir, save_dir):
	all_final_json = []
	real_final_json = []
	fake_final_json = []
	train_final_json = []
	test_final_json = []
	devel_final_json = []

	train_real_json = []
	train_fake_json = []
	test_real_json = []
	test_fake_json = []
	devel_real_json = []
	devel_fake_json = []

	label_save_dir = save_dir + '/'
	if not os.path.exists(label_save_dir):
		os.makedirs(label_save_dir)

	f_all = open(label_save_dir + 'all_label.json', 'w')
	f_real = open(label_save_dir + 'real_label.json', 'w')
	f_fake = open(label_save_dir + 'fake_label.json', 'w')
	f_train = open(label_save_dir + 'train_label.json', 'w')
	f_test = open(label_save_dir + 'test_label.json', 'w')
	f_devel = open(label_save_dir + 'devel_label.json', 'w')

	f_train_real = open(label_save_dir + 'train_real.json', 'w')
	f_train_fake = open(label_save_dir + 'train_fake.json', 'w')
	f_test_real = open(label_save_dir + 'test_real.json', 'w')
	f_test_fake = open(label_save_dir + 'test_fake.json', 'w')
	f_devel_real = open(label_save_dir + 'devel_real.json', 'w')
	f_devel_fake = open(label_save_dir + 'devel_fake.json', 'w')

	dataset_path = data_dir + '/'
	frame_number = 0
	# if dataset == 'CASIA':
	video_list = glob.glob(dataset_path + '/*_images_crop/**')
	video_list.sort()
	# elif dataset == 'Replay':
	#     video_list = glob.glob(dataset_path + '/**/**')

	# Casia: /TrainType/label/subjects/videoType/
	# Replay  real:  /TrainType/real/videoType/
	##        attack:  /TrainType/attack/Method/videoType/
	# msu: /TrainType/label/videoType/
	# Oulu /TrainType/videoType/
	# print('handling:', len(video_list))
	if dataset == 'CASIA':
		MAX_FAKE = 12
	else:
		MAX_FAKE = 12
	for video in video_list:
		frame_index = 0
		real_num = 0
		fake_num = 0
		for img in glob.glob(video + '/*jpg'):
			tmp = img.split('/')
			# depth_name = tmp[5] + '_depth'
			# depth_path = '/'.join(tmp[:5]) + '/' + depth_name + '/' + '/'.join(tmp[6:])
			depth_path = img.replace("_images_crop","_depth_crop")
			depth_path = depth_path.replace("_scene.jpg", "_depth_crop1D.jpg")
			TrainType = tmp[-3].split('_')[0]  # devel/train/test/
			# if 'Oulu' not in dataset:
			#     label = 1 if tmp[7] == 'real' else 0  # real/attack
			# else:
			#     label = 1 if tmp[-2].split('_')[3] == '1' else 0  # real/attack
			label_ = tmp[-2].split('_')[1]
			label = 0 if label_ == 'attack' else 1
			# print(label)
			video_index, sub_label = generate_multi_sublabel(dataset, tmp)
			dict = {}
			dict['photo_path'] = img
			dict['photo_label'] = label
			dict['photo_belong_to_video_ID'] = frame_number
			dict['sub_label'] = sub_label
			dict['mask_path'] = depth_path
			# dict['frame_index'] = frame_index
			frame_index += 1
			frame_number += 1
			if 'Oulu' not in dataset:
				if TrainType == 'train':
					train_final_json.append(dict)
					if label_ == 'real':
						train_real_json.append(dict)
						real_final_json.append(dict)
						real_num += 1
					else:
						train_fake_json.append(dict)
						fake_final_json.append(dict)
						fake_num += 1
				elif TrainType == 'test':
					test_final_json.append(dict)
					if label_ == 'real':
						test_real_json.append(dict)
						real_final_json.append(dict)
						real_num += 1
					else:
						test_fake_json.append(dict)
						fake_final_json.append(dict)
						fake_num += 1
				else:
					devel_final_json.append(dict)
					if label_ == 'real':
						devel_real_json.append(dict)
						real_final_json.append(dict)
						real_num += 1
					else:
						devel_fake_json.append(dict)
						fake_final_json.append(dict)
						fake_num += 1

			all_final_json.append(dict)
			if real_num == 36:
				break
			elif fake_num == MAX_FAKE:
				break

	print('(train): ', len(train_final_json))
	print('(test): ', len(test_final_json))
	print('(devel): ', len(devel_final_json))
	print('(all): ', len(all_final_json))
	print('(real): ', len(real_final_json))
	print('(fake): ', len(fake_final_json))
	json.dump(all_final_json, f_all, indent=4)
	f_all.close()
	json.dump(real_final_json, f_real, indent=4)
	f_real.close()
	json.dump(fake_final_json, f_fake, indent=4)
	f_fake.close()
	json.dump(train_final_json, f_train, indent=4)
	f_train.close()
	json.dump(test_final_json, f_test, indent=4)
	f_test.close()
	json.dump(devel_final_json, f_devel, indent=4)
	f_devel.close()

	json.dump(train_real_json, f_train_real, indent=4)
	f_train_real.close()
	json.dump(train_fake_json, f_train_fake, indent=4)
	f_train_fake.close()
	json.dump(test_real_json, f_test_real, indent=4)
	f_test_real.close()
	json.dump(test_fake_json, f_test_fake, indent=4)
	f_test_fake.close()
	json.dump(devel_real_json, f_devel_real, indent=4)
	f_devel_real.close()
	json.dump(devel_fake_json, f_devel_fake, indent=4)
	f_devel_fake.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# parser.add_argument('--data_dir', type=str, default='/DISK3/rosine/cross_dataset/frames_crop/Replay_crop/')
	parser.add_argument('--data_dir', type=str, default='/DISK3/houjyuf/dataset_old/ros_plgf/CASIA/')
	parser.add_argument('--mask_dir', type=str, default='/DISK3/rosine/cross_dataset/frames_crop/Casia_crop/')
	# parser.add_argument('--mask_dir', type=str,default='/DISK3/rosine/dataset/MSU_MFSD_new_1/')
	parser.add_argument('--dataset', type=str, default='CASIA')
	parser.add_argument('--save_dir', type=str, default='rosplgf_0414')
	parser.add_argument('--single', type=bool, default=False)
	parser.add_argument('--internal', type=int, default=1)
	args = parser.parse_args()
	save_dir = args.dataset+'_'+args.save_dir
	data_dir = args.data_dir # + args.dataset+'/'
	mask_dir = args.mask_dir
	R_stat = {}
	R_train_list = ['001', '002', '004', '006', '007', '008', '012', '016', '018', '025', '027', '103', '105', '108',
					'110']
	R_devel_list = ['003', '005', '010', '015', '017', '022', '029', '030', '101', '111', '113', '114', '116', '118',
					'119']
	R_test_list = ['009', '011', '013', '014', '019', '020', '021', '023', '024', '026',
				   '028', '031', '102', '104', '106', '107', '109', '112', '115', '117']
	C_devel_list = ['train_16', 'train_17', 'train_18', 'train_19', 'train_20',
					'test_1', 'test_2', 'test_3', 'test_4', 'test_5', 'test_6', 'test_7', 'test_8', 'test_9', 'test_10']
	M_train_list = ['001', '002', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '021', '022',
					'023']
	M_test_list = ['024', '026', '028', '029', '030', '032', '033', '034', '035', '036',
				   '037', '039', '042', '048', '049', '050', '051', '053', '054', '055']
	SAMPLE_TEST = args.internal
	SAMPLE_DEVEL = args.internal
	if args.single == True:
		process_multi(dataset=args.dataset, data_dir=args.data_dir, save_dir=args.save_dir)
	else:
		process(dataset=args.dataset, data_dir=data_dir, save_dir=save_dir)
