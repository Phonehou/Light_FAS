'''
Code of 'Searching Central Difference Convolutional Networks for Face Anti-Spoofing'
By Zitong Yu & Zhuo Su, 2019

If you use the code, please cite:
@inproceedings{yu2020searching,
    title={Searching Central Difference Convolutional Networks for Face Anti-Spoofing},
    author={Yu, Zitong and Zhao, Chenxu and Wang, Zezheng and Qin, Yunxiao and Su, Zhuo and Li, Xiaobai and Zhou, Feng and Zhao, Guoying},
    booktitle= {CVPR},
    year = {2020}
}

Only for research purpose, and commercial use is not allowed.

MIT License
Copyright (c) 2020
'''

import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import pdb



class AverageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


def best_hter_threshold(score_file):
    with open(score_file, 'r') as file:
        lines = file.readlines()

    data = []
    labels=[]
    scores = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        #pdb.set_trace()
        label = int(tokens[1])
        scores.append(score)
        labels.append(label)

        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1

    best_error = count    # account ACER (or ACC)
    best_threshold = 0.0
    best_ACC = 0.0
    best_ACER = 0.0
    best_APCER = 0.0
    best_BPCER = 0.0


    for d in data:
        threshold = d['map_score']

        type1 = len([s for s in data if s['map_score'] <= threshold and s['label'] == 1])
        type2 = len([s for s in data if s['map_score'] > threshold and s['label'] == 0])

        ACC = 1-(type1 + type2) / count
        APCER = type2 / num_fake
        BPCER = type1 / num_real
        ACER = (APCER + BPCER) / 2.0

        if ACER < best_error:
            best_error = ACER
            best_threshold = threshold
            best_ACC = ACC
            best_ACER = ACER
            best_APCER = APCER
            best_BPCER = BPCER


    FN = len([s for s in data if s['map_score'] <= best_threshold and s['label'] == 1])
    FP = len([s for s in data if s['map_score'] > best_threshold and s['label'] == 0])
    TN = int(num_fake-FP)
    TP = int(num_real-FN)

    fpr, tpr, thresholds_eer = roc_curve(labels, scores, pos_label=1)
    eer,best_th = get_err_threhold(fpr, tpr, thresholds_eer)
    auc = metrics.auc(fpr, tpr)
    return best_threshold, best_ACC, best_APCER, best_BPCER, best_ACER,FN,FP,TN,TP,eer,auc


def get_err_threhold(fpr, tpr, threshold):
    # RightIndex=(tpr+(1-fpr)-1);
    # right_index = np.argmax(RightIndex)
    # best_th = threshold[right_index]
    # err = fpr[right_index]
    differ_tpr_fpr_1=tpr+fpr-1.0
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]

    return err,best_th


def test_threshold_based(threshold, score_file):
    with open(score_file, 'r') as file:
        lines = file.readlines()

    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        data.append({'map_score': angle, 'label': type})
        if type==1:
            num_real += 1
        else:
            num_fake += 1


    type1 = len([s for s in data if s['map_score'] <= threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > threshold and s['label'] == 0])

    ACC = 1-(type1 + type2) / count
    APCER = type2 / num_fake
    BPCER = type1 / num_real
    ACER = (APCER + BPCER) / 2.0

    return ACC, APCER, BPCER, ACER


#def performances(dev_scores, dev_labels, test_scores, test_labels):
def performances(map_score_test_filename):
    # test
    with open(map_score_test_filename, 'r') as file2:
        lines = file2.readlines()
    test_scores = []
    test_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])    #label = int(tokens[1])
        test_scores.append(score)
        test_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1

    # test based on test_threshold
    fpr_test,tpr_test,thresholds_test = roc_curve(test_labels, test_scores, pos_label=1)
    err_test, best_test_threshold = get_err_threhold(fpr_test, tpr_test, thresholds_test)
    auc = metrics.auc(fpr_test, tpr_test)

    type1 = len([s for s in data if s['map_score'] <= best_test_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > best_test_threshold and s['label'] == 0])
    FN = type1
    FP = type2
    TN = num_fake - FP
    TP = num_real - FN
    test_threshold_ACC = 1-(type1 + type2) / count
    test_threshold_APCER = type2 / num_fake
    test_threshold_BPCER = type1 / num_real
    test_threshold_ACER = (test_threshold_APCER + test_threshold_BPCER) / 2.0

    return  best_test_threshold, test_threshold_ACC, test_threshold_APCER, test_threshold_BPCER, \
            test_threshold_ACER,FN,FP,TN,TP,err_test,auc











if __name__ == '__main__':
    txt_path = r'/DISK3/rosine/CVPR2020_paper_codes/logs/330_CDCNpp_map_score_test.txt'
    best_thrs, best_ACC, best_APCER, best_BPCER, best_hter,FN,FP,TN,TP,eer,auc = best_hter_threshold(txt_path)
    line = 'Get the best Hter: {} at  Thr: {}\n ' \
        'Best_Testing: EER:{:.6f} AUC:{:.6f} Acc:{:.3f} FNR:{:.6f}  FPR:{:.6f}  TN:{}  FP:{}  FN:{}  TP:{}' \
        .format(best_hter, best_thrs, eer, auc, best_ACC, best_BPCER, best_APCER, TN, FP, FN, TP)
#
#     eer_thrs, best_ACC, best_APCER, best_BPCER, hter, FN, FP, TN, TP, eer, auc = performances(txt_path)
#     line = 'Testing_result(eer:fnr=fpr====>>thr):  Hter: {} at  Thr: {}\n ' \
#     'EER:{:.6f}  AUC:{:.6f} Acc:{:.3f} FNR:{:.6f}  FPR:{:.6f}  TN:{}  FP:{}  FN:{}  TP:{}' \
#         .format(hter, eer_thrs, eer, auc, best_ACC, best_BPCER, best_APCER, TN, FP, FN, TP)
    print(line)
#

def performances_SiWM_EER(map_score_val_filename):

    # val
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1

    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1)
    val_err, val_threshold = get_err_threhold(fpr, tpr, threshold)

    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    val_ACC = 1-(type1 + type2) / count
    val_APCER = type2 / num_fake
    val_BPCER = type1 / num_real
    val_ACER = (val_APCER + val_BPCER) / 2.0



    return val_threshold, val_err, val_ACC, val_APCER, val_BPCER, val_ACER




def get_err_threhold_CASIA_Replay(fpr, tpr, threshold):
    RightIndex=(tpr+(1-fpr)-1);
    right_index = np.argmax(RightIndex)
    best_th = threshold[right_index]
    err = fpr[right_index]

    differ_tpr_fpr_1=tpr+fpr-1.0

    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]

    #print(err, best_th)
    return err, best_th, right_index


def performances_CASIA_Replay(map_score_val_filename):

    # val
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = float(tokens[1])  # int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1

    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1)
    val_err, val_threshold, right_index = get_err_threhold_CASIA_Replay(fpr, tpr, threshold)

    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    val_ACC = 1-(type1 + type2) / count

    FRR = 1- tpr    # FRR = 1 - TPR

    HTER = (fpr+FRR)/2.0    # error recognition rate &  reject recognition rate

    return val_ACC, fpr[right_index], FRR[right_index], HTER[right_index]










def performances_ZeroShot(map_score_val_filename):

    # val
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        count += 1
        tokens = line.split()
        score = float(tokens[0])
        label = int(tokens[1])
        val_scores.append(score)
        val_labels.append(label)
        data.append({'map_score': score, 'label': label})
        if label==1:
            num_real += 1
        else:
            num_fake += 1

    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1)
    auc_val = metrics.auc(fpr, tpr)

    val_err, val_threshold, right_index = get_err_threhold_CASIA_Replay(fpr, tpr, threshold)

    type1 = len([s for s in data if s['map_score'] <= val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    val_ACC = 1-(type1 + type2) / count

    FRR = 1- tpr    # FRR = 1 - TPR

    HTER = (fpr+FRR)/2.0    # error recognition rate &  reject recognition rate

    return val_ACC, auc_val, HTER[right_index]

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

