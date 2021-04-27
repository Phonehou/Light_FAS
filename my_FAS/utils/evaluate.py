from __future__ import division
from utils.utils import AverageMeter, accuracy, draw_roc, draw_far_frr_roc
from utils.statistic import get_EER_states, get_HTER_at_thr, calculate, calculate_threshold
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


def evaluate(valid_dataloader, model):
    criterion = nn.CrossEntropyLoss()
    valid_losses = AverageMeter()
    valid_top1 = AverageMeter()
    prob_dict = {}
    label_dict = {}
    model.eval()
    output_dict_tmp = {}
    target_dict_tmp = {}
    # print('\n'+'evaluate...')
    with torch.no_grad():
        for iter, (input, target, videoID, sub_label, mask) in enumerate(valid_dataloader):
            # if iter > 0:
            #     break
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            x_150, x_18, feature, cls_out = model(input)
            prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
            label = target.cpu().data.numpy()
            # print(label)
            # videoID = torch.Tensor(videoID)
            videoID = videoID.cpu().data.numpy()
            for i in range(len(prob)):
                if (videoID[i] in prob_dict.keys()):
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                else:
                    prob_dict[videoID[i]] = []
                    label_dict[videoID[i]] = []
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]] = []
                    target_dict_tmp[videoID[i]] = []
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
    prob_list = []
    label_list = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
        prob_list = np.append(prob_list, avg_single_video_prob)
        label_list = np.append(label_list, avg_single_video_label)
        # compute loss and acc for every video
        avg_single_video_output = sum(output_dict_tmp[key]) // len(output_dict_tmp[key])
        avg_single_video_target = sum(target_dict_tmp[key]) // len(target_dict_tmp[key])
        loss = criterion(avg_single_video_output, avg_single_video_target)
        acc_valid = accuracy(avg_single_video_output, avg_single_video_target, topk=(1,))
        valid_losses.update(loss.item())
        valid_top1.update(acc_valid[0])
    # print(label_list)
    auc_score = roc_auc_score(label_list, prob_list)
    # print('auc_score:',auc_score)
    cur_EER_valid, threshold, FRR_list, FAR_list, TPR_list = get_EER_states(prob_list, label_list)
    ACC_threshold, TN, FN, FP, TP = calculate_threshold(prob_list, label_list, threshold)
    cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)
    return [valid_losses.avg, valid_top1.avg, cur_EER_valid, cur_HTER_valid, auc_score, threshold, ACC_threshold * 100]

def evaluate_mask(valid_dataloader, model):
    criterion = nn.CrossEntropyLoss()
    valid_losses = AverageMeter()
    valid_top1 = AverageMeter()
    valid_masklosses = AverageMeter()

    prob_dict = {}
    label_dict = {}
    model.eval()
    # weights_txt = open('weights.txt','w')
    # for k, v in model.state_dict().items():
    #     weights_txt.write("layer {}".format(k)+'\n')
    #     weights_txt.write(str(v))
    #     weights_txt.write('\n')
    #     # if sum(v != v) > 0:
    #     #     print('dirty!', v)
    # weights_txt.close()
    output_dict_tmp = {}
    target_dict_tmp = {}

    with torch.no_grad():
        for iter, (input, target, videoID, sub_label, mask) in enumerate(valid_dataloader):
            if iter > 4:
                break
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            x_150, x_18, feature, cls_out = model(input)
            # cls_out, cue, x2, x4, x8, x16, x32 = model(input)

            # print(sum(x2))
            # if sum(x2) == nan:
            prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
            # print(prob)
            label = target.cpu().data.numpy()
            # print(label)
            # videoID = torch.Tensor(videoID)
            videoID = videoID.cpu().data.numpy()
            for i in range(len(prob)):
                if (videoID[i] in prob_dict.keys()):
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                else:
                    prob_dict[videoID[i]] = []
                    label_dict[videoID[i]] = []
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]] = []
                    target_dict_tmp[videoID[i]] = []
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
    prob_list = []
    label_list = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
        # print(sum(prob_dict[key]))
        prob_list = np.append(prob_list, avg_single_video_prob)
        label_list = np.append(label_list, avg_single_video_label)
        # compute loss and acc for every video
        avg_single_video_output = sum(output_dict_tmp[key]) // len(output_dict_tmp[key])
        avg_single_video_target = sum(target_dict_tmp[key]) // len(target_dict_tmp[key])
        loss = criterion(avg_single_video_output, avg_single_video_target)
        acc_valid = accuracy(avg_single_video_output, avg_single_video_target, topk=(1,))
        valid_losses.update(loss.item())
        valid_top1.update(acc_valid[0])

    # auc_score = roc_auc_score(label_list, prob_list)
    auc_score = 0
    cur_EER_valid, threshold, FRR_list, FAR_list, TPR_list = get_EER_states(prob_list, label_list)
    ACC_threshold, TN, FN, FP, TP = calculate_threshold(prob_list, label_list, threshold)
    cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)
    return [valid_losses.avg, valid_top1.avg, cur_EER_valid, cur_HTER_valid, auc_score, threshold, ACC_threshold * 100]

def test_mask(test_dataloader, model, valid_thre, epoch=''):
    valid_top1 = AverageMeter()
    prob_dict = {}
    label_dict = {}
    model.eval()
    output_dict_tmp = {}
    target_dict_tmp = {}
    number = 0
    with torch.no_grad():
        for iter, (input, target, videoID, sub_label, mask) in enumerate(test_dataloader):
            if iter > 4:
                break
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            # cls_out, cue, x2, x4, x8, x16, x32 = model(input)
            x_150, x_18, feature, cls_out = model(input)
            prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
            label = target.cpu().data.numpy()
            videoID = videoID.cpu().data.numpy()
            for i in range(len(prob)):
                if (videoID[i] in prob_dict.keys()):
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                else:
                    prob_dict[videoID[i]] = []
                    label_dict[videoID[i]] = []
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]] = []
                    target_dict_tmp[videoID[i]] = []
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                    number += 1
                    # if (number % 100 == 0):
                    # print('**Testing** ', number, ' photos done!')
    # print('**Testing** ', number, ' photos done!\n')
    prob_list = []
    label_list = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
        prob_list = np.append(prob_list, avg_single_video_prob)
        label_list = np.append(label_list, avg_single_video_label)
        # compute loss and acc for every video
        avg_single_video_output = sum(output_dict_tmp[key]) // len(output_dict_tmp[key])
        avg_single_video_target = sum(target_dict_tmp[key]) // len(target_dict_tmp[key])
        acc_valid = accuracy(avg_single_video_output, avg_single_video_target, topk=(1,))
        # ACC top1, threshold 0.5
        valid_top1.update(acc_valid[0])

    cur_EER_valid, threshold, FRR_list, FAR_list, TPR_list = get_EER_states(prob_list, label_list)
    # args at EER threshold
    avg_threshold = calculate_threshold(prob_list, label_list, threshold)  #[ACC_threshold, TN, FN, FP, TP]
    # args at valid threshold
    avg_valid_threshold = calculate_threshold(prob_list, label_list, valid_thre)
    # args at 0.5
    avg_half = calculate(prob_list, label_list) # APCER, NPCER, ACER, ACC
    auc_score = 0
    # auc_score = roc_auc_score(label_list, prob_list)
    draw_roc(TPR_list, FAR_list, auc_score, id=epoch)
    draw_far_frr_roc(FRR_list, FAR_list, auc_score, id=epoch)
    cur_HTER_best = get_HTER_at_thr(prob_list, label_list, threshold)
    cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, valid_thre)
    return [valid_top1.avg, cur_EER_valid, cur_HTER_best, auc_score, avg_threshold, threshold, avg_valid_threshold, cur_HTER_valid, avg_half]


