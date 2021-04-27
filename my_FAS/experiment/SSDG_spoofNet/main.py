import sys

sys.path.append('../../')

import argparse
import random
import os
from timeit import default_timer as timer
from datetime import datetime
import time
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from utils.utils import save_checkpoint, AverageMeter, Logger, accuracy, mkdirs, adjust_learning_rate, time_to_str, \
    draw_far_frr_roc, draw_roc
from sklearn.metrics import roc_auc_score
from utils.statistic import get_EER_states, get_HTER_at_thr, calculate, calculate_threshold
from utils.get_loader import get_mask_dataset
from loss.AdLoss import Real_AdLoss_single
from loss.hard_triplet_loss import HardTripletLoss
from utils.dataset import Normalization255_GRAY


from models.spoofNet import SpoofNet, Discriminator


def train(is_test=False):
    mkdirs(checkpoint_path, best_model_path, logs)

    # load data
    src1_train_dataloader_real, src1_train_dataloader_fake,\
    src1_valid_dataloader, src1_test_dataloader, tgt_test_dataloader = get_mask_dataset(config.src1_data+'_'+config.postfix, config.src1_train_num_frames,
                                                                   config.tgt_data+'_'+config.postfix, config.tgt_test_num_frames,config.batch_size)
    best_model_ACC = 0.0
    best_model_HTER = 1.0   # set on EER
    best_model_AUC = 0.0
    best_model_ACER = 0.0   # cross HTER at 0.5
    best_cross_HTER = 1.0   # cross HTER at EER

    # 0:cls_loss, 1:top-1, 2:EER, 3:HTER, 4:AUC, 5:threshold, 6:ACC@threshold, 7:reg_loss
    valid_args = [np.inf, 0, 0, 0, 0, 0, 0, np.inf]

    loss_classifier = AverageMeter()
    classifer_top1 = AverageMeter()

    net = SpoofNet().to(device)

    ad_net_real = Discriminator().to(device)

    criterion = {
        'softmax': nn.CrossEntropyLoss().cuda(),
        'triplet': HardTripletLoss(margin=0.1, hardest=False).cuda()
    }

    optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, net.parameters()), "lr": config.init_lr},
        {"params": filter(lambda p: p.requires_grad, ad_net_real.parameters()), "lr": config.init_lr},
    ]
    optimizer = optim.SGD(optimizer_dict, lr=config.init_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    init_param_lr = []
    for param_group in optimizer.param_groups:
        init_param_lr.append(param_group["lr"])

    src1_train_iter_real = iter(src1_train_dataloader_real)  # iter   CASIA: real 3600/10   attack 10800/10
    src1_iter_per_epoch_real = len(src1_train_iter_real)
    src1_train_iter_fake = iter(src1_train_dataloader_fake)
    src1_iter_per_epoch_fake = len(src1_train_iter_fake)
    # print(src1_iter_per_epoch_real,src1_iter_per_epoch_fake)
    iter_per_epoch = max(src1_iter_per_epoch_real, src1_iter_per_epoch_fake)
    max_iter = iter_per_epoch * config.epochs
    epoch = 0
    if (len(config.gpus) > 1):
        net = torch.nn.DataParallel(net).cuda()

    if config.resume:
        # path_checkpoint = "./models/checkpoint/ckpt_best_1.pth"  #
        checkpoint_path_to = os.path.join(checkpoint_path, '_checkpoint.pth.tar')
        checkpoint = torch.load(checkpoint_path_to)  #

        net.load_state_dict(checkpoint["state_dict"])  #

        # optimizer.load_state_dict(checkpoint["optimizer"])  #
        epoch = checkpoint["epoch"]  #
        print('resume from epoch:', epoch)

    log = Logger()
    log.open(logs + config.tgt_data + '_log_SSDG.txt', mode='a')

    log.write("\n------------------------------------- [START %s] %s\n\n" % (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 40))

    log.write('** start training target model! **\n')
    log.write(
        '------------|------------------ VALID ----------------|------ classifier ------|------ Current Best ------|-------------|\n')
    log.write(
        '    iter    |   loss   top-1   HTER    AUC  threshold |  c_loss      top-1     |   top-1   HTER    AUC    |    time     |\n')
    log.write(
        '------------------------------------------------------------------------------------------------------------------------|\n')
    start = timer()

    for iter_num in range(epoch*iter_per_epoch, max_iter + 1):
        # for iter_num in range(10):
        if (iter_num % src1_iter_per_epoch_real == 0):  # finish an epoch of src1_train, restart the iteration
            src1_train_iter_real = iter(src1_train_dataloader_real)

        if (iter_num % src1_iter_per_epoch_fake == 0):
            src1_train_iter_fake = iter(src1_train_dataloader_fake)

        if (iter_num != 0 and iter_num % iter_per_epoch == 0):
            epoch = epoch + 1
        param_lr_tmp = []
        for param_group in optimizer.param_groups:
            param_lr_tmp.append(param_group["lr"])

        net.train(True)
        ad_net_real.train(True)
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, epoch, init_param_lr, config.lr_epoch_1, config.lr_epoch_2)
        # print(src1_train_iter_real.next())
        ######### data prepare #########
        src1_img_real, src1_label_real = src1_train_iter_real.next()
        src1_img_real = src1_img_real.cuda()
        src1_label_real = src1_label_real.cuda()

        input1_real_shape = src1_img_real.shape[0]  # [10,3,256,256]

        src1_img_fake, src1_label_fake = src1_train_iter_fake.next()
        src1_img_fake = src1_img_fake.cuda()
        src1_label_fake = src1_label_fake.cuda()

        input1_fake_shape = src1_img_fake.shape[0]  # [10,3,256,256]

        input_data = torch.cat([src1_img_real, src1_img_fake], dim=0)  # [20,3,256,256]

        source_label = torch.cat([src1_label_real, src1_label_fake], dim=0)  # [20]
        if iter_num == 0:
            visual_map(input_data, epoch)
        # print(input_data.shape, source_label.shape)
        ######### forward #########
        x_150, x_18, feature, classifier_label_out = net(input_data)

        ######### single side adversarial learning #########
        # input1_shape = input1_real_shape + input1_fake_shape  # input1_shape 20
        # feature_real_1 = feature.narrow(0, 0, input1_real_shape)  # narrow(dims, start, length)
        # # x_150.shape [20,16,256,256],x_18.shape [20,32,32,32], feature.shape [20,512],classifier_label_out.shape [20,2]
        # feature_real = feature_real_1
        # discriminator_out_real = ad_net_real(feature_real)  # shape [10,3]
        # real_shape_list = []
        # real_shape_list.append(input1_real_shape)  # input1_real_shape 10
        # real_adloss = Real_AdLoss_single(discriminator_out_real, criterion["softmax"], real_shape_list)

        ######### unbalanced triplet loss #########
        real_domain_label_1 = torch.LongTensor(input1_real_shape, 1).fill_(0).cuda()
        fake_domain_label_1 = torch.LongTensor(input1_fake_shape, 1).fill_(1).cuda()
        source_domain_label = torch.cat([real_domain_label_1, fake_domain_label_1], dim=0).view(-1)
        triplet = criterion["triplet"](feature, source_domain_label)  # feature [20,512]  source_domain_label  [20,]

        ######### cross-entropy loss #########
        cls_loss = criterion["softmax"](classifier_label_out.narrow(0, 0, input_data.size(0)), source_label)

        ######### backward #########
        # total_loss = cls_loss + config.lambda_triplet * triplet + config.lambda_adreal * real_adloss
        total_loss = cls_loss + config.lambda_triplet * triplet
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_classifier.update(cls_loss.item())
        acc = accuracy(classifier_label_out.narrow(0, 0, input_data.size(0)), source_label, topk=(1,))
        classifer_top1.update(acc[0])
        print('\r', end='', flush=True)
        print(
            ' %5d / %3d |  %5.3f  %6.3f  %6.3f  %6.3f  %6.3f  |  %6.3f     %6.3f     |  %6.3f  %6.3f  %6.3f  | %s'
            % (
                iter_num + 1, iter_per_epoch,
                valid_args[0], valid_args[6], valid_args[3] * 100, valid_args[4] * 100, valid_args[5],
                loss_classifier.avg, classifer_top1.avg,
                float(best_model_ACC), float(best_model_HTER * 100), float(best_model_AUC * 100),
                time_to_str(timer() - start, 'min'))
            , end='', flush=True)

        # eval after an epoch

        if (iter_num != 0 and (iter_num + 1) % iter_per_epoch == 0):
        # if iter_num <= 1:
            # visualization
            #FeatureMap2Heatmap(input_data, x_150, x_18, visual_path, config.model, source_label, epoch, config.batch_size)

            # 0:loss, 1:top-1, 2:EER, 3:HTER, 4:AUC, 5:threshold, 6:ACC@threshold
            valid_args = evaluate_mask(src1_valid_dataloader, net)

            is_best = valid_args[3] <= best_model_HTER     # is_best=True when intra is best or cross is best
            best_model_HTER = min(valid_args[3], best_model_HTER)
            threshold = valid_args[5]
            if (valid_args[3] <= best_model_HTER):
                best_model_ACC = valid_args[6]
                best_model_AUC = valid_args[4]

            log.write(
                ' %5d /%3d |  %5.3f  %6.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f |  %6.3f  %6.3f  %6.3f  | %s'
                % (
                    (iter_num + 1), iter_per_epoch,
                    valid_args[0], valid_args[6], valid_args[3] * 100, valid_args[4] * 100, valid_args[5],
                    loss_classifier.avg, classifer_top1.avg,
                    float(best_model_ACC), float(best_model_HTER * 100), float(best_model_AUC * 100),
                    time_to_str(timer() - start, 'min'),
                ))
            log.write('\n')

            if config.src1_data == 'Replay':
                # test_args 0: ACC  1 EER  2  HTER  3 AUC 4 arg@Thre 5 threshold 6 arg@validThre 7 HTER@validThre 8 arg@0.5Thre
                test_args = evaluate_mask(src1_test_dataloader, net, 'test' ,threshold, str(epoch))
                top1, eer, auc, hter_eer, eer_thrs, [best_ACC, TN, FP, FN, TP] = \
                    test_args[0].item(), test_args[1], test_args[3], test_args[2], test_args[5], test_args[4]
                hter_valid, valid_thrs, valid_ACC, APCER, NPCER, ACER = test_args[7], threshold, test_args[6][0].item(), \
                                                                        test_args[8][0], test_args[8][1], test_args[8][2]
                line = 'Intra testing result:  top-1 ACC:{:6.3f}  EER:{:6.3f}  AUC:{:6.3f} \n ' \
                       'Hter: {:6.3f} at EER_Thr: {:.5f} Acc:{:6.3f} TN:{}  FP:{}  FN:{}  TP:{} \n ' \
                       'Hter: {:6.3f} at valid_Thr: {:.5f} Acc:{:6.3f}  at half_Thr: APCER:{:6.3f} NPCER:{:6.3f} ACER:{:6.3} ' \
                    .format(top1 * 100, eer * 100, auc * 100, hter_eer * 100, eer_thrs, best_ACC * 100, TN, FP, FN, TP,
                            hter_valid * 100, valid_thrs, valid_ACC * 100, APCER * 100, NPCER * 100, ACER * 100)
                log.write('{}\n'.format(line))

            if is_test == True:
                # test_args 0: ACC  1 EER  2  HTER  3 AUC 4 arg@Thre 5 threshold 6 arg@validThre 7 HTER@validThre 8 arg@0.5Thre
                test_args = evaluate_mask(tgt_test_dataloader, net, 'test', threshold, str(epoch))
                is_cross_best = test_args[2] <= best_cross_HTER
                best_model_ACER = test_args[8][2] if test_args[8][2] <= best_model_ACER else best_model_ACER
                best_cross_HTER = test_args[2] if test_args[2] <= best_cross_HTER else best_cross_HTER
                is_best = is_best or is_cross_best
                # top1, eer, auc, hter_eer, eer_thrs, [best_ACC, TN, FP, FN, TP] = \
                #     test_args[0].item(),test_args[1].item(),test_args[3].item(),test_args[2].item(),test_args[5].item(),test_args[4].item()
                # hter_valid, valid_thrs, valid_ACC, APCER, NPCER, ACER = test_args[7].item(),threshold,test_args[6][0].item(),test_args[8][0].item(),test_args[8][1].item(),test_args[8][2].item()
                top1, eer, auc, hter_eer, eer_thrs, [best_ACC, TN, FP, FN, TP] = \
                    test_args[0].item(), test_args[1], test_args[3], test_args[2], test_args[5], test_args[4]
                hter_valid, valid_thrs, valid_ACC, APCER, NPCER, ACER = test_args[7], threshold, test_args[6][0].item(), test_args[8][0], test_args[8][1], test_args[8][2]
                line = 'Cross testing result:  top-1 ACC:{:6.3f}  EER:{:6.3f}  AUC:{:6.3f} \n ' \
                        'Hter: {:6.3f} at EER_Thr: {:.5f} Acc:{:6.3f} TN:{}  FP:{}  FN:{}  TP:{} \n '\
                        'Hter: {:6.3f} at valid_Thr: {:.5f} Acc:{:6.3f}  at half_Thr: APCER:{:6.3f} NPCER:{:6.3f} ACER:{:6.3} '\
                    .format(top1*100, eer*100, auc*100, hter_eer*100, eer_thrs, best_ACC*100, TN, FP, FN, TP,
                            hter_valid*100, valid_thrs, valid_ACC*100, APCER*100, NPCER*100, ACER*100)
                log.write('{}\n'.format(line))

            is_best = True
            save_list = [epoch, valid_args, best_model_HTER, best_model_ACC, best_model_ACER, threshold]
            save_checkpoint(save_list, is_best, net, config.gpus, checkpoint_path, best_model_path)

            time.sleep(0.01)


def evaluate_mask(valid_dataloader=None, model=None, mode='val', valid_thre=0.5, epoch=''):
    criterion = nn.CrossEntropyLoss()
    valid_losses = AverageMeter()
    valid_top1 = AverageMeter()
    valid_masklosses = AverageMeter()

    prob_dict = {}
    label_dict = {}
    model.eval()
    output_dict_tmp = {}
    target_dict_tmp = {}

    with torch.no_grad():
        for iter, (input, target, videoID) in enumerate(valid_dataloader):
            # if iter < 4:
            #     break
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            x_150, x_18, feature, cls_out = model(input)
            # cls_out, cue, x2, x4, x8, x16, x32 = model(input)

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

    if mode == 'val':
        auc_score = roc_auc_score(label_list, prob_list)
        # auc_score = 0
        cur_EER_valid, threshold, FRR_list, FAR_list, TPR_list = get_EER_states(prob_list, label_list)
        ACC_threshold, TN, FN, FP, TP = calculate_threshold(prob_list, label_list, threshold)
        cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)
        return [valid_losses.avg, valid_top1.avg, cur_EER_valid, cur_HTER_valid, auc_score, threshold, ACC_threshold * 100]
    else:
        cur_EER_valid, threshold, FRR_list, FAR_list, TPR_list = get_EER_states(prob_list, label_list)
        # args at EER threshold
        avg_threshold = calculate_threshold(prob_list, label_list, threshold)  # [ACC_threshold, TN, FN, FP, TP]
        # args at valid threshold
        avg_valid_threshold = calculate_threshold(prob_list, label_list, valid_thre)
        # args at 0.5
        avg_half = calculate(prob_list, label_list)  # APCER, NPCER, ACER, ACC
        # auc_score = 0
        auc_score = roc_auc_score(label_list, prob_list)
        draw_roc(TPR_list, FAR_list, auc_score, id=epoch)
        draw_far_frr_roc(FRR_list, FAR_list, auc_score, id=epoch)
        cur_HTER_best = get_HTER_at_thr(prob_list, label_list, threshold)
        cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, valid_thre)
        return [valid_top1.avg, cur_EER_valid, cur_HTER_best, auc_score, avg_threshold, threshold, avg_valid_threshold,
                cur_HTER_valid, avg_half]

def visual_map(oris, iter_num):
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)
    # visualize plgfmap  [batch, 6, 224, 224]
    # plgf = plgfmap[:, 3:, :, :]  # last three channels

    # ind = [0, plgfmap.shape[0]//2]
    ind = oris.shape[0]
    for i in range(ind):
        # img = plgfmap[i].cpu().data.numpy()
        # I_b, I_g, I_r = Normalization255_GRAY(img[0]), Normalization255_GRAY(img[1]), Normalization255_GRAY(img[2])
        # img[0], img[1], img[2] = I_r, I_g, I_b
        # img = img.astype(np.uint8)
        # img = img.transpose(1, 2, 0)
        # cv2.imwrite(visual_path + str(iter_num) + '_'+str(i)+'_x150.jpg', img)

        img = oris[i].cpu().data.numpy()
        I_b, I_g, I_r = Normalization255_GRAY(img[0]), Normalization255_GRAY(img[1]), Normalization255_GRAY(img[2])
        img[0], img[1], img[2] = I_r, I_g, I_b
        img = img.astype(np.uint8)
        img = img.transpose(1, 2, 0)
        cv2.imwrite(visual_path + str(iter_num) + '_'+str(i)+'_input.jpg', img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default="2", help='runs gpu number')
    parser.add_argument('--model', type=str, default="C2R_ori", help='task')
    parser.add_argument('--postfix', type=str, default='rosplgf', help='select dataset type')
    parser.add_argument('--src1_data', type=str, default='CASIA', help='src1 dataset')
    parser.add_argument('--tgt_data', type=str, default='Replay', help='tgt dataset')

    parser.add_argument('--seed', type=int, default=666, help='gpu seed')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='SGD weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--init_lr', type=str, default=0.01, help='init learning rate')
    parser.add_argument('--lr_epoch_1', type=int, default=0, help='lr epoch 1')
    parser.add_argument('--lr_epoch_2', type=int, default=150, help='lr epoch 2')

    parser.add_argument('--lambda_triplet', type=float, default=1, help='lambda triplet')
    parser.add_argument('--lambda_adreal', type=float, default=0.5, help='lambda adreal')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=30, help='epochs')
    parser.add_argument('--tgt_best_model_name', type=str, default='model_best_0.0446_11.pth.tar',
                        help='tgt best model name')

    parser.add_argument('--src1_train_num_frames', type=int, default=1, help='src1 train num frames')
    parser.add_argument('--tgt_test_num_frames', type=int, default=2, help='tgt test num frames')
    parser.add_argument('--resume', type=bool, default=False, help='resume training')
    config = parser.parse_args()
    checkpoint_path = './' + config.src1_data + '2' + config.tgt_data + '_checkpoint/' + config.model + '/DGNet/'
    best_model_path = './' + config.src1_data + '2' + config.tgt_data + '_checkpoint/' + config.model + '/best_model/'
    logs = './logs_' + config.model + '/'
    visual_path = './visual_' + config.model + '/'


    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.cuda.manual_seed(config.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = 'cuda'

    train(is_test=True)
