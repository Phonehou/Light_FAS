import torch
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image

def plot(input,label,save_path, epoch, net_name, desc):   # input   [channel, height, width ]
    # cv2.imwrite(save_path + '/' + str(epoch) + '_original_'+str(label.item())+desc, input)
    heatmap = torch.zeros(input.size(1), input.size(2))
    for i in range(input.size(0)):
        heatmap += torch.pow(input[i, :, :], 2).view(input.size(1),input.size(2))

    heatmap = heatmap.data.numpy()
    title = 'Real-1,Attack-0/Label:' + ' ' + str(label.item())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.title(title)
    plt.colorbar()
    plt.savefig(save_path + '/' + str(epoch) + '_' + net_name +'_'+str(label.item())+desc)
    plt.close()

# feature  -->   [ batch, channel, height, width ]
def FeatureMap2Heatmap(x_input, x_150, x_18, save_path, net_name, label, epoch, mini_batch):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if mini_batch >= x_input.size(0):
        mini_batch = x_input.size(0) // 2
    input_first_frame = x_input[0, :, :, :].cpu()  ## the first real frame
    input_second_frame = x_input[mini_batch, :, :, :].cpu()  ## the first fake frame
    x150_first_frame = x_150[0, :, :, :].cpu()  ## the first real frame
    x150_second_frame = x_150[mini_batch, :, :, :].cpu()  ## the first fake frame
    x18_first_frame = x_18[0, :, :, :].cpu()  ## the first real frame
    x18_second_frame = x_18[mini_batch, :, :, :].cpu()  ## the first fake frame

    inputs = [input_first_frame,input_second_frame,x150_first_frame,x150_second_frame,x18_first_frame,x18_second_frame]
    label  = [label[0],label[mini_batch],label[0],label[mini_batch],label[0],label[mini_batch]]
    desc = ['_x_input.jpg','_x_input.jpg','_x_150.jpg','_x_150.jpg','_x_18.jpg','_x_18.jpg']

    for i,input_ in enumerate(inputs):
        plot(input_, label[i], save_path, epoch, net_name, desc[i])

# feature  -->   [ batch, channel, height, width ]
def CueMap2Heatmap(x_input, mask, cue, save_path, net_name, label, epoch, mini_batch):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if mini_batch >= x_input.size(0):
        mini_batch = x_input.size(0) // 2
    input_first_frame = x_input[0, :, :, :].cpu()  ## the first real frame
    input_second_frame = x_input[mini_batch, :, :, :].cpu()  ## the first fake frame
    mask_first_frame = mask[0, :, :, :].cpu()  ## the first real frame
    mask_second_frame = mask[mini_batch, :, :, :].cpu()  ## the first fake frame
    cue_first_frame = cue[0, :, :, :].cpu()  ## the first real frame
    cue_second_frame = cue[mini_batch, :, :, :].cpu()  ## the first fake frame

    inputs = [input_first_frame,input_second_frame,mask_first_frame,mask_second_frame,cue_first_frame,cue_second_frame]
    label = [label[0],label[mini_batch],label[0],label[mini_batch],label[0],label[mini_batch]]
    desc = ['_x_input.jpg','_x_input.jpg','_mask.jpg','_mask_jpg','_cue.jpg','_x_cue.jpg']

    for i,input_ in enumerate(inputs):
        plot(input_, label[i], save_path, epoch, net_name, desc[i])