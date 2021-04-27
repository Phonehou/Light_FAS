import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.nn import Parameter

import numpy as np
from torchsummary import summary
from utils.utils import accuracy

def basic_conv(in_planes, out_planes, kernel_size=(3, 3), stride=(1, 1), padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=False)

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

def spatial_gradient_x(input):
    # input: Tensor
    n, c, h, w = input.shape
    sobel_plane_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_plane_x = np.expand_dims(sobel_plane_x, axis=0)
    # sobel_plane_x = np.repeat(sobel_plane_x, input.shape[1], axis=0)
    sobel_plane_x = np.expand_dims(sobel_plane_x, axis=0)
    sobel_plane_x = np.repeat(sobel_plane_x, c, axis=0)

    # sobel_kernel_x = tf.constant(sobel_plane_x, dtype=tf.float32)
    sobel_plane_x = torch.Tensor(sobel_plane_x).to('cuda')
    weight = nn.Parameter(data=sobel_plane_x, requires_grad=False)
    # F.conv2d
    # input [minibatch, in_ch, i_H, i_W]
    # weight [out_ch, in_ch/groups, kH, kW]
    Spatial_Gradient_x = F.conv2d(input, weight, stride=[1, 1], padding=1, groups=c)
    # Spatial_Gradient_x = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
    # Spatial_Gradient_x = tf.nn.depthwise_conv2d(intput, filter=sobel_kernel_x, \
    #                                             strides=[1, 1, 1, 1], padding='SAME', name=name + '/spatial_gradient_x')
    return Spatial_Gradient_x

def spatial_gradient_y(input):
    n, c, h, w = input.shape
    sobel_plane_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_plane_y = np.expand_dims(sobel_plane_y, axis=0)
    sobel_plane_y = np.expand_dims(sobel_plane_y, axis=0)
    sobel_plane_y = np.repeat(sobel_plane_y, c, axis=0)

    sobel_plane_y = torch.Tensor(sobel_plane_y).to('cuda')
    weight = nn.Parameter(data=sobel_plane_y, requires_grad=False)
    # F.conv2d
    # input [minibatch, in_ch, i_H, i_W]
    # weight [out_ch, in_ch/groups, kH, kW]
    Spatial_Gradient_y = F.conv2d(input, weight, stride=[1, 1], padding=1, groups=c)
    return Spatial_Gradient_y

class Conv2d_res(nn.Module):
    def __init__(self, in_planes=3, out_planes=5, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_res, self).__init__()
        # self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
        #                       dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
        # self.pointwise = nn.Conv2d(in_planes, out_planes, 1, 1, 0, 1, 1, bias=bias)
        self.conv = nn.Sequential(
            basic_conv(in_planes=in_planes*2, out_planes=out_planes, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_planes),
            # nn.ReLU(),
        )

    def forward(self, x):
        # out_normal = self.conv(x)
        out_normal = x
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            # [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            # kernel_diff = self.conv.weight.sum(2).sum(2)
            # kernel_diff = kernel_diff[:, :, None, None]
            # out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
            #                     groups=self.conv.groups)
            gradient_x = spatial_gradient_x(x)
            gradient_y = spatial_gradient_y(x)
            div_x = gradient_x/(x+0.001)
            div_y = gradient_y/(x+0.001)
            div_x = div_x.pow(2)
            div_y = div_y.pow(2)
            # gradient_gabor = torch.sqrt(gradient_x**2+gradient_y**2)
            gradient_gabor = torch.atan(div_x + div_y)
            # gradient_gabor = gradient_gabor)
            # gradient_gabor_pw = self.conv(gradient_gabor)
            res = torch.cat((out_normal, self.theta * gradient_gabor), dim=1)  # [:,6,224,224]
            return self.conv(res)
            # return out_normal + self.theta * gradient_gabor_pw

class FaceMapNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, hidden_dim=64):
        super(FaceMapNet, self).__init__()
        self.conv0 = Conv2d_res(in_planes=in_channels, out_planes=hidden_dim, kernel_size=(3, 3), stride=(1, 1))

        self.conv1_1 = Conv2d_res(in_planes=hidden_dim, out_planes=hidden_dim*2, kernel_size=(3, 3), stride=(1, 1))
        self.conv1_2 = Conv2d_res(in_planes=hidden_dim*2, out_planes=hidden_dim*3, kernel_size=(3, 3), stride=(1, 1))
        self.conv1_3 = Conv2d_res(in_planes=hidden_dim*3, out_planes=hidden_dim*2, kernel_size=(3, 3), stride=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2_1 = Conv2d_res(in_planes=hidden_dim*2, out_planes=hidden_dim*2, kernel_size=(3, 3), stride=(1, 1))
        self.conv2_2 = Conv2d_res(in_planes=hidden_dim*2, out_planes=hidden_dim*3, kernel_size=(3, 3), stride=(1, 1))
        self.conv2_3 = Conv2d_res(in_planes=hidden_dim*3, out_planes=hidden_dim*2, kernel_size=(3, 3), stride=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3_1 = Conv2d_res(in_planes=hidden_dim*2, out_planes=hidden_dim*2, kernel_size=(3, 3), stride=(1, 1))
        self.conv3_2 = Conv2d_res(in_planes=hidden_dim*2, out_planes=hidden_dim*3, kernel_size=(3, 3), stride=(1, 1))
        self.conv3_3 = Conv2d_res(in_planes=hidden_dim*3, out_planes=hidden_dim*2, kernel_size=(3, 3), stride=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv4_1 = Conv2d_res(in_planes=hidden_dim*6, out_planes=hidden_dim*2,kernel_size=(3, 3), stride=(1, 1))
        self.conv4_2 = Conv2d_res(in_planes=hidden_dim*2, out_planes=hidden_dim, kernel_size=(3, 3), stride=(1, 1))
        self.conv4_3 = nn.Sequential(
            basic_conv(in_planes=hidden_dim, out_planes=out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[2]
        pre_off_list = []
        x = self.conv0(x)  # [2, 64, 256, 256]
        x_1 = self.conv1_3(self.conv1_2(self.conv1_1(x)))   # [2,128,128,128]
        pre_off_list.append(x_1)
        x_1 = self.pool1(x_1)
        x_2 = self.conv2_3(self.conv2_2(self.conv2_1(x_1)))   # [2, 128, 64, 64]
        pre_off_list.append(x_2)
        x_2 = self.pool1(x_2)
        x_3 = self.conv3_3(self.conv3_2(self.conv3_1(x_2)))   # [2, 128, 32, 32]
        pre_off_list.append(x_3)
        x_3 = self.pool1(x_3)
        with torch.no_grad():
            x_1c = x_1.clone()
            x_2c = x_2.clone()
            feature_1 = x_1c.resize_(x_1c.shape[0], x_1c.shape[1], size//8, size//8)
            feature_2 = x_2c.resize_(x_1c.shape[0], x_1c.shape[1], size//8, size//8)
        pool_concat = torch.cat([feature_1, feature_2, x_3], 1)
        net = self.conv4_2(self.conv4_1(pool_concat))
        net = self.conv4_3(net)
        return net, pre_off_list

# handle the information between the frame sequence
class OFFNet(nn.Module):
    # input: [batch*len_seq, channel, height, weight]
    def __init__(self, input_channel=128, len_seq=2, reduce_num=32):
        super(OFFNet, self).__init__()
        self.reduce_num = reduce_num
        self.len_seq = len_seq
        self.conv1x1 = nn.Sequential(basic_conv(input_channel, reduce_num, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(reduce_num),
                                     # nn.ReLU()
                                     )
        self.conv3x3 = basic_conv(reduce_num*5+input_channel, input_channel, kernel_size=3, stride=1)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Sequential(basic_conv(input_channel*2, input_channel, kernel_size=3, stride=1),
                                     nn.BatchNorm2d(input_channel),
                                     # nn.ReLU()
                                   )
        self.conv3 = nn.Sequential(basic_conv(input_channel*2, input_channel, kernel_size=3, stride=1),
                                   nn.BatchNorm2d(input_channel),
                                   # nn.ReLU()
                                   )
        self.conv4 = nn.Sequential(basic_conv(input_channel, input_channel, kernel_size=3, stride=1),
                                   nn.BatchNorm2d(input_channel),
                                   # nn.ReLU()
                                   )

    def off_block(self, pre_off_feature):

        feature_shape = pre_off_feature.shape
        net = self.conv1x1(pre_off_feature)  # net [batch_size*len_seq, reduce_num, h, w]
        net_reshape = net.reshape(-1, self.len_seq, self.reduce_num, feature_shape[2], feature_shape[3])
        Spatial_Gradient_x = spatial_gradient_x(net)
        Spatial_Gradient_x = Spatial_Gradient_x.reshape(-1, self.len_seq, self.reduce_num, feature_shape[2], feature_shape[3])
        Spatial_Gradient_y = spatial_gradient_y(net)
        Spatial_Gradient_y = Spatial_Gradient_y.reshape(-1, self.len_seq, self.reduce_num, feature_shape[2], feature_shape[3])
        Temporal_Gradient = net_reshape[:, :-1, :, :, :] - net_reshape[:, 1:, :, :, :]

        pre_off_feature_squence = pre_off_feature.reshape(-1, self.len_seq, feature_shape[1], feature_shape[2], feature_shape[3])

        single_feature_scale = 1.0
        off_feature_squence = torch.cat([
            pre_off_feature_squence[:, :-1, :, :, :] * single_feature_scale, \
            Spatial_Gradient_x[:, :-1, :, :, :], \
            Spatial_Gradient_y[:, :-1, :, :, :], \
            Spatial_Gradient_x[:, 1:, :, :, :], \
            Spatial_Gradient_y[:, 1:, :, :, :], \
            Temporal_Gradient], 2)
        # off_feature [batch*seq, new_channel=, h, w]
        off_feature_batch = off_feature_squence.reshape(-1, off_feature_squence.shape[2], feature_shape[2], feature_shape[3])
        res_feature = self.maxPool(self.conv3x3(off_feature_batch))
        return res_feature

    def forward(self, pre_off_list):
        # pre_off_list [batch*seq, channel, h, w]
        net1 = self.off_block(pre_off_list[0])  # [batch*seq, channel, h, w]
        net2 = self.off_block(pre_off_list[1])
        net3 = self.off_block(pre_off_list[2])
        # net1 = pre_off_list
        # net2 = pre_off_list
        # net3 = pre_off_list
        with torch.no_grad():
            net1_c = net1.clone()
            feature_1 = net1_c.resize_(net1_c.shape[0], net1_c.shape[1], 64, 64)  # 64,64

        net_concat = torch.cat([feature_1, net2], 1)
        net2 = self.conv2(net_concat)
        with torch.no_grad():
            net2_c = net2.clone()
            feature_2 = net2_c.resize_(net2_c.shape[0], net2_c.shape[1], 32, 32)

        net_concat = torch.cat([feature_2, net3], 1)
        net3 = self.conv3(net_concat)

        pool_concat = net3

        net = self.conv4(pool_concat)  # [batch*(seq-1),channel, h,w]
        net = net.reshape(-1, self.len_seq - 1, net.shape[1], net.shape[2], net.shape[3])
        return net

def contrast_depth_conv(input, dilation_rate=1):
    ''' compute contrast depth in both of (out, label) '''
    # print(input.shape)
    assert (input.shape[1] == 1)

    kernel_filter_list = [
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]], [[0, 1, 0], [0, -1, 0], [0, 0, 0]], [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [1, -1, 0], [0, 0, 0]], [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, -1, 0], [1, 0, 0]], [[0, 0, 0], [0, -1, 0], [0, 1, 0]], [[0, 0, 0], [0, -1, 0], [0, 0, 1]]
    ]
    kernel_filter = np.array(kernel_filter_list, np.float32)
    kernel_filter = np.expand_dims(kernel_filter, axis=1)  # shape [8,1,3,3]
    # kernel_filter = kernel_filter.transpose([1, 2, 3, 0])
    kernel_filter = torch.Tensor(kernel_filter).to('cuda')
    kernel_filter_tf = nn.Parameter(data=kernel_filter, requires_grad=False)

    if dilation_rate == 1:
        # contrast_depth = tf.nn.conv2d(input, kernel_filter_tf, strides=[1, 1, 1, 1], padding='SAME', name=op_name)
        contrast_depth = F.conv2d(input, kernel_filter_tf, stride=[1, 1], padding=1)
    # else:
    #     contrast_depth = tf.nn.atrous_conv2d(input, kernel_filter_tf, \
    #                                          rate=dilation_rate, padding='SAME', name=op_name)
    return contrast_depth

def contrast_depth_loss(out, label):
    '''
    compute contrast depth in both of (out, label),
    then get the loss of them
    tf.atrous_convd match tf-versions: 1.4
    '''
    contrast_out = contrast_depth_conv(out, 1)
    contrast_label = contrast_depth_conv(label, 1)

    loss = (contrast_out - contrast_label)**2
    loss = torch.mean(loss)

    return loss

class SGTDNet(nn.Module):
    def __init__(self, len_seq=2, hidden_dim=64, reduce_num=32):
        super(SGTDNet, self).__init__()
        self.len_seq = len_seq
        self.FaceMapModel = FaceMapNet(hidden_dim=hidden_dim)
        self.OFFModel = OFFNet(input_channel=hidden_dim*2, len_seq=len_seq, reduce_num=reduce_num)
        self.ConvLSTMNet = ConvLSTM(hidden_dim*2, [16,1], [(3,3),(3,3)], 2, True, True, False)

    def forward(self, input_image):
        # input images [batch*seq, channels, height, width]
        # input_tensor = sequence_to_batch(input_image)  # [batch*seq, channels, height, width]
        features_map, pre_off_list = self.FaceMapModel(input_image)  # feathres_map [batch*seq, 1, 32, 32]
        off_feature = self.OFFModel(pre_off_list) # [batch, seq-1, channels, height, width]
        logits_map_, _ = self.ConvLSTMNet(off_feature)   # [(batch,len=len_seq-1,1,32,32)]
        logits_map_split = logits_map_[0]
        # logits_map_split = logits_map_split.reshape(-1, logits_map_split.shape[2],logits_map_split.shape[3],logits_map_split.shape[4],)
        single_maps = features_map.reshape(-1, self.len_seq, logits_map_split.shape[2], logits_map_split.shape[3],logits_map_split.shape[4])
        single_ratio = 0.4
        beta = single_ratio
        alpha = 1 - beta
        logits_list = logits_map_split
        for i in range(self.len_seq-1):
            logits_list[:, i] = alpha * logits_map_split[:, i] + beta * single_maps[:, i]

        return logits_list


# def get_logtis_cla_from_logits_list(logits_map_concat):
#     # logits_map_concat = torch.cat(logits_list, 1)
#     # logits_map_concat = torch.mean(logits_map_concat, 1)
#     logits_cla = SoftmaxNetSub()(logits_map_concat)
#     return logits_cla


if __name__ == '__main__':
    import os
    gpus = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SGTD = FaceMapNet(hidden_dim=32).to(device)
    # SGTD = OFFNet_v2().to(device)

    summary(SGTD, (3, 256, 256), batch_size=24, device='cuda')
