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

def sequence_to_batch(input_image, len_seq):
    input_image_split = input_image.chunk(len_seq, 1)
    input_image = torch.cat(input_image_split, 0)
    return input_image


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

class FaceMapNet_v2(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, hidden_dim=64):
        super(FaceMapNet_v2, self).__init__()
        self.conv0 = Conv2d_res(in_planes=in_channels, out_planes=hidden_dim, kernel_size=(3, 3), stride=(1, 1))

        self.conv2_1 = Conv2d_res(in_planes=hidden_dim, out_planes=hidden_dim*2, kernel_size=(3, 3), stride=(1, 1))
        self.conv2_2 = Conv2d_res(in_planes=hidden_dim*2, out_planes=hidden_dim*3, kernel_size=(3, 3), stride=(1, 1))
        self.conv2_3 = Conv2d_res(in_planes=hidden_dim*3, out_planes=hidden_dim*2, kernel_size=(3, 3), stride=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3_1 = Conv2d_res(in_planes=hidden_dim*2, out_planes=hidden_dim*2, kernel_size=(3, 3), stride=(1, 1))
        self.conv3_2 = Conv2d_res(in_planes=hidden_dim*2, out_planes=hidden_dim*3, kernel_size=(3, 3), stride=(1, 1))
        self.conv3_3 = Conv2d_res(in_planes=hidden_dim*3, out_planes=hidden_dim*2, kernel_size=(3, 3), stride=(1, 1))
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv4_1 = Conv2d_res(in_planes=hidden_dim*4, out_planes=hidden_dim*2,kernel_size=(3, 3), stride=(1, 1))
        self.conv4_2 = Conv2d_res(in_planes=hidden_dim*2, out_planes=hidden_dim, kernel_size=(3, 3), stride=(1, 1))
        self.conv4_3 = nn.Sequential(
            basic_conv(in_planes=hidden_dim, out_planes=out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[2]
        pre_off_list = []
        x_1 = self.conv0(x)  # [2, 64, 128, 128]

        x_2 = self.conv2_3(self.conv2_2(self.conv2_1(x_1)))   # [2, 128, 64, 64]
        pre_off_list.append(x_2)
        x_2 = self.pool(x_2)
        x_3 = self.conv3_3(self.conv3_2(self.conv3_1(x_2)))   # [2, 128, 32, 32]
        pre_off_list.append(x_3)
        x_3 = self.pool(x_3)
        with torch.no_grad():
            x_2c = x_2.clone()
            feature_2 = x_2c.resize_(x_2c.shape[0], x_2c.shape[1], size//4, size//4)
        pool_concat = torch.cat([feature_2, x_3], 1)
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


class OFFNet_v2(nn.Module):
    # input: [batch*len_seq, channel, height, weight]
    def __init__(self, input_channel=128, len_seq=2, reduce_num=32):
        super(OFFNet_v2, self).__init__()
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
        # net1 = self.off_block(pre_off_list)  # 128,128
        # net2 = self.off_block(pre_off_list)  # 64,64
        # net3 = pre_off_list
        net1 = self.off_block(pre_off_list[0])  # [batch*seq, channel, h, w]  64,64
        net2 = self.off_block(pre_off_list[1])  # 32,32

        with torch.no_grad():
            net1_c = net1.clone()
            feature_1 = net1_c.resize_(net1_c.shape[0], net1_c.shape[1], 32, 32)  # 64,64

        net_concat = torch.cat([feature_1, net2], 1)
        net2 = self.conv2(net_concat)

        pool_concat = net2

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


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class SoftmaxNetSub(nn.Module):
    def __init__(self):
        super(SoftmaxNetSub, self).__init__()
        self.fc1 = nn.Linear(1024, 64)
        self.fc1.weight.data.normal_(0, 0.01)  # initialize the weights of fc1, normal distribution~(mean=0,var=0.01)
        self.fc1.bias.data.fill_(0.0)  # initialize the bias of fc1 as 0
        self.fc2 = nn.Linear(64, 2)
        self.fc2.weight.data.normal_(0, 0.3)  # initialize the weights of fc2, normal distribution~(mean=0,var=0.3)
        self.fc2.bias.data.fill_(0.0)  # initialize the bias of fc2 as 0
        self.fc = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc2
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        input = input.reshape(input.size(0), -1)
        fc_ = self.fc(input)
        fc_ = self.softmax(fc_)
        return fc_


def build_loss(logits_map, maps=None, label=None, len_seq=6):
    batch_size = logits_map.shape[0]
    # logits_cla = get_logtis_cla_from_logits_list(logits_map)   # logits_map [batchsize, 1, 32, 32]
    # logits_map = logits  # [batch, 1, heigth, width]
    logits_cla = torch.mean(logits_map, dim=1)
    logits_cla = torch.mean(logits_cla, dim=-1)
    logits_cla = torch.mean(logits_cla, dim=-1, keepdim=True)
    logits_cla_ = torch.cat([logits_cla, 1 - logits_cla], 1)
    logits = logits_cla_[:,0]  # logits_cla  [batch, 2]

    logits = logits.unsqueeze(-1)
    label = label.float()
    label = label.unsqueeze(-1)  # [batch, 1]

    label_s = torch.zeros((batch_size, 1),device='cuda')

    for i in range(batch_size):
        label_s[i] = label[len_seq*i]

    # print('logits-shape:', logits.shape)
    loss_cla = nn.BCEWithLogitsLoss()(logits, label_s)

    maps_reg = maps / 255.0
    logits_map_expand = torch.zeros((len_seq*batch_size, 1, logits_map.shape[2], logits_map.shape[3]),device='cuda')

    for j in range(batch_size):
        for i in range(len_seq):
            logits_map_expand[j*len_seq+i] = logits_map[j]

    loss_depth_1 = (logits_map_expand - maps_reg) ** 2
    loss_depth_1 = torch.mean(loss_depth_1)
    loss_depth_2 = contrast_depth_loss(logits_map_expand, maps_reg)
    loss_depth = loss_depth_1 + 0.5 * loss_depth_2

    loss = loss_cla + 0.5 * loss_depth
    acc = accuracy(logits, label_s, topk=(1,))
    # print('depth_loss:{:.4f}'.format(loss_depth.item()), 'cls_loss:{:.4f}'.format(loss_cla.item()),
    #       'map_reg_loss:{:.4f}'.format(loss_depth_1.item()), 'contrast_depth_loss:{:.4f}'.format(loss_depth_2.item()))
    return loss, loss_cla, acc, logits_cla, loss_depth_1, loss_depth_2

def build_loss_single(logits_map, maps=None, label=None, len_seq=6):
    batch_size = logits_map.shape[0]
    # logits_cla = get_logtis_cla_from_logits_list(logits_map)   # logits_map [batchsize, 1, 32, 32]
    # logits_map = logits  # [batch, 1, heigth, width]
    logits_cla = torch.mean(logits_map, dim=1)
    logits_cla = torch.mean(logits_cla, dim=-1)
    logits_cla = torch.mean(logits_cla, dim=-1, keepdim=True)
    # print('logits_cla:', logits_cla)
    logits_cla_ = torch.cat([logits_cla, 1 - logits_cla], 1)
    logits = logits_cla_[:,0]  # logits_cla  [batch, 2]

    logits = logits.unsqueeze(-1)
    label = label.float()
    label = label.unsqueeze(-1)  # [batch, 1]

    loss_cla = nn.BCEWithLogitsLoss()(logits, label)

    maps_reg = maps / 255.0

    loss_depth_1 = (logits_map - maps_reg) ** 2
    loss_depth_1 = torch.mean(loss_depth_1)
    loss_depth_2 = contrast_depth_loss(logits_map, maps_reg)
    loss_depth = loss_depth_1 + 0.5 * loss_depth_2

    loss = loss_cla + 0.5 * loss_depth
    acc = accuracy(logits, label, topk=(1,))
    # print('depth_loss:{:.4f}'.format(loss_depth.item()), 'cls_loss:{:.4f}'.format(loss_cla.item()),
    #       'map_reg_loss:{:.4f}'.format(loss_depth_1.item()), 'contrast_depth_loss:{:.4f}'.format(loss_depth_2.item()))
    return loss, loss_cla, acc, logits_cla, loss_depth_1, loss_depth_2


class SGTDNet(nn.Module):
    def __init__(self, len_seq=2, hidden_dim=64, reduce_num=32):
        super(SGTDNet, self).__init__()
        self.len_seq = len_seq
        self.FaceMapModel = FaceMapNet_v2(hidden_dim=hidden_dim)
        self.OFFModel = OFFNet_v2(input_channel=hidden_dim*2, len_seq=len_seq, reduce_num=reduce_num)
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


def get_logtis_cla_from_logits_list(logits_map_concat):
    # logits_map_concat = torch.cat(logits_list, 1)
    # logits_map_concat = torch.mean(logits_map_concat, 1)
    logits_cla = SoftmaxNetSub()(logits_map_concat)
    return logits_cla


if __name__ == '__main__':
    import os
    gpus = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SGTD = FaceMapNet_v2(hidden_dim=32).to(device)
    # SGTD = OFFNet_v2().to(device)

    summary(SGTD, (3, 128, 128), batch_size=24, device='cuda')
