import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
import numpy as np
from torchsummary import summary
from utils.utils import accuracy
from utils.dataset import Normalization255_GRAY

def normalize(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    return tensor

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
    # sobel_plane_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_plane_x = np.array([[-1 / (2 * np.sqrt(2)), 0, 1 / (2 * np.sqrt(2))],
              [-1, 0, 1],
              [-1 / (2 * np.sqrt(2)), 0, 1 / (2 * np.sqrt(2))]])
    # sobel_plane_x = np.array([
    #         [np.cos(6 / 8 * np.pi) / 8, np.cos(5 / 8 * np.pi) / 5, np.cos(4 / 8 * np.pi) / 4,
    #          np.cos(3 / 8 * np.pi) / 5, np.cos(2 / 8 * np.pi) / 8],
    #         [np.cos(7 / 8 * np.pi) / 5, -1 / (2 * np.sqrt(2)), 0, 1 / (2 * np.sqrt(2)), np.cos(1 / 8 * np.pi) / 5],
    #         [np.cos(8 / 8 * np.pi) / 4, -1, 0, 1, np.cos(0 / 8 * np.pi) / 4],
    #         [np.cos(9 / 8 * np.pi) / 5, -1 / (2 * np.sqrt(2)), 0, 1 / (2 * np.sqrt(2)), np.cos(15 / 8 * np.pi) / 5],
    #         [np.cos(10 / 8 * np.pi ) / 8, np.cos(11 / 8 * np.pi) / 5, np.cos(12 / 8 * np.pi) / 4, np.cos(13 / 8 * np.pi) / 5,
    #          np.cos(14 / 8 * np.pi) / 8]])
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
    # sobel_plane_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_plane_y = np.array([[1 / (2 * np.sqrt(2)), 1, 1 / (2 * np.sqrt(2))],
                   [0, 0, 0],
                   [-1 / (2 * np.sqrt(2)), -1, -1 / (2 * np.sqrt(2))]])
    # sobel_plane_y = np.array([
    #         [np.sin(6 / 8 * np.pi) / 8, np.sin(5 / 8 * np.pi) / 5, np.sin(4 / 8 * np.pi) / 4,
    #          np.sin(3 / 8 * np.pi) / 5, np.sin(2 / 8 * np.pi) / 8],
    #         [np.sin(7 / 8 * np.pi) / 5, 1 / (2 * np.sqrt(2)), 1, 1 / (2 * np.sqrt(2)), np.sin(1 / 8 * np.pi) / 5],
    #         [np.sin(8 / 8 * np.pi) / 4, 0, 0, 0, np.sin(0 / 8 * np.pi) / 4],
    #         [np.sin(9 / 8 * np.pi) / 5,-1 / (2 * np.sqrt(2)), -1, -1 / (2 * np.sqrt(2)), np.sin(15 / 8 * np.pi) / 5],
    #         [np.sin(10 / 8 * np.pi ) / 8, np.sin(11 / 8 * np.pi) / 5, np.sin(12 / 8 * np.pi) / 4, np.sin(13 / 8 * np.pi) / 5,
    #          np.sin(14 / 8 * np.pi) / 8]])

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
            gradient_x = spatial_gradient_x(x)
            gradient_y = spatial_gradient_y(x)
            div_x = gradient_x/(x+0.001)
            div_y = gradient_y/(x+0.001)
            div_x = div_x.pow(2)
            div_y = div_y.pow(2)
            # gradient_gabor = torch.sqrt(gradient_x**2+gradient_y**2)
            # gradient_gabor = torch.atan(div_x + div_y)
            div_xnp = div_x.cpu().data.numpy()
            div_ynp = div_y.cpu().data.numpy()
            gradient_gabor = np.sqrt(div_xnp + div_ynp)
            gradient_gabor = np.arctan(gradient_gabor)  # numpy array
            gradient_gabor = torch.tensor(gradient_gabor).cuda()
            gradient_gabor = gradient_gabor / 255
            gradient_gabor = normalize(gradient_gabor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # gradient_gabor = gradient_gabor)
            # gradient_gabor_pw = self.conv(gradient_gabor)
            # res = torch.cat((out_normal, self.theta * gradient_gabor), dim=1)  # [:,6,224,224]
            # return self.conv(res)
            # return out_normal + self.theta * gradient_gabor_pw
            return gradient_gabor, out_normal

class Conv2d_res_v2(nn.Module):
    def __init__(self, in_planes=3, out_planes=5, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_res_v2, self).__init__()
        # self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
        #                       dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
        # self.pointwise = nn.Conv2d(in_planes, out_planes, 1, 1, 0, 1, 1, bias=bias)
        # self.conv = nn.Sequential(
        #     basic_conv(in_planes=in_planes, out_planes=out_planes, kernel_size=kernel_size, stride=stride),
        #     nn.BatchNorm2d(out_planes),
        #     nn.ReLU(),
        # )
        self.conv = basic_conv(in_planes=in_planes, out_planes=out_planes, kernel_size=kernel_size, stride=stride)


    def forward(self, x):
        # out_normal = self.conv(x)
        out_normal = x
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            gradient_x = spatial_gradient_x(x)
            gradient_y = spatial_gradient_y(x)
            div_x = gradient_x/(x+0.001)
            div_y = gradient_y/(x+0.001)
            div_x = div_x.pow(2)
            div_y = div_y.pow(2)
            div_xnp = div_x.cpu().data.numpy()
            div_ynp = div_y.cpu().data.numpy()
            gradient_gabor = np.sqrt(div_xnp+div_ynp)
            gradient_gabor = np.arctan(gradient_gabor)   # numpy array
            gradient_gabor = torch.tensor(gradient_gabor).cuda()
            Max_ = torch.max(gradient_gabor,2)
            Min_ = torch.min(gradient_gabor,2)
            Max = torch.max(Max_[0],2)
            Min = torch.min(Min_[0],2)
            Max = Max[0].unsqueeze(-1).unsqueeze(-1)
            Max = Max.repeat(1,1,gradient_gabor.shape[-2], gradient_gabor.shape[-1])
            Min = Min[0].unsqueeze(-1).unsqueeze(-1)
            Min = Min.repeat(1, 1, gradient_gabor.shape[-2], gradient_gabor.shape[-1])
            gradient_gabor_n = ((gradient_gabor - Min) / (Max - Min)) * 254 + 1
            # return self.conv(gradient_gabor)
            gradient_gabor_n = torch.cat((out_normal, gradient_gabor_n), dim=1)
            gradient_gabor_n = gradient_gabor_n / 255
            # gradient_gabor_n = normalize(gradient_gabor_n, mean=[0.485, 0.456, 0.406],
            #             std=[0.229, 0.224, 0.225])
            gradient_gabor_n = normalize(gradient_gabor_n, mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225])

            return gradient_gabor_n, x

class Conv2d_res_v3(nn.Module):
    def __init__(self, in_planes=3, out_planes=5, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_res_v3, self).__init__()
        # self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
        #                       dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
        self.pointwise = nn.Conv2d(in_planes, in_planes, 1, 1, 0, 1, 1, bias=bias)
        self.conv = nn.Sequential(
            basic_conv(in_planes=in_planes*2, out_planes=out_planes, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_planes),
            # nn.ReLU(),
        )
        m_ = torch.FloatTensor(3, 224, 224).fill_(1.0).to('cuda')
        self.mask = nn.Parameter(data=m_, requires_grad=True)

    def forward(self, x):
        # out_normal = self.pointwise(x)
        out_normal = x[:] * self.mask
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            gradient_x = spatial_gradient_x(x)
            gradient_y = spatial_gradient_y(x)
            div_x = gradient_x/(x+0.001)
            div_y = gradient_y/(x+0.001)
            div_x = div_x.pow(2)
            div_y = div_y.pow(2)
            # gradient_gabor = torch.sqrt(gradient_x**2+gradient_y**2)
            # gradient_gabor = torch.atan(div_x + div_y)
            div_xnp = div_x.cpu().data.numpy()
            div_ynp = div_y.cpu().data.numpy()
            gradient_gabor = np.sqrt(div_xnp + div_ynp)
            gradient_gabor = np.arctan(gradient_gabor)  # numpy array
            gradient_gabor = torch.tensor(gradient_gabor).cuda()
            # gradient_gabor = gradient_gabor / 255
            # gradient_gabor = normalize(gradient_gabor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            # gradient_gabor_pw = self.conv(gradient_gabor)
            gradient_gabor = torch.cat((out_normal, gradient_gabor), dim=1)  # [:,6,224,224]
            # gradient_gabor = torch.add(out_normal, gradient_gabor)
            return gradient_gabor, x
            # return out_normal + self.theta * gradient_gabor_pw
            # return gradient_gabor, out_normal

def basic_conv(in_planes, out_planes, kernel_size=(3, 3), stride=(1, 1)):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=1, bias=False)

class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class GRL(torch.autograd.Function):
    def __init__(self):
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 4000  # be same to the max_iter of config.py

    @staticmethod
    def forward(self, input):
        self.iter_num += 1
        return input * 1.0

    @staticmethod
    def backward(self, gradOutput):
        coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter))
                         - (self.high - self.low) + self.low)
        return -coeff * gradOutput

class Discriminator(nn.Module):
    def __init__(self, out_channels=2):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(1568, 1024)
        self.fc1.weight.data.normal_(0, 0.01)    # initialize the weights of fc1, normal distribution~(mean=0,var=0.01)
        self.fc1.bias.data.fill_(0.0)            # initialize the bias of fc1 as 0
        self.fc2 = nn.Linear(1024, out_channels)
        self.fc2.weight.data.normal_(0, 0.3)    # initialize the weights of fc2, normal distribution~(mean=0,var=0.3)
        self.fc2.bias.data.fill_(0.0)             # initialize the bias of fc2 as 0
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc2
        )
        self.grl_layer = GRL()

    def forward(self, feature):
        # adversarial_out = self.ad_net(self.grl_layer(feature))
        adversarial_out = -0.5*self.ad_net(feature)
        return adversarial_out

class SpoofNet(nn.Module):
    def __init__(self,in_channels=3):
        super(SpoofNet, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Sequential(
            basic_conv(in_planes=self.in_channels,out_planes=16,kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=9, stride=8, padding=1)
        self.conv2 = nn.Sequential(
            basic_conv(in_planes=16, out_planes=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=6, stride=4, padding=1)
        self.dp_out1 = nn.Dropout(p=0.25)

        self.conv3 = nn.Sequential(
            basic_conv(in_planes=32, out_planes=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            basic_conv(in_planes=16, out_planes=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

        )
        self.dp_out2 = nn.Dropout(p=0.25)

        self.fc = nn.Sequential(
            nn.Linear(1568, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 2)
        )

        self.mlp = nn.Sequential(
            nn.Linear(1568, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self,x_input):
        x_150 = self.conv1(x_input)
        x_18 = self.pool1(x_150)
        x_18 = self.conv2(x_18)
        x_2 = self.pool2(x_18)

        x_2 = self.dp_out1(x_2)
        x3 = self.conv3(x_2)
        x3 = self.conv4(x3)
        x_concat = torch.add(x3, x_2)
        x4 = self.dp_out2(x_concat)
        x4 = x4.reshape(x4.size(0), -1)
        # feature = self.mlp(x4)
        out = self.fc(x4)
        # x_150 16*224*224  x_18 16*28*28  x4 1568  out 2
        return x_150, x_18, x4, out

class SpoofNet_v2(nn.Module):
    def __init__(self,in_channels=3):
        super(SpoofNet_v2, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Sequential(
            basic_conv(in_planes=self.in_channels,out_planes=16,kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=9, stride=8, padding=1)
        self.conv2 = nn.Sequential(
            mfm(16, 16, 3, 1,1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=6, stride=4, padding=1)
        self.dp_out1 = nn.Dropout(p=0.25)

        self.conv3 = nn.Sequential(
            basic_conv(in_planes=16, out_planes=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            mfm(16, 16, 3, 1,1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

        )
        self.dp_out2 = nn.Dropout(p=0.25)

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Linear(512, 128),
            nn.Linear(128, 2)
        )


    def forward(self,x_input):
        x_150 = self.conv1(x_input)
        x_18 = self.pool1(x_150)
        x_18 = self.conv2(x_18)
        x_2 = self.pool2(x_18)

        x_2 = self.dp_out1(x_2)
        x3 = self.conv3(x_2)
        x3 = self.conv4(x3)
        x_concat = torch.add(x3, x_2)
        x4 = self.dp_out2(x_concat)
        x4 = x4.reshape(x4.size(0), -1)
        # feature = self.mlp(x4)
        out = self.fc(x4)
        # x_150 16*224*224  x_18 16*28*28  x4 1568  out 2
        return x_150, x_18, x4, out

class SpoofNet_contra(nn.Module):
    def __init__(self,in_channels=3):
        super(SpoofNet_contra, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Sequential(
            basic_conv(in_planes=self.in_channels,out_planes=16,kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=9, stride=8, padding=1)
        self.conv2 = nn.Sequential(
            basic_conv(in_planes=16, out_planes=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=6, stride=4, padding=1)
        self.dp_out1 = nn.Dropout(p=0.25)

        self.conv3 = nn.Sequential(
            basic_conv(in_planes=32, out_planes=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            basic_conv(in_planes=16, out_planes=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

        )
        self.dp_out2 = nn.Dropout(p=0.25)

        self.fc = nn.Sequential(
            nn.Linear(1568, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 2)
        )

        self.mlp = nn.Sequential(
            nn.Linear(1568, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self,x_input):
        x_150 = self.conv1(x_input)
        x_18 = self.pool1(x_150)
        x_18 = self.conv2(x_18)
        x_2 = self.pool2(x_18)

        x_2 = self.dp_out1(x_2)
        x3 = self.conv3(x_2)
        x3 = self.conv4(x3)
        x_concat = torch.add(x3, x_2)
        x4 = self.dp_out2(x_concat)
        x4 = x4.reshape(x4.size(0), -1)
        feature = self.mlp(x4)
        out = self.fc(x4)
        # x_150 16*224*224  x_18 16*28*28  x4 1568  out 2
        return x_150, x_18, x4, out, feature

class SpoofNet_rsgb(nn.Module):
    def __init__(self,in_channels=3):
        super(SpoofNet_rsgb, self).__init__()
        self.in_channels = in_channels
        self.conv_res = Conv2d_res_v3(in_planes=self.in_channels, out_planes=16, kernel_size=(3, 3), stride=(1, 1))
        self.conv1_res = nn.Sequential(
            basic_conv(in_planes=in_channels*2, out_planes=16, kernel_size=(3, 3), stride=(1, 1)),
            # Conv2d_res_v2(in_planes=self.in_channels,out_planes=16,kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(kernel_size=9, stride=8, padding=1)
        self.conv2 = nn.Sequential(
            basic_conv(in_planes=16, out_planes=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=6, stride=4, padding=1)
        self.dp_out1 = nn.Dropout(p=0.25)

        self.conv3 = nn.Sequential(
            basic_conv(in_planes=32, out_planes=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            basic_conv(in_planes=16, out_planes=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

        )
        self.dp_out2 = nn.Dropout(p=0.25)

        self.fc = nn.Sequential(
            nn.Linear(1568, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 2)
        )

    def forward(self, x_input):
        x_input, ori = self.conv_res(x_input)
        x_150 = self.conv1_res(x_input)
        x_18 = self.pool1(x_150)
        x_18 = self.conv2(x_18)
        x_2 = self.pool2(x_18)

        x_2 = self.dp_out1(x_2)
        x3 = self.conv3(x_2)
        x3 = self.conv4(x3)
        x_concat = torch.add(x3, x_2)
        x4 = self.dp_out2(x_concat)
        x4 = x4.reshape(x4.size(0), -1)
        out = self.fc(x4)
        # x_150 16*224*224  x_18 16*28*28  x4 1568  out 2
        return x_150, x_18, x4, out, x_input, ori

class SpoofNet_rsgb256(nn.Module):
    def __init__(self,in_channels=3):
        super(SpoofNet_rsgb256, self).__init__()
        self.in_channels = in_channels
        self.conv_res = Conv2d_res_v2(in_planes=self.in_channels, out_planes=16, kernel_size=(3, 3), stride=(1, 1))
        self.conv1_res = nn.Sequential(
            basic_conv(in_planes=in_channels*2, out_planes=16, kernel_size=(3, 3), stride=(1, 1)),
            # Conv2d_res_v2(in_planes=self.in_channels,out_planes=16,kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(kernel_size=9, stride=8, padding=1)
        self.conv2 = nn.Sequential(
            basic_conv(in_planes=16, out_planes=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=6, stride=4, padding=1)
        self.dp_out1 = nn.Dropout(p=0.25)

        self.conv3 = nn.Sequential(
            basic_conv(in_planes=32, out_planes=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            basic_conv(in_planes=16, out_planes=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

        )
        self.dp_out2 = nn.Dropout(p=0.25)

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 2)
        )

    def forward(self, x_input):
        x_input, ori = self.conv_res(x_input)
        x_150 = self.conv1_res(x_input)
        x_18 = self.pool1(x_150)
        x_18 = self.conv2(x_18)
        x_2 = self.pool2(x_18)

        x_2 = self.dp_out1(x_2)
        x3 = self.conv3(x_2)
        x3 = self.conv4(x3)
        x_concat = torch.add(x3, x_2)
        x4 = self.dp_out2(x_concat)
        x4 = x4.reshape(x4.size(0), -1)
        out = self.fc(x4)
        # x_150 16*224*224  x_18 16*28*28  x4 1568  out 2
        return x_150, x_18, x4, out, x_input, ori

class SpoofNet_256(nn.Module):
    def __init__(self, in_channels=3):
        super(SpoofNet_256, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Sequential(
            basic_conv(in_planes=self.in_channels, out_planes=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=9, stride=8, padding=1)
        self.conv2 = nn.Sequential(
            basic_conv(in_planes=16, out_planes=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=6, stride=4, padding=1)
        self.dp_out1 = nn.Dropout(p=0.25)

        self.conv3 = nn.Sequential(
            basic_conv(in_planes=32, out_planes=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            basic_conv(in_planes=16, out_planes=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

        )
        self.dp_out2 = nn.Dropout(p=0.25)

        self.mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 2)
        )

    def forward(self, x_input):
        x_150 = self.conv1(x_input)
        x_18 = self.pool1(x_150)
        x_18 = self.conv2(x_18)
        x_2 = self.pool2(x_18)

        x_2 = self.dp_out1(x_2)
        x3 = self.conv3(x_2)
        x3 = self.conv4(x3)
        x_concat = torch.add(x3, x_2)
        x4 = self.dp_out2(x_concat)
        x4 = x4.reshape(x4.size(0), -1)
        # feature = self.mlp(x4)
        out = self.fc(x4)
        # x_150 16*224*224  x_18 16*28*28  x4 1568  out 2  feature 128
        return x_150, x_18, x4, out


def build_loss(logits, maps=None, label=None):
    logits_map = logits
    logits_cla = torch.mean(logits_map, dim=1)
    logits_cla = torch.mean(logits_cla, dim=1)
    logits_cla = torch.mean(logits_cla, dim=1, keepdim=True)
    logits_cla_ = torch.cat([logits_cla, 1 - logits_cla], axis=1)
    logits = logits_cla
    label = label.float()
    label = label.unsqueeze(-1)
    # print('logits-shape:', logits.shape)
    loss_cla = nn.BCEWithLogitsLoss()(logits, label)
    maps_reg = maps / 255.0
    loss_depth_1 = (logits_map - maps_reg) ** 2
    loss_depth_1 = torch.mean(loss_depth_1)
    loss_depth_2 = contrast_depth_loss(logits_map, maps_reg)
    loss_depth = loss_depth_1 + 0.5 * loss_depth_2
    loss = loss_cla + 0.5 * loss_depth
    acc = accuracy(logits, label, topk=(1,))

    return loss, acc, logits_cla_

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



if __name__ == '__main__':
    import os
    gpus = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # inputs = torch.rand((10,3)).cuda()
    # cdc1 = FaceMapNet().to(device)
    rsgb = SpoofNet_v2().to(device)
    summary(rsgb, (6, 256, 256), device='cuda')

    # summary(disc, (1568,))
    # x_input, x_150, x_18, out = cdc1(inputs)
    from PIL import Image
    from torchvision import transforms as T
    from utils.dataset import filter_8_1
    import cv2
    # test()
    # img_path = '/DISK3/rosine/cross_dataset/frames_crop/Replay_crop/train/real/client027_session01_webcam_authenticate_adverse_2/346.jpg'  # real
    # img_path = "/DISK3/rosine/cross_dataset/frames_crop/Replay_crop/train/attack/hand/print_client018_session01_highdef_photo_adverse/222.jpg" # attack
    # plgf real[1]: "/DISK3/houjyuf/dataset_old/ros_plgf/CASIA/train/real/8/HR_1/188.jpg"
    img_path = "/DISK3/houjyuf/dataset_old/ros_plgf/CASIA/train/attack/8/8/98.jpg"  # plgf fake[0]:
    save_path = '/DISK3/houjyuf/dataset/train_attack_8_8_98.jpg'
    ori_path = "/DISK3/rosine/cross_dataset/frames_crop/Casia_crop/train/attack/8/8/98.jpg"
    # ori_path = '/DISK3/houjyuf/dataset/trains_attack_8_8_98.jpg'
    img_plgf = Image.open(img_path)
    img_p = np.array(img_plgf)

    img_ori = Image.open(ori_path)  # RGB
    img_o = np.array(img_ori)
    I = cv2.imread(ori_path, -1)
    I_224 = cv2.resize(I, (224, 224))

    ###### PLGF ######
    # I_r, I_g, I_b = I[:, :, 0], I[:, :, 1], I[:, :, 2]
    #
    # model_type = '8_1'
    # if model_type == '8_1':
    #     I_r = filter_8_1(I_r)
    #     I_g = filter_8_1(I_g)
    #     I_b = filter_8_1(I_b)
    #
    # I[:, :, 0], I[:, :, 1], I[:, :, 2] = I_r, I_g, I_b
    # cv2.imwrite(save_path, I)
    # img_cv = cv2.imread(save_path, -1)
    # img_ = Image.open(save_path)
    # img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    # img_i = np.array(img_)
    # diff = img_cv - img_p

    transforms = T.Compose([
        T.Resize(size=[224, 224]),
        # RandomErasing(),  # narray [h,w,c]
        # RandomHorizontalFlip(),  # narray [h,w,c]
        # T.ToTensor(),  # narray -> tensor [h,w,c]
        # Cutout(),  # tensor [h,w,c]
    ])
    arr = transforms(img_plgf)
    arr_n = np.array(arr)

    # image_224 = cv2.cvtColor(I_224,cv2.COLOR_RGB2BGR)

    # norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # arr = norm(arr)  # input = (input - mean) / std
    # print(arr.shape)
    img = torch.from_numpy(I.transpose((2, 0, 1))).unsqueeze(0)
    img = img.float().cuda()

    conv_res = Conv2d_res_v2(in_planes=3, out_planes=16, kernel_size=(3, 3), stride=(1, 1)).cuda()
    x_input, ori = conv_res(img)
    myplgf = x_input[0].cpu().data.numpy()

    myplgf = myplgf.astype(np.uint8)
    myplgf = myplgf.transpose(1, 2, 0)  # C,H,W --> H, W, C
    temp_path = '/DISK3/houjyuf/dataset/trainattack_8_8_98.jpg'
    # myplgf = cv2.cvtColor(myplgf, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(temp_path, myplgf)
    myplgf_i = Image.fromarray(cv2.cvtColor(myplgf,cv2.COLOR_BGR2RGB))
    myplgf_i.save(temp_path)
    ## open as image type
    Imag_mp = Image.open(temp_path)
    Imag_mp = np.array(Imag_mp)
    ## open as numpy type
    Imag_np = cv2.imread(temp_path, -1)
    Imag_np = cv2.cvtColor(Imag_np, cv2.COLOR_RGB2BGR)
    diff_ori = Imag_mp - myplgf
    diffs = Imag_np - myplgf
    diff_bet = Imag_np - Imag_mp
    print(myplgf.shape)

