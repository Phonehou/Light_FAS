import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
import numpy as np
from torchsummary import summary

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

def spatial_gradient_x(input):
    # input: Tensor
    n, c, h, w = input.shape
    sobel_plane_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # sobel_plane_x = np.array([[-1 / (2 * np.sqrt(2)), 0, 1 / (2 * np.sqrt(2))],
    #           [-1, 0, 1],[-1 / (2 * np.sqrt(2)), 0, 1 / (2 * np.sqrt(2))]])
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
    sobel_plane_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # sobel_plane_y = np.array([[1 / (2 * np.sqrt(2)), 1, 1 / (2 * np.sqrt(2))],
    #                [0, 0, 0], [-1 / (2 * np.sqrt(2)), -1, -1 / (2 * np.sqrt(2))]])
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

def basic_conv(in_planes, out_planes, kernel_size=(3, 3), stride=(1, 1)):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=1, bias=False)

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
        m_ = torch.FloatTensor(3, 224, 224).fill_(1.0).to('cuda')
        self.mask = nn.Parameter(data=m_, requires_grad=True)

    def forward(self, x):
        # out_normal = self.pointwise(x)
        # out_normal = x[:] * self.mask
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
            # Max_ = torch.max(gradient_gabor, 2)
            # Min_ = torch.min(gradient_gabor, 2)
            # Max = torch.max(Max_[0], 2)
            # Min = torch.min(Min_[0], 2)
            # Max = Max[0].unsqueeze(-1).unsqueeze(-1)
            # Max = Max.repeat(1, 1, gradient_gabor.shape[-2], gradient_gabor.shape[-1])
            # Min = Min[0].unsqueeze(-1).unsqueeze(-1)
            # Min = Min.repeat(1, 1, gradient_gabor.shape[-2], gradient_gabor.shape[-1])
            # gradient_gabor = ((gradient_gabor - Min) / (Max - Min)) * 254 + 1
            # gradient_gabor = gradient_gabor / 255
            # gradient_gabor = normalize(gradient_gabor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # gradient_gabor_pw = self.conv(gradient_gabor)
            # res = torch.cat((out_normal, self.theta * gradient_gabor), dim=1)  # [:,6,224,224]
            # return self.conv(res)
            # return out_normal + self.theta * gradient_gabor_pw
            return gradient_gabor, out_normal

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


    def forward(self, feature):
        # adversarial_out = self.ad_net(self.grl_layer(feature))
        adversarial_out = -0.5*self.ad_net(feature)
        return adversarial_out

class SpoofNet(nn.Module):
    def __init__(self,in_channels=3):
        super(SpoofNet, self).__init__()
        self.in_channels = in_channels
        self.conv_res = Conv2d_res(in_planes=self.in_channels, out_planes=16, kernel_size=(3, 3), stride=(1, 1))
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
        x_input, ori = self.conv_res(x_input)
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
        return x_150, x_18, x4, out, x_input, ori

if __name__ == '__main__':
    import os
    gpus = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # inputs = torch.rand((10,3)).cuda()
    # cdc1 = FaceMapNet().to(device)
    rsgb = SpoofNet().to(device)
    summary(rsgb, (3, 224, 224), device='cuda')