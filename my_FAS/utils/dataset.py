import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np
import math
import cv2
import torch.nn as nn
import torch.nn.functional as F


def Normalization255_GRAY(img, max_value=255, min_value=0):
    Max = np.max(img)
    Min = np.min(img)
    img = ((img - Min) / (Max - Min)) * (max_value - min_value) + min_value
    return img

def filter_nn(img, kernel, padding=1):
    # expected img:   FloatTensor [w,h]
    # expected kernel:  FloatTensor [size,size]
    img = torch.Tensor(img)
    img = img.unsqueeze(0).unsqueeze(0)
    img = img.float()
    kernel = torch.Tensor(kernel)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    weight = nn.Parameter(data=kernel, requires_grad=False)
    res = F.conv2d(img, weight, padding=padding)
    return res

def produce_x(img_gray, R=1):
    # produce x component,
    # require:img_gray as narray
    # return: img_x as tensor
    if R == 1:
        filter_x = np.array([[-1 / (2 * np.sqrt(2)), 0, 1 / (2 * np.sqrt(2))],
                             [-1, 0, 1],
                             [-1 / (2 * np.sqrt(2)), 0, 1 / (2 * np.sqrt(2))]])
    if R == 2:
        filter_x = np.array([
            [np.cos(6 / 8 * np.pi) / 8, np.cos(5 / 8 * np.pi) / 5, np.cos(4 / 8 * np.pi) / 4,
             np.cos(3 / 8 * np.pi) / 5, np.cos(2 / 8 * np.pi) / 8],
            [np.cos(7 / 8 * np.pi) / 5, -1 / (2 * np.sqrt(2)), 0, 1 / (2 * np.sqrt(2)), np.cos(1 / 8 * np.pi) / 5],
            [np.cos(8 / 8 * np.pi) / 4, -1, 0, 1, np.cos(0 / 8 * np.pi) / 4],
            [np.cos(9 / 8 * np.pi) / 5, -1 / (2 * np.sqrt(2)), 0, 1 / (2 * np.sqrt(2)), np.cos(15 / 8 * np.pi) / 5],
            [np.cos(10 / 8 * np.pi ) / 8, np.cos(11 / 8 * np.pi) / 5, np.cos(12 / 8 * np.pi) / 4, np.cos(13 / 8 * np.pi) / 5,
             np.cos(14 / 8 * np.pi) / 8]])
    img_x = filter_nn(img_gray, filter_x, padding=R)
    img_xorl = np.array(img_x).reshape(img_x.shape[2], -1)
    return img_xorl

def produce_y(img_gray, R=1):
    # produce x component,
    # require:img_gray as narray
    # return: img_x as tensor
    if R == 1:
        filter_y = np.array([[1 / (2 * np.sqrt(2)), 1, 1 / (2 * np.sqrt(2))],
                             [0, 0, 0],
                             [-1 / (2 * np.sqrt(2)), -1, -1 / (2 * np.sqrt(2))]])
    if R == 2:
        filter_y = np.array([
            [np.sin(6 / 8 * np.pi) / 8, np.sin(5 / 8 * np.pi) / 5, np.sin(4 / 8 * np.pi) / 4,
             np.sin(3 / 8 * np.pi) / 5, np.sin(2 / 8 * np.pi) / 8],
            [np.sin(7 / 8 * np.pi) / 5, 1 / (2 * np.sqrt(2)), 1, 1 / (2 * np.sqrt(2)), np.sin(1 / 8 * np.pi) / 5],
            [np.sin(8 / 8 * np.pi) / 4, 0, 0, 0, np.sin(0 / 8 * np.pi) / 4],
            [np.sin(9 / 8 * np.pi) / 5,-1 / (2 * np.sqrt(2)), -1, -1 / (2 * np.sqrt(2)), np.sin(15 / 8 * np.pi) / 5],
            [np.sin(10 / 8 * np.pi ) / 8, np.sin(11 / 8 * np.pi) / 5, np.sin(12 / 8 * np.pi) / 4, np.sin(13 / 8 * np.pi) / 5,
             np.sin(14 / 8 * np.pi) / 8]])

    img_y = filter_nn(img_gray, filter_y, padding=R)
    img_yorl = np.array(img_y).reshape(img_y.shape[2], -1)
    return img_yorl

def filter_8_1(img):
    img_xorl = produce_x(img,1)  # default 2
    img_yorl = produce_y(img,1)  # default 2

    #magtitude = np.arctan(np.divide(img_yorl+0.001,img_xorl+0.001))
    magtitude = np.arctan(np.sqrt((np.divide(img_xorl,img+0.001) ** 2) + (np.divide(img_yorl,img+0.001) ** 2)))
    magtitude = Normalization255_GRAY(magtitude, 255, 1)
    return magtitude

def filter_16_2(img):
    img_xorl = Normalization255_GRAY(produce_x(img,2),255,1)
    img_yorl = Normalization255_GRAY(produce_y(img,2),255,1)
    img = Normalization255_GRAY(img,255,1)
    magtitude = np.arctan(np.divide(np.sqrt((img_xorl ** 2) + (img_yorl ** 2)),img))
    magtitude = Normalization255_GRAY(magtitude, 255, 1)
    return magtitude

class PLGF(object):
    def __call__(self, img):

        # img = np.array(img)
        # arr = np.asarray(img).astype(np.float32)
        # Require: numpy.narray([224,224,3])
        I_r, I_g, I_b = arr[:, :, 0], arr[:, :, 1], arr[:,:,2]
        model_type = '8_1'
        if model_type == '8_1':
            I_r = filter_8_1(I_r)
            I_g = filter_8_1(I_g)
            I_b = filter_8_1(I_b)
        arr_plgf = np.stack([I_r, I_g, I_b], axis=2).astype(np.float32)
        return arr_plgf

class ColorJitter(object):
    def __call__(self, image_x):
        """
        color jitter
        :param img: (PIL Image) Input image   PIL: [H,W,C] BGR   narray: [C,H,W] RGB
        :return:  PIL Image: Color jittered image
        """
        # print(type(image_x))
        # image_x, spoofing_label = sample['image_x'], sample['spoofing_label']
        # image_x = Image.fromarray(cv2.cvtColor(image_x,cv2.COLOR_BGR2RGB))
        p = random.random()
        if p < 0.7:
            image_x = T.ColorJitter(contrast=(0.7, 1.3), brightness=(0.7, 1.3), saturation=(0.7, 1.3),hue=(-0.02, 0.02))(image_x)
        if 0.3<p < 0.5:
            image_x = T.ColorJitter(brightness=(0.7, 1.3),hue=(-0.02, 0.02))(image_x)
        if 0.5<p < 0.7:
            image_x = T.ColorJitter(brightness=(0.7, 1.3),saturation=(0.7, 1.3))(image_x)
        # image_x = cv2.cvtColor(np.asarray(image_x), cv2.COLOR_RGB2BGR)
        return image_x

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability=0.5, sl=0.01, sh=0.05, r1=0.5, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        # print('Before Erasing:',type(img))
        img = np.array(img)
        if random.uniform(0, 1) < self.probability:
            attempts = np.random.randint(1, 3)
            for attempt in range(attempts):
                area = img.shape[0] * img.shape[1]

                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < img.shape[1] and h < img.shape[0]:
                    x1 = random.randint(0, img.shape[0] - h)
                    y1 = random.randint(0, img.shape[1] - w)

                    img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                    img[x1:x1 + h, y1:y1 + w, 1] = self.mean[1]
                    img[x1:x1 + h, y1:y1 + w, 2] = self.mean[2]
        return img

class SetDiff(object):
    def __call__(self, img):
        p = random.random()
        # print('SetDiff',img.shape,type(img))
        if p < 0.5:
            arr = np.asarray(img).astype(np.float32)
            diff1 = arr[0, :, :] - arr[1, :, :]
            diff2 = arr[0, :, :] - arr[2, :, :]
            diff3 = arr[1, :, :] - arr[2, :, :]

            # diff1 = (diff1 - np.min(diff1)) / (np.max(diff1) - np.min(diff1))
            # diff2 = (diff2 - np.min(diff2)) / (np.max(diff2) - np.min(diff2))
            # diff3 = (diff3 - np.min(diff3)) / (np.max(diff3) - np.min(diff3))
            diff1 = (diff1 + 255) / 510
            diff2 = (diff2 + 255) / 510
            diff3 = (diff3 + 255) / 510
            arr_diff = np.stack([diff1, diff2, diff3],axis=0).astype(np.float32)
            print('SetDiff', arr_diff.shape, type(arr_diff))
            return arr_diff
        else:
            print('img', img.shape, type(img))
            return img

class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""

    def __call__(self, image_x):
        # new_image_x = np.zeros((256, 256, 3))
        # new_map_x = np.zeros((32, 32))
        # print('Before Flip:', type(image_x))
        p = random.random()
        if p < 0.5:
            # print('Flip')
            new_image_x = cv2.flip(image_x, 1)   # src cv2.Umat()  np.float32()

            return new_image_x
        else:
            # print('no Flip')
            return image_x

class Cutout(object):

    def __init__(self, length=28):
        """
            input: Tensor,  output: Tensor
        """
        self.length = length

    def __call__(self, img):
        # print('Before Cutout:', type(img))
        # img = torch.from_numpy(img)
        h, w = img.shape[1], img.shape[2]  # Tensor [1][2],  nparray [0][1],  PIL size[0]  PIL size[1]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        length_new = np.random.randint(1, self.length)

        y1 = np.clip(y - length_new // 2, 0, h)
        y2 = np.clip(y + length_new // 2, 0, h)
        x1 = np.clip(x - length_new // 2, 0, w)
        x2 = np.clip(x + length_new // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

# original dataset
# return train: image,label    test: image, label, videoID, sub_label
# transforms: [RandomErasing(), RandomHorizontalFlip(), Cutout()]
class YunpeiDataset(Dataset):
    def __init__(self, data_pd, transforms=None, train=True):
        self.train = train
        self.photo_path = data_pd['photo_path'].tolist()
        self.photo_label = data_pd['photo_label'].tolist()
        self.photo_belong_to_video_ID = data_pd['photo_belong_to_video_ID'].tolist()
        self.sub_label = data_pd['sub_label'].tolist()
        if transforms is None:
            if not train:
                self.transforms = T.Compose([
                    T.Resize(size=150),
                    # T.ToTensor(),   # PIL -> tensor [h,w,c]
                    # PLGF(),         # PIL -> narray [h,w,c]
                    T.ToTensor(),   # narray -> tensor  [h,w,c]
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(size=150),
                    # T.ToTensor(),    # PIL -> tensor [h,w,c]
                    # PLGF(),      # PIL -> narray [h,w,c]
                    RandomErasing(),   # narray [h,w,c]
                    # SetDiff(),
                    RandomHorizontalFlip(),   # narray [h,w,c]
                    # T.RandomHorizontalFlip(),
                    # T.RandomCrop(size=64),

                    # ColorJitter(),
                    T.ToTensor(),     # narray -> tensor [h,w,c]
                    Cutout(),         # tensor [h,w,c]
                    # T.Normalize expected input as sensor
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):
        if self.train:
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            img = Image.open(img_path)
            img = self.transforms(img)
            return img, label
        else:
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            videoID = self.photo_belong_to_video_ID[item]
            sub_label = self.sub_label[item]
            img = Image.open(img_path)   # img PIL
            img = self.transforms(img)
            # print(img.shape,type(img))
            return img, label, videoID, sub_label

# handle the mask file
# return train: image,label    test: image, label, videoID
# transforms: [RandomErasing(), RandomHorizontalFlip(), Cutout()]
class HoujyufDataset(Dataset):
    def __init__(self, data_pd, transforms=None, train=True, diff=False):
        self.train = train
        self.diff = diff
        self.photo_path = data_pd['photo_path'].tolist()
        self.photo_label = data_pd['photo_label'].tolist()
        self.photo_belong_to_video_ID = data_pd['photo_belong_to_video_ID'].tolist()
        self.sub_label = data_pd['sub_label'].tolist()
        self.mask_path = data_pd['mask_path'].tolist()
        if transforms is None:
            if not train:
                self.transforms = T.Compose([
                    T.Resize(size=[224,224]),
                    T.ToTensor(),   # narray -> tensor  [h,w,c]
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(size=[224,224]),
                    RandomErasing(),  # narray [h,w,c]
                    RandomHorizontalFlip(),  # narray [h,w,c]
                    T.ToTensor(),  # narray -> tensor [h,w,c]
                    Cutout(),  # tensor [h,w,c]
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):
        if self.train:
            img_path = self.photo_path[item]
            mask_path = self.mask_path[item]
            label = self.photo_label[item]

            img = Image.open(img_path)
            img = self.transforms(img)

            if mask_path == '':
                mask = torch.zeros_like(img)
            else:
                if self.diff == True:
                    mask = Image.open(mask_path)
                    mask = self.transforms(mask)
                    mask = img - mask
                # m_shape = mask.shape
                # print(m_shape)
                # mask = torch.full(m_shape, 1.0)
                else:
                    mask = torch.ones_like(img)
            return img, label
        else:
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            mask_path = self.mask_path[item]
            videoID = self.photo_belong_to_video_ID[item]
            sub_label = self.sub_label[item]
            img = Image.open(img_path)   # img PIL
            img = self.transforms(img)

            if mask_path == '':
                mask = torch.zeros_like(img)
            else:
                if self.diff == True:
                    mask = Image.open(mask_path)
                    mask = self.transforms(mask)
                    mask = img - mask

                else:
                    mask = torch.ones_like(img)
            return img, label, videoID

class HoujyufDataset_v2(Dataset):
    def __init__(self, data_pd, transforms=None, train=True, diff=False):
        self.train = train
        self.diff = diff
        self.photo_path = data_pd['photo_path'].tolist()
        self.photo_label = data_pd['photo_label'].tolist()
        self.photo_belong_to_video_ID = data_pd['photo_belong_to_video_ID'].tolist()
        self.sub_label = data_pd['sub_label'].tolist()
        self.mask_path = data_pd['mask_path'].tolist()
        if transforms is None:
            if not train:
                self.transforms = T.Compose([
                    # T.Resize(size=[224,224]),
                    T.ToTensor(),   # narray -> tensor  [h,w,c]
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    # T.Resize(size=[224,224]),
                    RandomErasing(),  # narray [h,w,c]
                    RandomHorizontalFlip(),  # narray [h,w,c]
                    T.ToTensor(),  # narray -> tensor [h,w,c]
                    Cutout(),  # tensor [h,w,c]
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):
        if self.train:
            img_path = self.photo_path[item]
            mask_path = self.mask_path[item]
            label = self.photo_label[item]

            img = Image.open(img_path)
            img = self.transforms(img)

            mask = Image.open(mask_path)
            mask = self.transforms(mask)

            return img, mask, label
        else:
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            mask_path = self.mask_path[item]
            videoID = self.photo_belong_to_video_ID[item]
            sub_label = self.sub_label[item]
            img = Image.open(img_path)   # img PIL
            img = self.transforms(img)

            mask = Image.open(mask_path)
            mask = self.transforms(mask)
            return img, mask, label, videoID


class RsgbDataset(Dataset):
    def __init__(self, data_pd, transforms=None, train=True, diff=False):
        self.train = train
        self.diff = diff
        self.photo_path = data_pd['photo_path'].tolist()
        self.photo_label = data_pd['photo_label'].tolist()
        self.photo_belong_to_video_ID = data_pd['photo_belong_to_video_ID'].tolist()
        self.sub_label = data_pd['sub_label'].tolist()
        self.mask_path = data_pd['mask_path'].tolist()
        if transforms is None:
            if not train:
                self.transforms = T.Compose([
                    T.Resize(size=[224,224]),
                    # T.ToTensor(),   # narray -> tensor  [h,w,c]
                    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(size=[224,224]),
                    # RandomErasing(),  # narray [h,w,c]
                    # RandomHorizontalFlip(),  # narray [h,w,c]
                    # T.ToTensor(),  # narray -> tensor [h,w,c]
                    # Cutout(),  # tensor [h,w,c]
                    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):
        if self.train:
            img_path = self.photo_path[item]
            mask_path = self.mask_path[item]
            label = self.photo_label[item]

            # img = Image.open(img_path)
            # img = self.transforms(img)
            I = cv2.imread(img_path, -1)
            # I_224 = cv2.resize(I, (224, 224))
            img = torch.from_numpy(I.transpose((2, 0, 1)))
            img = img.float()
            return img, label
        else:
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            mask_path = self.mask_path[item]
            videoID = self.photo_belong_to_video_ID[item]
            sub_label = self.sub_label[item]
            I = cv2.imread(img_path, -1)
            # I_224 = cv2.resize(I, (224, 224))
            img = torch.from_numpy(I.transpose((2, 0, 1)))
            img = img.float()
            # img = Image.open(img_path)
            # img = self.transforms(img)

            return img, label, videoID

class DepthDataset(Dataset):
    def __init__(self, data_pd, transforms=None, train=True, diff=False):
        self.train = train
        self.diff = diff
        self.photo_path = data_pd['photo_path'].tolist()
        self.photo_label = data_pd['photo_label'].tolist()
        self.photo_belong_to_video_ID = data_pd['photo_belong_to_video_ID'].tolist()
        self.sub_label = data_pd['sub_label'].tolist()
        self.mask_path = data_pd['mask_path'].tolist()
        if transforms is None:
            if not train:
                self.transforms = T.Compose([
                    T.Resize(size=[224,224]),
                    T.ToTensor(),   # narray -> tensor  [h,w,c]
                    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(size=[224,224]),
                    RandomErasing(),  # narray [h,w,c]
                    RandomHorizontalFlip(),  # narray [h,w,c]
                    T.ToTensor(),  # narray -> tensor [h,w,c]
                    Cutout(),  # tensor [h,w,c]
                    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):
        if self.train:
            img_path = self.photo_path[item]
            mask_path = self.mask_path[item]
            label = self.photo_label[item]

            img = Image.open(img_path)
            img = self.transforms(img)

            return img, label
        else:
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            mask_path = self.mask_path[item]
            videoID = self.photo_belong_to_video_ID[item]
            sub_label = self.sub_label[item]
            img = Image.open(img_path)   # img PIL
            img = self.transforms(img)

            return img, label, videoID

# handle the grg file
class grgDataset(Dataset):
    def __init__(self, data_pd, transforms=None, train=True, diff=False, size=224):
        self.train = train
        self.diff = diff
        self.size = size
        self.photo_path = data_pd['photo_path'].tolist()
        self.photo_label = data_pd['photo_label'].tolist()
        self.photo_belong_to_video_ID = data_pd['photo_belong_to_video_ID'].tolist()
        if transforms is None:
            if not train:
                self.transforms = T.Compose([
                    # T.Resize(size=(224, 224)),
                    # PLGF(),         # PIL -> narray [h,w,c]
                    T.ToTensor(),   # narray -> tensor  [h,w,c]
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    # T.Resize(size=224),
                    # PLGF(),  # PIL -> narray [h,w,c]
                    RandomErasing(),  # narray [h,w,c]
                    RandomHorizontalFlip(),  # narray [h,w,c]
                    T.ToTensor(),  # narray -> tensor [h,w,c]
                    Cutout(),  # tensor [h,w,c]
                    # T.Normalize expected input as sensor
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):
        if self.train:
            img_path = self.photo_path[item]
            label = self.photo_label[item]

            # img = Image.open(img_path)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.size, self.size))
            # ft_sample = torch.from_numpy(ft_sample).float()
            # ft_sample = torch.unsqueeze(ft_sample, 0)
            img = self.transforms(img)

            return img, label
        else:
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            videoID = self.photo_belong_to_video_ID[item]
            # img = Image.open(img_path)   # img PIL
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.size, self.size))
            img = self.transforms(img)

            return img, label, videoID


class ContrasiveDataset(Dataset):
    def __init__(self, data_pd, transforms=None, train=True, diff=False):
        self.train = train
        self.diff = diff
        self.photo_path = data_pd['photo_path'].tolist()
        self.photo_label = data_pd['photo_label'].tolist()
        self.photo_belong_to_video_ID = data_pd['photo_belong_to_video_ID'].tolist()
        self.sub_label = data_pd['sub_label'].tolist()
        self.mask_path = data_pd['mask_path'].tolist()
        if transforms is None:
            if not train:
                self.transforms = T.Compose([
                    T.Resize(size=224),
                    T.ToTensor(),   # narray -> tensor  [h,w,c]
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.RandomResizedCrop(size=224, scale=(0.2, 1.)),
                    T.RandomHorizontalFlip(),
                    T.RandomApply([
                        T.ColorJitter(0.4, 0.4, 0.4, 0.1)
                    ], p=0.8),
                    T.RandomGrayscale(p=0.2),
                    T.ToTensor(),  # narray -> tensor [h,w,c]
                    Cutout(),  # tensor [h,w,c]
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):
        if self.train:
            img_path = self.photo_path[item]

            label = self.photo_label[item]
            sub_label = self.sub_label[item]
            img = Image.open(img_path)
            img_1 = self.transforms(img)
            img_2 = self.transforms(img)
            return img_1, img_2, label
        else:
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            videoID = self.photo_belong_to_video_ID[item]
            sub_label = self.sub_label[item]
            img = Image.open(img_path)   # img PIL
            img = self.transforms(img)

            return img, label, videoID


if __name__ == "__main__":
    #img_path = '/DISK3/rosine/cross_dataset/frames_crop/Replay_crop/train/real/client027_session01_webcam_authenticate_adverse_2/346.jpg'  # real
    #img_path = "/DISK3/rosine/cross_dataset/frames_crop/Replay_crop/train/attack/hand/print_client018_session01_highdef_photo_adverse/222.jpg" # attack
    # plgf real[1]: "/DISK3/houjyuf/dataset_old/ros_plgf/CASIA/train/real/8/HR_1/188.jpg"
    img_path = "/DISK3/houjyuf/dataset_old/ros_plgf/CASIA/train/attack/8/8/98.jpg"  # plgf fake[0]:
    save_path = '/DISK3/houjyuf/dataset/real_346.jpg'
    img = Image.open(img_path)  # RGB
    # tran = PLGF()
    # img = tran(img)
    # img_ = Image.fromarray(I.astype('uint8')).convert('RGB')
    transforms = T.Compose([
        T.Resize(size=[224, 224]),
        RandomErasing(),  # narray [h,w,c]
        RandomHorizontalFlip(),  # narray [h,w,c]
        T.ToTensor(),  # narray -> tensor [h,w,c]
        Cutout(),  # tensor [h,w,c]
    ])
    arr = transforms(img)
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    arr = norm(arr)  # input = (input - mean) / std
    print(arr.shape)