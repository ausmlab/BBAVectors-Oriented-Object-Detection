
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision.transforms.functional import pad
import numpy as np
import numbers

def get_padding(image):
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding

# code from:
# https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/2
class NewPad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return pad(img, get_padding(img), self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)


def build_transform(args,is_train=True):
    transform = transforms.Compose([
        NewPad(),
        transforms.Resize([args.size, args.size]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform

def build_dataloader(args):
    transform_train = build_transform(args, is_train=True)
    dataset_train=ImageFolder(os.path.join(args.dataset_path,"train"),
                              transform_train)
    dataloader_train=DataLoader(dataset_train,batch_size=args.batch_size,
                                shuffle=True,num_workers=args.nthreads,drop_last=True)

    transform_val = build_transform(args, is_train=False)
    dataset_val = ImageFolder(os.path.join(args.dataset_path, "val"),
                              transform_val)
    dataloader_val = DataLoader(dataset_val, batch_size=8,
                                  shuffle=False, num_workers=args.nthreads)

    num_classes=len(dataset_train.classes)
    n_iter_per_epoch_train=len(dataloader_train)

    return dataloader_train,dataloader_val,dataset_train,dataset_val,\
           num_classes,n_iter_per_epoch_train


def build_val_dataloader(args, nthreads=8):
    transform_val = build_transform(args, is_train=False)
    dataset_val = ImageFolder(os.path.join(args.data_dir, "val"),
                              transform_val)
    dataloader_val = DataLoader(dataset_val, batch_size=8,
                                  shuffle=False, num_workers=nthreads)

    return dataloader_val, dataset_val
