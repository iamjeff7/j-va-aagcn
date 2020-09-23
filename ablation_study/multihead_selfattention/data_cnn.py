# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import h5py
import os.path as osp
import sys
import scipy.misc
from PIL import Image
#from skimage.transform import resize
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle



class NTUDataset(Dataset):
    """
    NTU Skeleton Dataset.

    Args:
        x (list): Input dataset, each element in the list is an ndarray corresponding to
        a joints matrix of a skeleton sequence sample
        y (list): Action labels
    """

    def __init__(self, x, y):
        self.x = x
        self.y = np.array(y, dtype='int')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return [self.x[index], int(self.y[index])]


class NTUDataLoaders(object):
    def __init__(self, dataset = 'NTU', case = 1, aug = 0, ratio=1, six_class=0):
        self.dataset = dataset
        self.case = case
        self.aug = aug
        self.ratio = ratio
        self.six_class = six_class
        self.create_datasets()
        self.train_set = NTUDataset(self.train_X, self.train_Y)
        self.val_set = NTUDataset(self.val_X, self.val_Y)
        self.test_set = NTUDataset(self.test_X, self.test_Y)

    def get_train_loader(self, batch_size, num_workers):
        if self.aug == 1:
            return DataLoader(self.train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
        else:
            return DataLoader(self.train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)

    def get_val_loader(self, batch_size, num_workers):
        return DataLoader(self.val_set, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers,
                          pin_memory=True)

    def get_test_loader(self, batch_size, num_workers):
        return DataLoader(self.test_set, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers,
                          pin_memory=True)

    def get_train_size(self):
        return len(self.train_Y)

    def get_val_size(self):
        return len(self.val_Y)

    def get_test_size(self):
        return len(self.test_Y)

    def create_datasets(self):
        if self.dataset =='NTU':
            if self.case == 0:
                self.metric = 'CS'
            else:
                self.metric = 'CV'
            path = osp.join('../data/ntu', 'NTU_' + self.metric + '.h5')

        if self.six_class:
            f = h5py.File(path, 'r')
            i = np.argmax(f['y'][:],-1)
            condition = (i==9)|(i==10)|(i==11)|(i==28)|(i==29)|(i==33)
            wanted_idx = np.where(condition)[0]
            self.train_X = f['x'][:][wanted_idx]
            self.train_Y = np.argmax(f['y'][:][wanted_idx],-1)

            i = np.argmax(f['valid_y'][:],-1)
            condition = (i==9)|(i==10)|(i==11)|(i==28)|(i==29)|(i==33)
            wanted_idx = np.where(condition)[0]
            self.val_X = f['valid_x'][:][wanted_idx]
            self.val_Y = np.argmax(f['valid_y'][:][wanted_idx],-1)

            i = np.argmax(f['test_y'][:],-1)
            condition = (i==9)|(i==10)|(i==11)|(i==28)|(i==29)|(i==33)
            wanted_idx = np.where(condition)[0]
            self.test_X = f['test_x'][:][wanted_idx]
            self.test_Y = np.argmax(f['test_y'][:][wanted_idx],-1)
            
        else:
            f = h5py.File(path, 'r')
            self.train_X = f['x'][::self.ratio]
            self.train_Y = np.argmax(f['y'][::self.ratio],-1)
            self.val_X = f['valid_x'][::self.ratio]
            self.val_Y = np.argmax(f['valid_y'][::self.ratio], -1)
            self.test_X = f['test_x'][::self.ratio]
            self.test_Y = np.argmax(f['test_y'][::self.ratio], -1)

        if self.dataset == 'NTU':
            self.max = 5.18858098984
            self.min = -5.28981208801

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def center(rgb):
    rgb[:,:,0] -= 110
    rgb[:,:,1] -= 110
    rgb[:,:,2] -= 110
    return rgb

def padding(joints, max_len=300, pad_value=0.):
    num_frames, feat_dim = joints.shape
    if feat_dim == 75:
        joints = np.hstack((joints, np.zeros((num_frames, 75), dtype=joints.dtype)))
    if num_frames < max_len:
        joints = np.vstack(
            (joints, np.ones((max_len - num_frames, 150), dtype=joints.dtype) * pad_value))

    return joints

def _rot(rot):
    cos_r, sin_r = rot.cos(), rot.sin()
    zeros = rot.new(rot.size()[:2] + (1,)).zero_()
    ones = rot.new(rot.size()[:2] + (1,)).fill_(1)

    r1 = torch.stack((ones, zeros, zeros),dim=-1)
    rx2 = torch.stack((zeros, cos_r[:,:,0:1], sin_r[:,:,0:1]), dim = -1)
    rx3 = torch.stack((zeros, -sin_r[:,:,0:1], cos_r[:,:,0:1]), dim = -1)
    rx = torch.cat((r1, rx2, rx3), dim = 2)

    ry1 = torch.stack((cos_r[:,:,1:2], zeros, -sin_r[:,:,1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros),dim=-1)
    ry3 = torch.stack((sin_r[:,:,1:2], zeros, cos_r[:,:,1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 2)

    rz1 = torch.stack((cos_r[:,:,2:3], sin_r[:,:,2:3], zeros), dim =-1)
    r3 = torch.stack((zeros, zeros, ones),dim=-1)
    rz2 = torch.stack((-sin_r[:,:,2:3], cos_r[:,:,2:3],zeros), dim =-1)
    rz = torch.cat((rz1, rz2, r3), dim = 2)

    rot = rz.matmul(ry).matmul(rx)

    return rot

def _transform(x):
    x = x.contiguous().view(x.size()[:2] + (-1, 3))

    rot = x.new(x.size()[0],3).uniform_(-0.3, 0.3)

    rot = rot.repeat(1, x.size()[1])
    rot = rot.contiguous().view((-1, x.size()[1], 3))
    rot = _rot(rot)
    x = torch.transpose(x, 2, 3)
    x = torch.matmul(rot, x)
    x = torch.transpose(x, 2, 3)

    x = x.contiguous().view(x.size()[:2] + (-1,))
    return x

def make_dir(dataset, case, subdir):
    if dataset == 'NTU':
        output_dir = os.path.join('./models/va-cnn/NTU/')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir

def get_cases(dataset):
    if dataset[0:3] == 'NTU':
        cases = 2

    return cases

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_num_classes(dataset):
    if dataset == 'NTU':
        return 60

