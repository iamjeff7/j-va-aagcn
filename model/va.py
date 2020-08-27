# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import division
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ViewAdaptive(nn.Module):
    """The layer for transforming the skeleton to the observed viewpoints"""
    def __init__(self):
        super(ViewAdaptive, self).__init__()

        # translation
        '''
        self.trans_lstm = nn.LSTM(150, 100, batch_first=True)
        self.trans_dropout = nn.Dropout(0.5, inplace=False)
        self.trans_fc = nn.Linear(100, 3)
        '''

        # rotation
        self.rot_lstm = nn.LSTM(150, 100, batch_first=True)
        self.rot_dropout = nn.Dropout(0.5, inplace=False)
        self.rot_fc = nn.Linear(100, 3)

        self.init_weights()


    def forward(self, x):

        # translation
        '''
        trans, _ = self.trans_lstm(x)
        trans = self.trans_dropout(trans)
        trans_reshape = trans.contiguous().view(-1, trans.size(-1))
        trans_y = self.trans_fc(trans_reshape)
        trans = trans_y.contiguous().view(trans.size(0), -1, trans_y.size(-1))
        '''

        # rotation
        rot, _ = self.rot_lstm(x)
        rot = self.rot_dropout(rot)
        rot_reshape = rot.contiguous().view(-1, rot.size(-1))
        rot_y = self.rot_fc(rot_reshape)
        rot = rot_y.contiguous().view(rot.size(0), -1, rot_y.size(-1))

        # translation = torch.cat((rot, trans), dim=-1)

        x = VA([x, rot])

        return x


    def init_weights(self):
        #for layer in [self.trans_lstm, self.rot_lstm]:
        for layer in [self.rot_lstm]:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, 0.001)
                if 'bias' in name:
                    param.data.zero_()


def VA(x):
    #print('-'*60)
    conv_input, theta = x
    s = theta.size()
    theta = theta.contiguous().view(-1, s[2])

    ### For translation
    #trans = _trans(theta)
    #output = _transform_trans(trans, conv_input)
    output = conv_input
    
    ### Calculate centroid
    cx = torch.mean(output.contiguous().view(-1,output.size(1)*output.size(2))[:, 0::3], dim=1)
    cy = torch.mean(output.contiguous().view(-1,output.size(1)*output.size(2))[:, 1::3], dim=1)
    cz = torch.mean(output.contiguous().view(-1,output.size(1)*output.size(2))[:, 2::3], dim=1)
    output = _translate_by_centroid(output, cx, cy, cz, d=-1)

    ### For rotation
    M = _fusion(theta)
    output = _transform_rot(M, output, output.size()[0])

    ### Translate back
    output = _translate_by_centroid(output, cx, cy, cz, d=1)

    return output


def _translate_by_centroid(output, cx, cy, cz, d):
    cx = cx.view(-1, 1).repeat(1, 300).view(-1, 1)*d
    cy = cy.view(-1, 1).repeat(1, 300).view(-1, 1)*d
    cz = cz.view(-1, 1).repeat(1, 300).view(-1, 1)*d

    zero = torch.zeros_like(cx)
    one = torch.ones_like(cx)
    first  = torch.cat([one, zero, zero, cx], dim=1).view(-1,1,4)
    second = torch.cat([zero, one, zero, cy], dim=1).view(-1,1,4)
    third  = torch.cat([zero, zero, one, cz], dim=1).view(-1,1,4)
    trans  = torch.cat([first, second, third], dim=1)

    o = output
    batch, step, dim = output.shape
    o = o.view(batch, step, dim//3,3)
    o = o.contiguous().view(batch*step, dim//3,3)
    o = o.permute(0,2,1)
    add = torch.ones(batch*step,1,dim//3).to(device)
    o = torch.cat([o, add], dim=1)

    output = torch.bmm(trans,o)
    output = output.permute(0,2,1)
    output = output.contiguous().view(output.size()[0], dim)
    output = output.view(batch, step, output.size()[1])

    return output


def _trans(theta):
    tx = theta[:, 3:4]
    ty = theta[:, 4:5]
    tz = theta[:, 5:6]

    tx = torch.div(torch.sum(tx.view(-1, 300), dim=1), 300.).view(-1, 1).repeat(1, 300).view(-1,1)
    ty = torch.div(torch.sum(ty.view(-1, 300), dim=1), 300.).view(-1, 1).repeat(1, 300).view(-1,1)
    tz = torch.div(torch.sum(tz.view(-1, 300), dim=1), 300.).view(-1, 1).repeat(1, 300).view(-1,1)

    zero = torch.zeros_like(tx)
    one = torch.ones_like(tx)
    first = torch.cat([one, zero, zero, tx], dim=1).view(-1,1,4)
    second = torch.cat([zero, one, zero, ty], dim=1).view(-1,1,4)
    third = torch.cat([zero, zero, one, tz], dim=1).view(-1,1,4)
    trans = torch.cat([first, second, third], dim=1)

    return trans


def _transform_trans(theta, input):
    batch, step, dim = input.shape
    input = input.contiguous().view(batch, step, dim//3,3)
    input = input.contiguous().view(batch*step, dim//3,3)
    input = input.permute(0,2,1)
    add = torch.ones(batch*step,1,dim//3).to(device)
    input = torch.cat([input, add], dim=1)

    output = torch.bmm(theta,input)
    output = output.permute(0,2,1)
    output = output.contiguous().view(output.size()[0], dim)
    output = output.view(batch, step, output.size()[1])

    return output


def _fusion(theta):
    rotation_x = _rotation_x(theta)
    rotation_y = _rotation_y(theta)
    rotation_z = _rotation_z(theta)

    rot = torch.bmm(rotation_z, rotation_y)
    rot = torch.bmm(rot, rotation_x)

    return rot


def _rotation_x(theta):
    r1 = theta[:,0:1]
    #r1 = torch.div(torch.sum(r1.view(-1, 300), dim=1),300.).view(-1, 1).repeat(1, 300).view(-1, 1).cos()
    r1 = torch.sum(r1.view(-1, 300), dim=1).view(-1, 1).repeat(1, 300).view(-1, 1).cos()
    r2 = theta[:,0:1]
    #r2 = torch.div(torch.sum(r2.view(-1, 300), dim=1),300.).view(-1, 1).repeat(1, 300).view(-1, 1).sin()
    r2 = torch.sum(r2.view(-1, 300), dim=1).view(-1, 1).repeat(1, 300).view(-1, 1).sin()

    zero = torch.zeros_like(r1)
    one = torch.ones_like(r1)
    first = torch.cat([one, zero, zero, zero], dim=1).view(-1,1,4)
    second = torch.cat([zero, r1, -r2, zero], dim=1).view(-1,1,4)
    thrid = torch.cat([zero, r2, r1, zero], dim=1).view(-1,1,4)
    fourth = torch.cat([zero, zero, zero, one], dim=1).view(-1,1,4)
    rotation_x = torch.cat([first, second, thrid, fourth], dim=1)
    
    return rotation_x
    
    
def _rotation_y(theta):
    r1 = theta[:,1:2]
    #r1 = torch.div(torch.sum(r1.view(-1, 300), dim=1),300.).view(-1, 1).repeat(1, 300).view(-1, 1).cos()
    r1 = torch.sum(r1.view(-1, 300), dim=1).view(-1, 1).repeat(1, 300).view(-1, 1).cos()
    r2 = theta[:,1:2]
    #r2 = torch.div(torch.sum(r2.view(-1, 300), dim=1),300.).view(-1, 1).repeat(1, 300).view(-1, 1).sin()
    r2 = torch.sum(r2.view(-1, 300), dim=1).view(-1, 1).repeat(1, 300).view(-1, 1).sin()

    zero = torch.zeros_like(r1)
    one = torch.ones_like(r1)
    first = torch.cat([r1, zero, r2, zero], dim=1).view(-1,1,4)
    second = torch.cat([zero, one, zero, zero], dim=1).view(-1,1,4)
    thrid = torch.cat([-r2, zero, r1, zero], dim=1).view(-1,1,4)
    fourth = torch.cat([zero, zero, zero, one], dim=1).view(-1,1,4)
    rotation_y = torch.cat([first, second, thrid, fourth], dim=1)

    return rotation_y


def _rotation_z(theta):
    r1 = theta[:,2:3]
    #r1 = torch.div(torch.sum(r1.view(-1, 300), dim=1),300.).view(-1, 1).repeat(1, 300).view(-1, 1).cos()
    r1 = torch.sum(r1.view(-1, 300), dim=1).view(-1, 1).repeat(1, 300).view(-1, 1).cos()
    r2 = theta[:,2:3]
    #r2 = torch.div(torch.sum(r2.view(-1, 300), dim=1),300.).view(-1, 1).repeat(1, 300).view(-1, 1).sin()
    r2 = torch.sum(r2.view(-1, 300), dim=1).view(-1, 1).repeat(1, 300).view(-1, 1).sin()
    
    zero = torch.zeros_like(r1)
    one = torch.ones_like(r1)
    first = torch.cat([r1, -r2, zero, zero], dim=1).view(-1,1,4)
    second = torch.cat([r2, r1, zero, zero], dim=1).view(-1,1,4)
    thrid = torch.cat([zero, zero, one, zero], dim=1).view(-1,1,4)
    fourth = torch.cat([zero, zero, zero, one], dim=1).view(-1,1,4)
    rotation_z = torch.cat([first, second, thrid, fourth], dim=1)
    
    return rotation_z


def _transform_rot(theta, input, batch_shape):
    batch, step, dim = input.size()
    input = input.view(-1, dim//3, 3)
    input = input.permute(0,2,1)
    add = torch.ones(batch*step, 1, dim//3).to(device)
    input = torch.cat((input, add), dim=1)

    output = torch.bmm(theta, input)
    output = output.permute(0,2,1)
    output = output[:,:,0:3]
    output = output.contiguous().view(-1, dim)
    output = output.view(-1, step, dim)

    return output

def bye():
    print('\n\n\t >> Terminate by user <<\a')
    sys.exit()


if __name__ == '__main__':
    model = ViewAdaptive()

    # data
    x = torch.rand(1,300,150)

    print(x)
    print()
    output = model(x)
    print(output)

