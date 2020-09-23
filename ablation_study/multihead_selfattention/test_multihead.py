import numpy as np
import copy
import math
import time
import sys
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_cnn import NTUDataLoaders 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * s1 * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        mask = mask.double()
        mask = torch.sum(mask, dim=3)
        mask = mask.unsqueeze(-1)
        mask = mask.repeat(1,scores.size(1),1,scores.size(-1))
        mask = mask.bool()
        scores = scores.masked_fill(mask==0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalization
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff=d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff=2048):
        super().__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads, d_ff), N)
        self.norm = Norm(d_model)
    def forward(self, x, mask):
        for i in range(self.N):
            x = self.layers[i](x, mask)
        x = self.norm(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, d_ff=2048):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, d_ff)
        self.out = nn.Linear(d_model, trg_vocab)
        self.name = 'N('+str(N)+')H('+str(heads)+')'
    def forward(self, x, mask):
        x = self.encoder(x, mask)

        x = torch.sum(x, dim=1)
        m = mask.float()
        m = torch.sum(m, dim=1) + 1e-9
        x = x / m

        x = self.out(x)
        return x
        

''' train '''
def train(N, heads, d_ff, test=True, return_acc=False):
    src_vocab = 150
    trg_vocab = 60
    d_model = 150

    model = SelfAttention(src_vocab, trg_vocab, d_model, N, heads, d_ff)
    model.to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    folder = 'cp_'
    for i in [N, heads, d_ff]:
        folder = folder + str(i) + '_'
    print(folder)
    os.mkdir('checkpoints/'+folder)

    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    running_loss = 0.0
    print_loss = True
    epochs = 200
    lr = 0.001
    best_acc = 0
    acc_his = []
    max_acc = 15
    min_diff = 0.001
    
    print('\nEpochs:', epochs)
    for epoch in range(epochs):
        t0 = time.time()
        for batch in tqdm(train_loader):
            x, y = batch[0].to(device), batch[1].to(device)
            mask = (x==0)

            output = model(x, mask)
            
            optim.zero_grad()
            loss = criterion(output, y)
            loss.backward()
            optim.step()

        with torch.no_grad():
            # train accuracy
            correct = 0
            total = 0
            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                mask = x==0

                output = model(x, mask)
                _, preds = torch.max(output.data, 1)
                total += y.size(0)
                correct += (preds == y).sum().item()
            acc = correct / total
            print(f'epoch:{epoch+1:3d} train acc: {acc:.6f}  [{correct:5d}/{total:5d}]',end='\t')
            with open('checkpoints/'+folder+'/train_record.txt', 'a') as f:
                f.write(str(acc)+'\n')

            # test accuracy
            correct = 0
            total = 0
            for batch in test_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                mask = x==0

                output = model(x, mask)
                _, preds = torch.max(output.data, 1)
                total += y.size(0)
                correct += (preds == y).sum().item()
            acc = correct / total
            print(f'test  acc: {acc:.6f}  [{correct:5d}/{total:5d}]')
            with open('checkpoints/'+folder+'/test_record.txt', 'a') as f:
                f.write(str(acc)+'\n')

            if best_acc < acc:
                best_acc = acc
                ep = str(epoch+1) if epoch>10 else '0'+str(epoch+1)
                path = 'checkpoints/'+folder+'/model_'+ep+'_'+str(acc)+'_'+model.name+'.pth'
                torch.save(model.state_dict(), path)
                best_acc = acc
                counter = 0
            else:
                counter += 1
                if counter == max_acc:
                    break

    if test:
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in tqdm(test_loader):
                x, y = batch[0].to(device), batch[1].to(device)
                mask = x==0

                output = model(x, mask)
                _, preds = torch.max(output.data, 1)
                total += y.size(0)
                correct += (preds == y).sum().item()
            acc = correct / total
            print(f'Test acc: {acc:.6f}  [{correct:5d}/{total:5d}]')

    return best_acc


def test(PATH):
    src_vocab = 150
    trg_vocab = 60
    d_model = 150
    N = 6
    heads = 5

    model = SelfAttention(src_vocab, trg_vocab, d_model, N, heads)
    model.load_state_dict(torch.load(PATH))
    model.to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            mask = x==0

            output = model(x, mask)
            _, preds = torch.max(output.data, 1)
            total += y.size(0)
            correct += (preds == y).sum().item()
        acc = correct / total
        print(f'Test acc: {acc:.6f}  [{correct:5d}/{total:5d}]')


''' info '''
batch_size = 16
workers = 8
ratio = 1

''' data '''
print('Loading data...')
start = time.time()
ntu_loaders = NTUDataLoaders('NTU', 0, 0, 1)
train_loader = ntu_loaders.get_train_loader(batch_size, workers)
val_loader = ntu_loaders.get_val_loader(batch_size, workers)
test_loader = ntu_loaders.get_test_loader(batch_size, workers)
train_size = ntu_loaders.get_train_size()
val_size = ntu_loaders.get_val_size()
test_size = ntu_loaders.get_test_size()
total_steps = math.ceil(train_size/batch_size)
print('Train on %d samples, validate on %d samples, test on %d samples' %
      (train_size, val_size, test_size))
print(f'Total train steps: {total_steps}')
print(f'Load data: {time.time()-start:.3f} seconds\n')


def run_experiments():
    c = 1
    for N in range(1,7):
        for heads in [5, 10, 15]:
            for d_ff in [128, 256, 512, 1024, 2048]:
                c+=1
                print(f'start {c}/{6*3*5} {N} {heads} {d_ff}')
                try:
                    acc = train(N, heads, d_ff, test=False, return_acc=False)
                except:
                    print('\n\n[ERROR]',N, heads, d_ff, '\n\n')
                print(f'end {c}/{6*3*5}')
                

if __name__ == '__main__':

    run_experiments()
    print('\n\n')
    print('DONE')

