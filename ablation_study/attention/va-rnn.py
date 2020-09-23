# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
parser = argparse.ArgumentParser(description='View adaptive')
parser.add_argument('--dataset', type=str, default='NTU',
                    help='type of dataset')
parser.add_argument('--model', type=str, default='VA',
                    help='type of recurrent net (VA, basline)')
parser.add_argument('--nhid', type=int, default=100,
                    help='number of hidden units per layer')
parser.add_argument('--lr', type=float, default=0.005,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=1.,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=256,
                    help='batch size')
parser.add_argument('--case', type=int, default=0,
                    help='select dataset')
parser.add_argument('--norm', type=float, default=0.001,
                    help='LSMT recurrent initializer')
parser.add_argument('--aug', type=int, default=1,
                    help='data augmentation')
parser.add_argument('--save', type=int, default=1,
                    help='save results')
parser.add_argument('--gpu', type=int, default=0,
                    help='which gpu is used to train')
parser.add_argument('--train', type=int, default=1,
                    help='train or test')
args = parser.parse_args()

import time
t0 = time.time()
import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
import keras.backend as K
import theano.tensor as T
from keras import initializers
from keras.optimizers import Adam
from keras.layers import Input, Dropout, LSTM, Dense, Activation, TimeDistributed, Masking, Concatenate
from keras.callbacks import EarlyStopping,CSVLogger,ReduceLROnPlateau, ModelCheckpoint
from transform_rnn import VA, Noise,MeanOverTime, augmentaion
from data_rnn import  get_data, get_cases, get_activation
#jeff
from keras.models import Model
from keras.layers import Layer, Flatten, RepeatVector, Permute, multiply, Bidirectional

FLAG = 0



def divider():
    print('\n\n')
    print('+-----------------------------------------------------------------+')
    print('\n\n')
divider()
t1 = time.time()
print('\n\nTime used for loading dependencies: {} seconds\n'.format(round(t1-t0,2)))


# Attention1 = not consider mask
# Attention2 = consider mask
class Attention1(Layer):
    def __init__(self,**kwargs):
        super(Attention1,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")
        super(Attention1, self).build(input_shape)

    def call(self,x,mask=None):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return output

    def compute_output_shape(self,input_shape):
        print('input_shape:',input_shape)
        print('len:',len(input_shape))
        return (input_shape[0],input_shape[1],input_shape[2])

    def get_config(self):
        return super(Attention1,self).get_config()

    def compute_mask(self, inputs, mask=None):
        return mask

class Attention2(Layer):
    def __init__(self,**kwargs):
        super(Attention2,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")
        super(Attention2, self).build(input_shape)

    def call(self,x,mask=None):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            et = et*mask
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        
        output = x*at
        output = K.sum(output,axis=1, keepdims=True)
        return output

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(Attention2,self).get_config()

    def compute_mask(self, inputs, mask=None):
        return mask


def creat_model(input_shape, num_class):

    init = initializers.Orthogonal(gain=args.norm)
    sequence_input =Input(shape=input_shape)
    mask = Masking(mask_value=0.)(sequence_input)
    if args.aug:
        mask = augmentaion()(mask)
    X = Noise(0.075)(mask)
    if args.model[0:2]=='VA':
        # VA
        trans = Bidirectional(LSTM(args.nhid,recurrent_activation='sigmoid',return_sequences=True,implementation=2,recurrent_initializer=init))(X)
        trans = Dropout(0.5)(trans)
        trans = TimeDistributed(Dense(3,kernel_initializer='zeros'))(trans)
        rot = Bidirectional(LSTM(args.nhid,recurrent_activation='sigmoid',return_sequences=True,implementation=2,recurrent_initializer=init))(X)
        rot = Dropout(0.5)(rot)
        rot = TimeDistributed(Dense(3,kernel_initializer='zeros'))(rot)
        transform = Concatenate()([rot,trans])
        X = VA()([mask,transform])

    X = LSTM(args.nhid,recurrent_activation='sigmoid',return_sequences=True,implementation=2,recurrent_initializer=init)(X)
    X = Dropout(0.5)(X)
    X = LSTM(args.nhid,recurrent_activation='sigmoid',return_sequences=True,implementation=2,recurrent_initializer=init)(X)
    X = Dropout(0.5)(X)
    X = LSTM(args.nhid,recurrent_activation='sigmoid',return_sequences=True,implementation=2,recurrent_initializer=init)(X)
    X = Dropout(0.5)(X)
    X = Attention2()(X)
    X = TimeDistributed(Dense(num_class))(X)

    X = MeanOverTime()(X)
    X = Dense(num_class)(X)
    X = Activation('softmax')(X)

    model=Model(sequence_input,X)
    print(model.summary())
    return model

def main(rootdir, case, results):
    t0 = time.time()
    divider()
    print('main()')
    print('load data')
    fast = False
    
    n = 3000
    train_x = np.random.rand((n*300*150)).reshape(n,300,150)
    train_y = np.random.rand((n*60)).reshape(n,60)
    n = 3000
    valid_x = np.random.rand((n*300*150)).reshape(n,300,150) 
    valid_y = np.random.rand((n*60)).reshape(n,60)
    n = 3000
    test_x  = np.random.rand((n*300*150)).reshape(n,300,150)
    test_y  = np.random.rand((n*60)).reshape(n,60)

    if not fast:
        train_x, train_y, valid_x, valid_y, test_x, test_y = get_data(args.dataset, case)

    print('train_x:', train_x.shape)
    print('train_y:', train_y.shape)
    print('valid_x:', valid_x.shape)
    print('valid_y:', valid_y.shape)
    print('test_x:', test_x.shape)
    print('test_y:', test_y.shape)

    print('get input shape')
    input_shape = (train_x.shape[1], train_x.shape[2])
    num_class = train_y.shape[1]
    print('path stuff')
    if not os.path.exists(rootdir): os.makedirs(rootdir)
    filepath = os.path.join(rootdir, str(case) + '.hdf5')
    saveto = os.path.join(rootdir, str(case) + '.csv')
    optimizer = Adam(lr=args.lr, clipnorm=args.clip)
    pred_dir = os.path.join(rootdir, str(case) + '_pred.txt')
	
    t1 = time.time()
    divider()
    print('Time used for before training: {} seconds'.format(round(t1-t0,2)))
    t0 = time.time()

    if args.train==1:
        print('Start training...')
        print('create model...')
        model = creat_model(input_shape, num_class)
        divider()
        early_stop = EarlyStopping(monitor='val_accuracy', patience=15, mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, mode='auto', cooldown=3., verbose=1)
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
        csv_logger = CSVLogger(saveto)
        if args.dataset=='NTU' or args.dataset == 'PKU':
            callbacks_list = [csv_logger, checkpoint, early_stop, reduce_lr]
        else:
            callbacks_list = [csv_logger, checkpoint]

        print('Compile model...')
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        print('fit()')
        model.fit(train_x, train_y, validation_data=[valid_x, valid_y], epochs=args.epochs,
                  batch_size=args.batch_size, callbacks=callbacks_list, verbose=1)

    # test
    model = creat_model(input_shape, num_class)
    model.load_weights(filepath)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    scores = get_activation(model, test_x, test_y, pred_dir, VA=10, par=9)
    print(round(scores,4))
    results.append(round(scores, 4))

    t1 = time.time()
    divider()
    print('Time used for training: {} minutes\n'.format(round((t1-t0)/60,2)))


if __name__ == '__main__':
    results = list()
    rootdir = os.path.join('./results/VA-RNN', args.dataset, args.model)
    cases = get_cases(args.dataset)

    for case in range(cases):
        divider()
        print('Case:',case)
        args.case = case
        main(rootdir, args.case, results)
        print('case:',case,'is done')
        print('\nBreak\n')
        break
    divider()
    FLAG += 1
    
    np.savetxt(rootdir + '/result.txt', results, fmt = '%f')

