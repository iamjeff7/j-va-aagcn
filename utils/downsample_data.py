import time
import os, sys
import pickle
import numpy as np

def find_indexes(p, data):
    c = np.zeros(60)
    idx = []
    for index, i in enumerate(data):
        if c[i] < p:
            c[i] += 1
            idx.append(index)
        if np.sum(c)==p*60:
            return idx


def main(s, n):
    root = '/Volumes/Untitled/Jeff/Project/2s-AGCN-master/data/ntu/xsub/'
    t1 = time.time()
    y = np.load(root+s+'_label.pkl', allow_pickle=True)[1]
    t2 = time.time()

    indices = find_indexes(n, y)
    x = np.load(root+s+'_data_joint.npy')[indices]
    y_temp = np.load(root+s+'_label.pkl', allow_pickle=True) 
    y0 = [y_temp[0][i] for i in indices]
    y1 = [y_temp[1][i] for i in indices]
    y = [y0, y1]

    des = '/Users/jeff/Temp/'
    np.save(des+s+'_data_joint_'+str(n)+'.npy', x)
    with open(des+s+'_label_'+str(n)+'.pkl', 'wb') as f:
        pickle.dump(y, f)

    print((t2-t1), 'seconds')
    print(np.load(des+s+'_data_joint_'+str(n)+'.npy').shape)
    print(len(np.load(des+s+'_label_'+str(n)+'.pkl', allow_pickle=True)))
    print(len(np.load(des+s+'_label_'+str(n)+'.pkl', allow_pickle=True)[1]))
    print()



if __name__ == '__main__':
    main('val', 2)
    main('train', 10)

