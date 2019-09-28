import torch
from keras.datasets import mnist
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from Q1 import Q1, WSampler, to_torch
import matplotlib.pyplot as plt
import pickle

def draw_mnist(X, name):
    wa = np.reshape(X, (28, 28))
    plt.imshow(wa,cmap='gray')
    plt.savefig(name)

def split_consistency():
    (X_tr, Y_tr), (X_t, Y_t) = mnist.load_data()
    X_tr, X_t = X_tr / 255, X_t / 255

    w_sampler = WSampler(X_tr, 60000 * [1])
    
    q1 = Q1((1,28,28)).cuda()
    q1.load('./q1.mdl')

    q1_rec = Q1((1,28,28)).cuda()
    q1_rec.load('./q1_recovered.mdl')
    
    q1_descr = q1.np_describe(X_t)
    q1_rec_descr = q1_rec.np_describe(X_t)

    q1_amax = np.argmax(q1_descr, axis=1)
    q1_rec_amax = np.argmax(q1_rec_descr, axis=1)
    print (q1_amax == q1_rec_amax)
    print (np.sum(q1_amax == q1_rec_amax)/len(q1_amax))

def num_split_consistency():
    (X_tr, Y_tr), (X_t, Y_t) = mnist.load_data()
    X_tr, X_t = X_tr / 255, X_t / 255


    num_rec = Q1((1,28,28)).cuda()
    num_rec.load('./num_recovered.mdl')
    
    num_descr = num_rec.np_describe(X_t)
    cor = 0
    for i,ha in enumerate(num_descr):
        if ha[0] > ha[1] and Y_t[i] < 5:
            cor += 1
        if ha[0] < ha[1] and Y_t[i] > 5:
            cor += 1

    print (cor / len(Y_t))

if __name__ == '__main__':
    num_split_consistency()
