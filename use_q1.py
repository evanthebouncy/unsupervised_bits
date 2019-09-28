import torch
from keras.datasets import mnist
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from Q1_grounded import Q1, WSampler, to_torch
import matplotlib.pyplot as plt
import pickle

def draw_mnist(X, name):
    wa = np.reshape(X, (28, 28))
    plt.imshow(wa,cmap='gray')
    plt.savefig(name)

def draw_split():
    (X_tr, Y_tr), (X_t, Y_t) = mnist.load_data()
    X_tr, X_t = X_tr / 255, X_t / 255

    w_sampler = WSampler(X_t, len(Y_t) * [1], Y_t)
    
    q1 = Q1((1,28,28)).cuda()
    q1.load('./q1.mdl')
    
    X,y = w_sampler.get_sample(100)
    description = q1.np_describe(X)
    for i, des in enumerate(description):
        print (des)
        if des[0] > des[1]:
            draw_mnist(X[i], f"drawings/{y[i]}_T_{i}.png")
        else:
            draw_mnist(X[i], f"drawings/{y[i]}_F_{i}.png")

def data_split():
    (X_tr, Y_tr), (X_t, Y_t) = mnist.load_data()
    X_tr, X_t = X_tr / 255, X_t / 255
    
    q1 = Q1((1,28,28)).cuda()
    q1.load('./q1.mdl')
    
    group1, group2 = [], []
    for i in range(len(Y_tr) // 1000):
        data = X_tr[i * 1000 : (i+1) * 1000]
        description = q1.np_describe(data)
        for j, des in enumerate(description):
            idxx = i * 1000 + j
            if des[0] > des[1]:
                group1.append(idxx)
            else:
                group2.append(idxx)
    pickle.dump((group1, group2), open("splitt/learned_split.p","wb"))

def data_split_num():
    (X_tr, Y_tr), (X_t, Y_t) = mnist.load_data()
    X_tr, X_t = X_tr / 255, X_t / 255
    
    
    group1, group2 = [], []
    for i in range(len(Y_tr) // 1000):
        data = X_tr[i * 1000 : (i+1) * 1000]
        descc = Y_tr[i * 1000 : (i+1) * 1000]
        for j, des in enumerate(descc):
            idxx = i * 1000 + j
            if des <5:
                group1.append(idxx)
            else:
                group2.append(idxx)
    pickle.dump((group1, group2), open("splitt/num_split.p","wb"))

if __name__ == '__main__':
    #data_split_num()
    draw_split()
