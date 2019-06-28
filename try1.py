import torch
from keras.datasets import mnist
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time

if torch.cuda.is_available():
    def to_torch(x, dtype, req = False):
        tor_type = torch.cuda.LongTensor if dtype == "int" else torch.cuda.FloatTensor
        x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
        return x
else:
    def to_torch(x, dtype, req = False):
        tor_type = torch.LongTensor if dtype == "int" else torch.FloatTensor
        x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
        return x

class WSampler:

    # make the sampler
    def __init__(self, X, W):
        self.X, self.W = X, W

    def get_sample(self, n):
        W = self.W
        if n > len(W):
            n = len(W)
        prob = np.array(W) / np.sum(W)
        r_idx = np.random.choice(range(len(W)), size=n, replace=True, p=prob)
        return self.X[r_idx]

class CNN1(nn.Module):

    # init with (channel, height, width) and out_dim for classiication
    def __init__(self, ch_h_w, n_features):
        super(CNN1, self).__init__()
        self.name = "CNN1"

        self.ch, self.h, self.w = ch_h_w

        self.conv1 = nn.Conv2d(self.ch, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.fea_fcs = nn.Linear(320, n_features)

        self.opt = torch.optim.RMSprop(self.parameters(), lr=0.00001)
  
    def describe(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        features = torch.sigmoid(self.fea_fcs(x))
        return features

    def learn_once(self, X):
        X = to_torch(X, "float").view(-1, self.ch, self.h, self.w)

        ll = X.size()[0]
        assert ll % 2 == 0, "must be even"
        X1, X2 = X[:ll//2], X[ll//2:]
  
        # optimize 
        self.opt.zero_grad()
        fea1 = self.describe(X1)
        fea2 = self.describe(X2)
        loss = -torch.sum((fea1 - fea2) ** 2)
        loss.backward()
        self.opt.step()
  
        return loss

if __name__ == '__main__':

    (X_tr, Y_tr), (X_t, Y_t) = mnist.load_data()
    X_tr, X_t = X_tr / 255, X_t / 255

    w_sampler = WSampler(X_tr, 60000 * [1])
    
    cnn = CNN1((1,28,28), 10).cuda()
    
    for i in range(10000):
        X = w_sampler.get_sample(100)
        losss = cnn.learn_once(X)

        if i % 1000 == 0:
            feat1 = cnn.describe(to_torch(X_tr[:100], "float").view(-1,1,28,28))
            print (feat1)
            print (Y_tr[:100])

    while True:
        idx = int(input("idx plox\n"))
        feat1 = cnn.describe(to_torch(X_tr[idx:idx+1], "float").view(-1,1,28,28))
        print (feat1)
        print (Y_tr[idx:idx+1])


