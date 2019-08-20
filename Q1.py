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

class Q1(nn.Module):

    # init with (channel, height, width) and out_dim for classiication
    def __init__(self, ch_h_w):
        super(Q1, self).__init__()
        self.name = "Q1"

        self.ch, self.h, self.w = ch_h_w

        self.conv1 = nn.Conv2d(self.ch, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.fea_fcs = nn.Linear(320, 2)

        self.opt = torch.optim.RMSprop(self.parameters(), lr=0.00001)
  
    def describe(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        features = F.softmax(self.fea_fcs(x), dim=1)
        return features

    def entropy_loss(self, probs):
        p_yes = probs[:,0]
        p_no = probs[:,1]
        
        def ent(p):
            p_tot = torch.sum(p)
            p = p + 1e-5
            p = p / torch.sum(p)
            return - torch.sum(torch.log(p) * p) * p_tot

        return ent(p_yes) + ent(p_no)

    def learn_once(self, X):
        X = to_torch(X, "float").view(-1, self.ch, self.h, self.w)

        # optimize 
        self.opt.zero_grad()
        fea = self.describe(X)
        loss = self.entropy_loss(fea)
        loss.backward()
        self.opt.step()
  
        return loss

    def save(self, loc):
        torch.save(self.state_dict(), loc)

    def load(self, loc):
        self.load_state_dict(torch.load(loc))

if __name__ == '__main__':

    (X_tr, Y_tr), (X_t, Y_t) = mnist.load_data()
    X_tr, X_t = X_tr / 255, X_t / 255

    w_sampler = WSampler(X_tr, 60000 * [1])
    
    q1 = Q1((1,28,28)).cuda()
    
    for i in range(100000):
        X = w_sampler.get_sample(100)
        losss = q1.learn_once(X)
        if i % 1000 == 0:
            print (losss)
            q1.save('./q1.mdl')

