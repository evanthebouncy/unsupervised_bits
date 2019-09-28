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
    def __init__(self, X, W, y=None):
        self.X, self.W, self.y = X, W, y

    def get_sample(self, n):
        W = self.W
        if n > len(W):
            n = len(W)
        prob = np.array(W) / np.sum(W)
        r_idx = np.random.choice(range(len(W)), size=n, replace=False, p=prob)
        if self.y is None:
            return self.X[r_idx]
        else:
            return self.X[r_idx], self.y[r_idx]

class Q1(nn.Module):

    # init with (channel, height, width) and out_dim for classiication
    def __init__(self, ch_h_w):
        super(Q1, self).__init__()
        self.name = "Q1"

        self.ch, self.h, self.w = ch_h_w

        self.conv1 = nn.Conv2d(self.ch, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.fea_fcs = nn.Linear(320, 100)

        self.q_split = nn.Linear(100, 2)
        self.digit_split = nn.Linear(100, 10)

        self.opt = torch.optim.RMSprop(self.parameters(), lr=0.00001)

    def encode(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fea_fcs(x))
        return x
  
    def describe(self, x):
        x = self.encode(x)
        features = F.softmax(self.q_split(x), dim=1)
        return features

    def classify(self, x):
        x = self.encode(x)
        features = F.softmax(self.digit_split(x), dim=1)
        return features

    def np_describe(self, X):
        X = to_torch(X, "float").view(-1, self.ch, self.h, self.w)
        return self.describe(X).detach().cpu().numpy()

    def entropy_loss(self, probs):
        p_yes = probs[:,0]
        p_no = probs[:,1]
        
        # get entropy of an unnormalized p
        def ent(p):
            # renormalise
            p = p + 1e-5
            p = p / torch.sum(p)
            # get entropy
            return - torch.sum(torch.log(p) * p)

        return ent(p_yes) * torch.sum(p_yes) + ent(p_no) * torch.sum(p_no)

    def learn_once(self, X):
        X = to_torch(X, "float").view(-1, self.ch, self.h, self.w)

        # optimize 
        self.opt.zero_grad()
        fea = self.describe(X)
        loss = self.entropy_loss(fea)
        loss.backward()
        self.opt.step()
  
        return loss

    def learn_label_once(self, X,y, label_kind):
        X = to_torch(X, "float").view(-1, self.ch, self.h, self.w)
        y = to_torch(y, "float")

        # optimize 
        self.opt.zero_grad()
        if label_kind is "q":
            fea = self.describe(X) + 1e-5
        else:
            fea = self.classify(X) + 1e-5
        loss = - torch.sum(y * torch.log(fea))
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

    w_sampler = WSampler(X_tr, 60000 * [1], Y_tr)
    
    q1 = Q1((1,28,28)).cuda()
    
    for i in range(100000):
        X,y = w_sampler.get_sample(100)

        y_1hot = np.zeros((100,10))
        y_1hot[np.arange(100), y] = 1

        loss_q = q1.learn_once(X)
        loss_digit = q1.learn_label_once(X,y_1hot,"digit")
        if i % 1000 == 0:
            print (f"loss {loss_q} digit_loss {loss_digit}")
            q1.save('./q1.mdl')

