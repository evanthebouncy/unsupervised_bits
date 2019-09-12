import torch
from keras.datasets import mnist
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time

from Q1 import to_torch, WSampler, Q1
from Q20 import data_split
from use_q1 import draw_mnist
import tqdm
import random
NUM_Q = 40

def get_hash(qs, X):
    dess = np.array([qq.np_describe(X) for qq in [x[1] for x in q10]])
    xx = np.transpose(dess, (1,0,2))
    ret = np.argmax(xx, axis=2)
    return ret

class Game:
    def __init__(self, X):
        self.the_item = random.choice(X)

    def answer(self, qry):
        return qry.np_describe(np.array([self.the_item]))[0]

class Agent:
    def __init__(self, qs, cur_set):
        self.qs = qs
        self.cur_set = cur_set

    def best_question(self):
        X = self.cur_set
        qs = self.qs
        losss = 9999
        to_ret = None, None
        for qname, qq in qs:
            grp1, grp2 = data_split(qq, X)
            loss = abs(len(grp1) - len(grp2))
            if loss < losss:
                losss = loss
                to_ret = (qname, qq)
        return to_ret

    def update_cur_set(self, qry, result):
        grp1, grp2 = data_split(qry, self.cur_set)
        if result[0] > result[1]:
            self.cur_set = grp1
        else:
            self.cur_set = grp2

def plot_hash():
    (X_tr, Y_tr), (X_t, Y_t) = mnist.load_data()
    X_tr, X_t = X_tr / 255, X_t / 255
    w_sampler = WSampler(X_tr, 60000 * [1])

    q10 = [(f'q_{i+1}', Q1((1,28,28)).cuda()) for i in range(NUM_Q)]
    for qq_name, qq in q10:
        qq.load(f'./saved_models/20q/{qq_name}.mdl')

    print ("models loaded")

    X = w_sampler.get_sample(40)
    X_hash = get_hash(q10, X)

    dups = {}
    for i, h in enumerate(X_hash):
        h_key = tuple(h)
        if h_key not in dups:
            dups[h_key] = []
        dups[h_key].append(i)

    print (f"all key size {len(dups.keys())}")
    # for dkey in dups:
    #     prefix = ''.join([str(z) for z in dkey])
    #     for jj, y in enumerate(dups[dkey]):
    #         draw_mnist(X[y], f"drawings/{prefix}_{jj}.png")
    for dkey in dups:
        for jj, y in enumerate(dups[dkey]):
            if dkey[22] == 0:
                draw_mnist(X[y], f"drawings/0_{y}.png")
            else:
                draw_mnist(X[y], f"drawings/1_{y}.png")

if __name__ == '__main__':
    (X_tr, Y_tr), (X_t, Y_t) = mnist.load_data()
    X_tr, X_t = X_tr / 255, X_t / 255
    w_sampler = WSampler(X_tr, 60000 * [1])

    q10 = [(f'q_{i+1}', Q1((1,28,28)).cuda()) for i in range(NUM_Q)]
    for qq_name, qq in q10:
        qq.load(f'./saved_models/20q/{qq_name}.mdl')

    X = w_sampler.get_sample(1000)
    oracle = Game(X)
    agent = Agent(q10, X)

    for i in range(20):
        print (f"question {i}, cur set size {len(agent.cur_set)}")
        if len(agent.cur_set) == 1:
            print ("we are done")
            break
        best_qry = agent.best_question()
        print (f"best query is {best_qry[0]}")
        ans = oracle.answer(best_qry[1])
        agent.update_cur_set(best_qry[1], ans)

    print ("verifying same-ness")
    print (np.sum(oracle.the_item - agent.cur_set[0]))
