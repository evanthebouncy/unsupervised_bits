import torch
from keras.datasets import mnist
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time

from Q1_grounded import to_torch, WSampler, Q1
import tqdm
NUM_Q = 10

# give a data batch X split it with qq
def data_split(qq, X):
    
    group1, group2 = [], []
    description = qq.np_describe(X)
    for j, des in enumerate(description):
        if des[0] > des[1]:
            group1.append(X[j])
        else:
            group2.append(X[j])
    return np.array(group1), np.array(group2)

def train_20q():
    (X_tr, Y_tr), (X_t, Y_t) = mnist.load_data()
    X_tr, X_t = X_tr / 255, X_t / 255

    w_sampler = WSampler(X_tr, 60000 * [1], Y_tr)
    
    q10 = [(f'q_{i+1}', Q1((1,28,28)).cuda()) for i in range(NUM_Q)]
    choose2 = lambda : np.random.choice([xx for xx in range(NUM_Q)],2,replace=False)
    
    # iterate over some epochs
    for i in tqdm.tqdm(range(1000000)):
        # get a random sample of 100
        X, Y = w_sampler.get_sample(200)
        y_1hot = np.zeros((200,10))
        y_1hot[np.arange(200), Y] = 1

        # get two questions
        q1_id, q2_id = choose2()
        q1, q2 = q10[q1_id][1], q10[q2_id][1]

        # ground the two questions
        loss_digit1 = q1.learn_label_once(X,y_1hot,"digit")
        loss_digit2 = q2.learn_label_once(X,y_1hot,"digit")
        # train individual losses on q1 and q2
        loss_q1 = q1.learn_once(X)
        loss_q2 = q2.learn_once(X)
        # make sure q1 and q2 are mutually independent
        q1_groups = data_split(q1, X)
        if len(q1_groups[0]) > 10:
            loss_1T2 = q2.learn_once(q1_groups[0])
        if len(q1_groups[1]) > 10:
            loss_1F2 = q2.learn_once(q1_groups[1])
        q2_groups = data_split(q2, X)
        if len(q2_groups[0]) > 10:
            loss_2T1 = q1.learn_once(q2_groups[0])
        if len(q2_groups[1]) > 10:
            loss_2F1 = q1.learn_once(q2_groups[1])

        if i % 1000 == 0:
            try:
                print (f"questions {q10[q1_id][0]} {q10[q2_id][0]}")
                print (loss_q1, loss_q2, loss_1T2, loss_1F2, loss_2T1, loss_2F1)
                print (f"first splits {len(q1_groups[0])} {len(q1_groups[1])}")
                q21_split = data_split(q2, q1_groups[0])
                print (f"second splits {len(q21_split[0])} {len(q21_split[1])}")
                q21_split = data_split(q2, q1_groups[1])
                print (f"second splits {len(q21_split[0])} {len(q21_split[1])}")
                print (f"grounded loss {loss_digit1} {loss_digit2}")
            except:
                pass
            for qq_name, qq in q10:
                qq.save(f'./saved_models/20q/{qq_name}.mdl')


if __name__ == '__main__':
    train_20q()
