# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:25:35 2019

@author: WT
"""

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import pickle
import os

def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

class gcn(nn.Module):
    def __init__(self, X_size, A_hat, bias=True): # X_size = num features
        super(gcn, self).__init__()
        self.A_hat = torch.tensor(A_hat).float()
        self.weight = nn.parameter.Parameter(torch.FloatTensor(X_size, 130))
        var = 2./(self.weight.size(1)+self.weight.size(0))
        self.weight.data.normal_(0,var)
        self.weight2 = nn.parameter.Parameter(torch.FloatTensor(130, 90))
        var2 = 2./(self.weight2.size(1)+self.weight2.size(0))
        self.weight2.data.normal_(0,var2)
        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(130))
            self.bias.data.normal_(0,var)
            self.bias2 = nn.parameter.Parameter(torch.FloatTensor(90))
            self.bias2.data.normal_(0,var2)
        else:
            self.register_parameter("bias", None)
        self.fc1 = nn.Linear(90,66)
        
    def forward(self, X, selected):
        X = F.relu(torch.mm(X, self.weight))
        if self.bias is not None:
            X = (X + self.bias)
        s = torch.mm(torch.tensor(self.A_hat).float(), torch.tensor(X).float()).numpy()[selected]
        X = F.relu(torch.mm(torch.tensor(s).float(), self.weight2))
        if self.bias2 is not None:
            X = (X + self.bias2)
        return self.fc1(X)

def evaluate(output, labels_e):
    _, labels = output.max(1); labels = labels.numpy()
    return sum(labels_e == labels)/len(labels)

df_data = load_pickle("df_data.pkl")
G = load_pickle("text_graph.pkl")
A = nx.to_numpy_matrix(G, weight="weight"); A = A + np.eye(G.number_of_nodes())
degrees = []
for d in G.degree(weight=None):
    if d == 0:
        degrees.append(0)
    else:
        degrees.append(d[1]**(-0.5))
degrees = np.diag(degrees)
X = np.eye(G.number_of_nodes()) # Features are just identity matrix
A_hat = degrees@A@degrees
f = A_hat@X # (n X n) X (n X n) x (n X n) X (n X n) input of net

### stratified test samples
test_idxs = []
for b_id in df_data["b"].unique():
    dum = df_data[df_data["b"] == b_id]
    if len(dum) >= 4:
        test_idxs.extend(list(np.random.choice(dum.index, size=round(0.2*len(dum)), replace=False)))
        
# select only certain labelled nodes for semi-supervised GCN
selected = []
for i in range(len(df_data)):
    if i not in test_idxs:
        selected.append(i)

f_selected = f[selected]; f_selected = torch.from_numpy(f_selected).float()
labels_selected = [l for idx, l in enumerate(df_data["b"]) if idx in selected]
f_not_selected = f[test_idxs]; f_not_selected = torch.from_numpy(f_not_selected).float()
labels_not_selected = [l for idx, l in enumerate(df_data["b"]) if idx not in selected]
f = torch.from_numpy(f).float()

net = gcn(X.shape[1], A_hat)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0007)

net.train()
losses_per_epoch = []; evaluation_trained = []; evaluation_untrained = []
for e in range(5000):
    optimizer.zero_grad()
    output = net(f, selected)
    loss = criterion(output, torch.tensor(labels_selected).long() -1)
    losses_per_epoch.append(loss.item())
    loss.backward()
    optimizer.step()
    if e % 10 == 0:
        ### Evaluate other untrained nodes and check accuracy of labelling
        net.eval()
        pred_labels = net(f, test_idxs)
        trained_accuracy = evaluate(output, labels_selected); untrained_accuracy = evaluate(pred_labels, labels_not_selected)
        evaluation_trained.append((e, trained_accuracy)); evaluation_untrained.append((e, untrained_accuracy))
        print("Evaluation accuracy of trained, untrained nodes respectively: ", trained_accuracy,\
              untrained_accuracy)
        net.train()
evaluation_trained = np.array(evaluation_trained); evaluation_untrained = np.array(evaluation_untrained) 

fig = plt.figure(figsize=(13,13))
ax = fig.add_subplot(111)
ax.scatter([i for i in range(1,len(losses_per_epoch)+1)], losses_per_epoch)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Loss vs Epoch")

fig = plt.figure(figsize=(13,13))
ax = fig.add_subplot(111)
ax.scatter(evaluation_trained[:,0], evaluation_trained[:,1])
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy on trained nodes")
ax.set_title("Accuracy (trained nodes) vs Epoch")

fig = plt.figure(figsize=(13,13))
ax = fig.add_subplot(111)
ax.scatter(evaluation_untrained[:,0], evaluation_untrained[:,1])
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy on untrained nodes")
ax.set_title("Accuracy (untrained nodes) vs Epoch")