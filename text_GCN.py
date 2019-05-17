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
import pickle
import os

def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

class gcn(nn.Module):
    def __init__(self, X_size, A_hat, bias=True): # X_size = num features
        super(gcn, self).__init__()
        self.A_hat = torch.tensor(A_hat, requires_grad=False).float()
        self.weight = nn.parameter.Parameter(torch.FloatTensor(X_size, 330))
        var = 2./(self.weight.size(1)+self.weight.size(0))
        self.weight.data.normal_(0,var)
        self.weight2 = nn.parameter.Parameter(torch.FloatTensor(330, 130))
        var2 = 2./(self.weight2.size(1)+self.weight2.size(0))
        self.weight2.data.normal_(0,var2)
        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(330))
            self.bias.data.normal_(0,var)
            self.bias2 = nn.parameter.Parameter(torch.FloatTensor(130))
            self.bias2.data.normal_(0,var2)
        else:
            self.register_parameter("bias", None)
        self.fc1 = nn.Linear(130,66)
        
    def forward(self, X): ### 2-layer GCN architecture
        X = torch.mm(X, self.weight)
        if self.bias is not None:
            X = (X + self.bias)
        X = F.relu(torch.mm(self.A_hat, X))
        X = torch.mm(X, self.weight2)
        if self.bias2 is not None:
            X = (X + self.bias2)
        X = F.relu(torch.mm(self.A_hat, X))
        return self.fc1(X)

def evaluate(output, labels_e):
    _, labels = output.max(1); labels = labels.numpy()
    return sum([(e-1) for e in labels_e] == labels)/len(labels)

### Loads model and optimizer states
def load(net, optimizer, load_best=True):
    base_path = "./data/"
    if load_best == False:
        checkpoint = torch.load(os.path.join(base_path,"checkpoint.pth.tar"))
    else:
        checkpoint = torch.load(os.path.join(base_path,"model_best.pth.tar"))
    start_epoch = checkpoint['epoch']
    best_pred = checkpoint['best_acc']
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return start_epoch, best_pred

if __name__ == "__main__":
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
    f = X # (n X n) X (n X n) x (n X n) X (n X n) input of net
    
    ### stratified test samples
    test_idxs = []
    for b_id in df_data["b"].unique():
        dum = df_data[df_data["b"] == b_id]
        if len(dum) >= 4:
            test_idxs.extend(list(np.random.choice(dum.index, size=round(0.1*len(dum)), replace=False)))
    save_as_pickle("test_idxs.pkl", test_idxs)
    # select only certain labelled nodes for semi-supervised GCN
    selected = []
    for i in range(len(df_data)):
        if i not in test_idxs:
            selected.append(i)
    save_as_pickle("selected.pkl", selected)
    
    f_selected = f[selected]; f_selected = torch.from_numpy(f_selected).float()
    labels_selected = [l for idx, l in enumerate(df_data["b"]) if idx in selected]
    f_not_selected = f[test_idxs]; f_not_selected = torch.from_numpy(f_not_selected).float()
    labels_not_selected = [l for idx, l in enumerate(df_data["b"]) if idx not in selected]
    f = torch.from_numpy(f).float()
    save_as_pickle("labels_selected.pkl", labels_selected)
    save_as_pickle("labels_not_selected.pkl", labels_not_selected)    
    
    net = gcn(X.shape[1], A_hat)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.009)
    try:
        start_epoch, best_pred = load(net, optimizer, load_best=True)
    except:
        start_epoch = 0; best_pred = 0
    stop_epoch = 100; end_epoch = 7000
    
    net.train()
    losses_per_epoch = []; evaluation_trained = []; evaluation_untrained = []
    for e in range(start_epoch,end_epoch):
        optimizer.zero_grad()
        output = net(f)
        loss = criterion(output[selected], torch.tensor(labels_selected).long() -1)
        losses_per_epoch.append(loss.item())
        loss.backward()
        optimizer.step()
        if e % 50 == 0:
            ### Evaluate other untrained nodes and check accuracy of labelling
            net.eval()
            pred_labels = net(f)
            trained_accuracy = evaluate(output[selected], labels_selected); untrained_accuracy = evaluate(pred_labels[test_idxs], labels_not_selected)
            evaluation_trained.append((e, trained_accuracy)); evaluation_untrained.append((e, untrained_accuracy))
            print("Evaluation accuracy of trained, untrained nodes respectively: ", trained_accuracy,\
                  untrained_accuracy)
            print(output[selected].max(1)[1])
            net.train()
            if trained_accuracy > best_pred:
                best_pred = trained_accuracy
                torch.save({
                        'epoch': e + 1,\
                        'state_dict': net.state_dict(),\
                        'best_acc': trained_accuracy,\
                        'optimizer' : optimizer.state_dict(),\
                    }, os.path.join("./data/" ,"model_best.pth.tar"))
        if (e % 250) == 0:
            torch.save({
                    'epoch': e + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': trained_accuracy,\
                    'optimizer' : optimizer.state_dict(),\
                }, os.path.join("./data/" ,"checkpoint.pth.tar"))
    
    evaluation_trained = np.array(evaluation_trained); evaluation_untrained = np.array(evaluation_untrained)
    save_as_pickle("loss_vs_epoch.pkl", losses_per_epoch)
    save_as_pickle("evaluation_trained.pkl", evaluation_trained)
    save_as_pickle("evaluation_untrained.pkl", evaluation_untrained)
    
    fig = plt.figure(figsize=(13,13))
    ax = fig.add_subplot(111)
    ax.scatter([i for i in range(start_epoch,end_epoch)], losses_per_epoch)
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Loss", fontsize=15)
    ax.set_title("Loss vs Epoch", fontsize=20)
    plt.savefig(os.path.join("./data/", "loss_vs_epoch.png"))
    
    fig = plt.figure(figsize=(13,13))
    ax = fig.add_subplot(111)
    ax.scatter(evaluation_trained[:,0], evaluation_trained[:,1])
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Accuracy on trained nodes", fontsize=15)
    ax.set_title("Accuracy (trained nodes) vs Epoch", fontsize=20)
    plt.savefig(os.path.join("./data/", "trained_accuracy_vs_epoch.png"))
    
    fig = plt.figure(figsize=(13,13))
    ax = fig.add_subplot(111)
    ax.scatter(evaluation_untrained[:,0], evaluation_untrained[:,1])
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Accuracy on untrained nodes", fontsize=15)
    ax.set_title("Accuracy (untrained nodes) vs Epoch", fontsize=20)
    plt.savefig(os.path.join("./data/", "untrained_accuracy_vs_epoch.png"))
    
    fig = plt.figure(figsize=(13,13))
    ax = fig.add_subplot(111)
    ax.scatter(evaluation_trained[:,0], evaluation_trained[:,1], c="red", marker="v", \
               label="Trained Nodes")
    ax.scatter(evaluation_untrained[:,0], evaluation_untrained[:,1], c="blue", marker="o",\
               label="Untrained Nodes")
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Accuracy", fontsize=15)
    ax.set_title("Accuracy vs Epoch", fontsize=20)
    ax.legend(fontsize=20)
    plt.savefig(os.path.join("./data/", "combined_plot_accuracy_vs_epoch.png"))