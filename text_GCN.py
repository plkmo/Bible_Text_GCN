# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:25:35 2019

@author: WT
"""

import os
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from generate_train_test_datasets import load_pickle, save_as_pickle, generate_text_graph
from models import gcn
from evaluate_results import evaluate_model_results
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def load_datasets(args):
    """Loads dataset and graph if exists, else create and process them from raw data
    Returns --->
    f: torch tensor input of GCN (Identity matrix)
    X: input of GCN (Identity matrix)
    A_hat: transformed adjacency matrix A
    selected: indexes of selected labelled nodes for training
    test_idxs: indexes of not-selected nodes for inference/testing
    labels_selected: labels of selected labelled nodes for training
    labels_not_selected: labels of not-selected labelled nodes for inference/testing
    """
    logger.info("Loading data...")
    df_data_path = "./data/df_data.pkl"
    graph_path = "./data/text_graph.pkl"
    if not os.path.isfile(df_data_path) or not os.path.isfile(graph_path):
        logger.info("Building datasets and graph from raw data... Note this will take quite a while...")
        generate_text_graph()
    df_data = load_pickle("df_data.pkl")
    G = load_pickle("text_graph.pkl")
    
    logger.info("Building adjacency and degree matrices...")
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
    
    logger.info("Splitting labels for training and inferring...")
    ### stratified test samples
    test_idxs = []
    for b_id in df_data["b"].unique():
        dum = df_data[df_data["b"] == b_id]
        if len(dum) >= 4:
            test_idxs.extend(list(np.random.choice(dum.index, size=round(args.test_ratio*len(dum)), replace=False)))
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
    logger.info("Split into %d train and %d test lebels." % (len(labels_selected), len(labels_not_selected)))
    return f, X, A_hat, selected, labels_selected, labels_not_selected, test_idxs
    
def load_state(net, optimizer, scheduler, model_no=0, load_best=False):
    """ Loads saved model and optimizer states if exists """
    logger.info("Initializing model and optimizer states...")
    base_path = "./data/"
    checkpoint_path = os.path.join(base_path,"test_checkpoint_%d.pth.tar" % model_no)
    best_path = os.path.join(base_path,"test_model_best_%d.pth.tar" % model_no)
    start_epoch, best_pred, checkpoint = 0, 0, None
    if (load_best == True) and os.path.isfile(best_path):
        checkpoint = torch.load(best_path)
        logger.info("Loaded best model.")
    elif os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        logger.info("Loaded checkpoint model.")
    if checkpoint != None:
        start_epoch = checkpoint['epoch']
        best_pred = checkpoint['best_acc']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("Loaded model and optimizer.")    
    return start_epoch, best_pred

def load_results(model_no=0):
    """ Loads saved results if exists """
    losses_path = "./data/test_losses_per_epoch_%d.pkl" % model_no
    accuracy_path = "./data/test_accuracy_per_epoch_%d.pkl" % model_no
    if os.path.isfile(losses_path) and os.path.isfile(accuracy_path):
        losses_per_epoch = load_pickle("test_losses_per_epoch_%d.pkl" % model_no)
        accuracy_per_epoch = load_pickle("test_accuracy_per_epoch_%d.pkl" % model_no)
        logger.info("Loaded results buffer")
    else:
        losses_per_epoch, accuracy_per_epoch = [], []
    return losses_per_epoch, accuracy_per_epoch

def evaluate(output, labels_e):
    _, labels = output.max(1); labels = labels.numpy()
    return sum([(e-1) for e in labels_e] == labels)/len(labels)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hidden_size_1", type=int, default=330, help="Size of first GCN hidden weights")
    parser.add_argument("--hidden_size_2", type=int, default=130, help="Size of second GCN hidden weights")
    parser.add_argument("--num_classes", type=int, default=66, help="Number of prediction classes")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of test to training nodes")
    parser.add_argument("--num_epochs", type=int, default=3300, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.011, help="learning rate")
    parser.add_argument("--model_no", type=int, default=0, help="Model ID")
    args = parser.parse_args()
    save_as_pickle("args.pkl", args)
    
    f, X, A_hat, selected, labels_selected, labels_not_selected, test_idxs = load_datasets(args)
    net = gcn(X.shape[1], A_hat, args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000,2000,3000,4000,5000,6000], gamma=0.77)
    
    start_epoch, best_pred = load_state(net, optimizer, scheduler, model_no=args.model_no, load_best=True)
    losses_per_epoch, evaluation_untrained = load_results(model_no=args.model_no)
    
    logger.info("Starting training process...")
    net.train()
    evaluation_trained = []
    for e in range(start_epoch, args.num_epochs):
        optimizer.zero_grad()
        output = net(f)
        loss = criterion(output[selected], torch.tensor(labels_selected).long() -1)
        losses_per_epoch.append(loss.item())
        loss.backward()
        optimizer.step()
        if e % 50 == 0:
            ### Evaluate other untrained nodes and check accuracy of labelling
            net.eval()
            with torch.no_grad():
                pred_labels = net(f)
                trained_accuracy = evaluate(output[selected], labels_selected); untrained_accuracy = evaluate(pred_labels[test_idxs], labels_not_selected)
            evaluation_trained.append((e, trained_accuracy)); evaluation_untrained.append((e, untrained_accuracy))
            print("[Epoch %d]: Evaluation accuracy of trained nodes: %.7f" % (e, trained_accuracy))
            print("[Epoch %d]: Evaluation accuracy of test nodes: %.7f" % (e, untrained_accuracy))
            print("Labels of trained nodes: \n", output[selected].max(1)[1])
            net.train()
            if trained_accuracy > best_pred:
                best_pred = trained_accuracy
                torch.save({
                    'epoch': e + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': trained_accuracy,\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                }, os.path.join("./data/" ,\
                    "test_model_best_%d.pth.tar" % args.model_no))
        if (e % 250) == 0:
            save_as_pickle("test_losses_per_epoch_%d.pkl" % args.model_no, losses_per_epoch)
            save_as_pickle("test_accuracy_per_epoch_%d.pkl" % args.model_no, evaluation_untrained)
            torch.save({
                    'epoch': e + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': trained_accuracy,\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                }, os.path.join("./data/",\
                    "test_checkpoint_%d.pth.tar" % args.model_no))
        scheduler.step()
        
    logger.info("Finished training!")
    evaluation_trained = np.array(evaluation_trained); evaluation_untrained = np.array(evaluation_untrained)
    save_as_pickle("test_losses_per_epoch_%d_final.pkl" % args.model_no, losses_per_epoch)
    save_as_pickle("train_accuracy_per_epoch_%d_final.pkl" % args.model_no, evaluation_trained)
    save_as_pickle("test_accuracy_per_epoch_%d_final.pkl" % args.model_no, evaluation_untrained)
    
    fig = plt.figure(figsize=(13,13))
    ax = fig.add_subplot(111)
    ax.scatter([i for i in range(len(losses_per_epoch))], losses_per_epoch)
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
    
    logger.info("Evaluate results...")
    evaluate_model_results(args=args)