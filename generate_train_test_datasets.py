# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:28:24 2019

@author: WT
"""
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import numpy as np
import networkx as nx
from collections import OrderedDict
import math
from tqdm import tqdm

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

### remove stopwords and non-words from tokens list
def filter_tokens(tokens, stopwords):
    tokens1 = []
    for token in tokens:
        if (token not in stopwords) and (token not in [".",",",";","&","'s", ":", "?", "!","(",")",\
            "'","'m","'no","***","--","...","[","]"]):
            tokens1.append(token)
    return tokens1

def dummy_fun(doc):
    return doc

def word_word_edges(p_ij):
    dum = []; word_word = []; counter = 0
    cols = list(p_ij.columns); cols = [str(w) for w in cols]
    for w1 in tqdm(cols, total=len(cols)):
        for w2 in cols:
            #if (counter % 300000) == 0:
            #    print("Current Count: %d; %s %s" % (counter, w1, w2))
            if (w1 != w2) and ((w1,w2) not in dum) and (p_ij.loc[w1,w2] > 0):
                word_word.append((w1,w2,{"weight":p_ij.loc[w1,w2]})); dum.append((w2,w1))
            counter += 1
    return word_word

def pool_word_word_edges(w1):
    dum = []; word_word = {}
    for w2 in p_ij.index:
        if (w1 != w2) and ((w1,w2) not in dum) and (p_ij.loc[w1,w2] > 0):
            word_word = [(w1,w2,{"weight":p_ij.loc[w1,w2]})]; dum.append((w2,w1))
    return word_word

if __name__=="__main__":
    print("Preparing data...")
    datafolder = "./data/"
    df = pd.read_csv(os.path.join(datafolder,"t_bbe.csv"))
    df.drop(["id", "v"], axis=1, inplace=True)
    df = df[["t","c","b"]]
    book_dict = pd.read_csv(os.path.join(datafolder, "key_english.csv"))
    book_dict = {book.lower():number for book, number in zip(book_dict["field.1"], book_dict["field"])}
    stopwords = list(set(nltk.corpus.stopwords.words("english")))
    
    ### one chapter per document, labelled by book
    df_data = pd.DataFrame(columns=["c", "b"])
    for book in df["b"].unique():
        dum = pd.DataFrame(columns=["c", "b"])
        dum["c"] = df[df["b"] == book].groupby("c").apply(lambda x: (" ".join(x["t"])).lower())
        dum["b"] = book
        df_data = pd.concat([df_data,dum], ignore_index=True)
    del df
    
    ### tokenize & remove funny characters
    df_data["c"] = df_data["c"].apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: filter_tokens(x, stopwords))
    save_as_pickle("df_data.pkl", df_data)
    
    ### Tfidf
    vectorizer = TfidfVectorizer(input="content", max_features=None, tokenizer=dummy_fun, preprocessor=dummy_fun)
    vectorizer.fit(df_data["c"])
    df_tfidf = vectorizer.transform(df_data["c"])
    df_tfidf = df_tfidf.toarray()
    vocab = vectorizer.get_feature_names()
    vocab = np.array(vocab)
    df_tfidf = pd.DataFrame(df_tfidf,columns=vocab)
    
    ### PMI between words
    window = 10 # sliding window size to calculate point-wise mutual information between words
    names = vocab
    occurrences = OrderedDict((name, OrderedDict((name, 0) for name in names)) for name in names)
    # Find the co-occurrences:
    no_windows = 0; print("calculating co-occurences")
    for l in tqdm(df_data["c"], total=len(df_data["c"])):
        for i in range(len(l)-window):
            no_windows += 1
            d = l[i:(i+window)]; dum = []
            for x in range(len(d)):
                for item in d[:x] + d[(x+1):]:
                    if item not in dum:
                        occurrences[d[x]][item] += 1; dum.append(item)
            
    print("Calculating PMI...")
    df_occurences = pd.DataFrame(occurrences, columns=occurrences.keys())
    df_occurences = (df_occurences + df_occurences.transpose())/2 ## symmetrize it as window size on both sides may not be same
    del occurrences
    ### convert to PMI
    p_i = df_occurences.sum(axis=0)/no_windows
    p_ij = df_occurences/no_windows
    del df_occurences
    for col in p_ij.columns:
        p_ij[col] = p_ij[col]/p_i[col]
    for row in p_ij.index:
        p_ij.loc[row,:] = p_ij.loc[row,:]/p_i[row]
    p_ij = p_ij + 1E-9
    for col in p_ij.columns:
        p_ij[col] = p_ij[col].apply(lambda x: math.log(x))
        
    ### Build graph
    print("Building graph...")
    G = nx.Graph()
    G.add_nodes_from(df_tfidf.index) ## document nodes
    G.add_nodes_from(vocab) ## word nodes
    ### build edges between document-word pairs
    document_word = [(doc,w,{"weight":df_tfidf.loc[doc,w]}) for doc in df_tfidf.index for w in df_tfidf.columns]
    
    print("Building word-word edges")
    word_word = word_word_edges(p_ij)
    save_as_pickle("word_word_edges.pkl", word_word)
    G.add_edges_from(document_word)
    G.add_edges_from(word_word)
    save_as_pickle("text_graph.pkl", G)
    