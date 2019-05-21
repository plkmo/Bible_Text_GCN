# Graph Convolutional Network for Bible book classification

The text-based graph convolutional network (GCN) model is an interesting and novel state-of-the-art semi-supervised learning concept that is proposed recently, which is able to very accurately predict the labels of some unknown textual data given related known labeled textual data. It does so by embedding the entire corpus into a graph with documents and words as nodes, with each document-word & word-word edges having some predetermined weights based on their relationships with each other (eg. Tf-idf). A GCN is then trained on this graph with documents nodes that have known labels, and the trained GCN model is then used to infer the labels of unlabelled documents.

We implement text-GCN here using the Holy Bible as the corpus. The Holy Bible consists of 66 Books (Genesis, Exodus, etc) and 1189 Chapters. The goal here is to train a language model that is able to correctly classify the Book that some unlabelled Chapters belong to, given the labels of other Chapters. (Since we actually do know the exact labels of all Chapters, we intentionally mask the labels of some 10-20 % of the Chapters, which will be used as test set during model inference to measure the model accuracy) To do that, the language model needs to be able to distinguish between the contexts associated with the various Books (eg. Book of Genesis talks more about Adam & Eve while Book of Ecclesiastes talks about the life of King Solomon). The good results of the text-GCN model show that the graph structure is able to capture such context nicely, where the document (Chapter)-word edges encode the context within Chapters, while the word-word edges encode the relative context between Chapters.

The Bible text data used here (BBE version) is obtained courtesy of https://github.com/scrollmapper/bible_databases.
Implementation follows the paper on Text-based Graph Convolutional Network (https://arxiv.org/abs/1809.05679)
For more details on the scripts & implementation, see this article: https://towardsdatascience.com/text-based-graph-convolutional-network-for-semi-supervised-bible-book-classification-c71f6f61ff0f

Requirements: Python (3.6+), networkx (2.1), torch (1.0.0), torchvision (0.2.1), standard Python libraries

You will find the following:
1) generate_train_test_datasets.py - script to compute the edges weights, build and save the graph
2) text_GCN.py - Construct the GCN and trains the model
3) evaluate_results.py - evaluate the results and misclassified labels
4) Data folder containing the Bible data (t_bbe.csv)
