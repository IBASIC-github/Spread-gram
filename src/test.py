# -*- encoding: utf-8 -*-
'''
Created on 2017年5月26日

@author: Baijie
'''
import numpy as np
from model_training import SpreadGram
import dataprepare as dp
import math, time, random, copy
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support

def output(string, logfilepath=None):
    print(string)
    if logfilepath != None:
        logfile = open(logfilepath, 'a')
        logfile.write(string)
        logfile.close()


def classification(X, Y, testsize=0.3):
    X = np.array(X)
    Y = np.array(Y)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    test_num = int(math.ceil(testsize * X.shape[0]))
    X_train = X[test_num:]
    X_test = X[:test_num]
    Y_train = Y[test_num:]
    Y_test = Y[:test_num]
    clf = SVC(C=2000).fit(X_train, Y_train)
    predicted = clf.predict(X_test)
    return Y_test, predicted



def multilabel(X, Y, testsize = 0.3):
    X = np.array(X)
    Y = np.array(Y)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    test_num = int(math.ceil(testsize*X.shape[0]))
    X_train = X[test_num:]
    X_test = X[:test_num]
    Y_train = Y[test_num:]
    Y_test = Y[:test_num]
    classif = DecisionTreeClassifier(criterion='entropy')#, max_leaf_nodes = 64, max_leaf_nodes = 8, splitter = 'random', max_depth = 3)
    classif.fit(X_train, Y_train)
    predicted = classif.predict(X_test)
    return Y_test, predicted
    # # auc = roc_auc_score(Y_test, predicted)
    # auc = 0.5
    # percision, recall, f1, s = precision_recall_fscore_support(Y_test, predicted,
    #                                                            average='weighted')
    # return auc


def link_prediction(data, model, logfilepath):
    def add_neighbor(s, t, dic):
        if s in dic:
            dic[s].append(t)
        else:
            dic[s] = [t]

    X, labels = [], []

    node_list = list(model.keys())
    if data.dataname in ['amazon', 'dblp']:
        # node_dict = dp.read_dicts(data.nodedictpath)
        for pub, item, nodes in data:
            if item in model and random.random() < 0.01:
                item = np.array(model[item])
                n_pos = random.choice(nodes)
                nodes = set(nodes)
                while True:
                    n_neg = random.choice(node_list)
                    if n_neg not in nodes:
                        break
                if n_pos in model and n_neg in model:
                    X.append(item - np.array(model[n_pos]))
                    X.append(item - np.array(model[n_neg]))
                    labels.append(1)
                    labels.append(0)
    else:
        node_neighbors = {}
        # candidate_neighbors = set()
        for node1, node2 in data:
            if node1 in model and node2 in model:
                add_neighbor(node1, node2, node_neighbors)
                add_neighbor(node2, node1, node_neighbors)
                # candidate_neighbors.add(node1)
                # candidate_neighbors.add(node2)
        # candidate_neighbors = list(candidate_neighbors)
        for node, neighbors in node_neighbors.items():
            n_pos = random.choice(neighbors)
            neighbors = set(neighbors)
            while True:
                n_neg = random.choice(node_list)
                if n_neg not in neighbors:
                    break
            if n_pos in model and n_neg in model:
                if node < n_pos:
                    node, n_pos = n_pos, node
                if node < n_neg:
                    node, n_neg = n_neg, node
                X.append(np.array(model[node]) - np.array(model[n_pos]))
                X.append(np.array(model[node]) - np.array(model[n_neg]))
                labels.append(1)
                labels.append(0)
    Y_test, predicted = classification(X, copy.deepcopy(labels), testsize=0.3)
    accuracy = np.mean([a == b for a, b in zip(predicted, Y_test)])

    output(str(accuracy), logfilepath)
    output(time.strftime('%Y-%m-%d %H:%M:%S'), logfilepath)


def item_classification(data, model, logfilepath):
    for label_size in range(2, data.group_nums):
        output('*******************************************\nLabel size: ' + str(label_size), logfilepath)
        if data.dataname in ['amazon', 'dblp']:
            label_item = dp.read_item_labels_h(data, label_size=label_size)
        else:
            label_item = []
            for (label, items) in list(data.collect_labels())[:label_size]:
                for item in items:
                    label_item.append((label, item))
        X, Y = [], []
        for (label, item) in label_item:
            if item in model:
                vec = np.array(model[item])
                X.append(vec)
                Y.append(label)
        Y_test, predicted = classification(X, Y, testsize=0.3)

        percision, recall, f1, s = precision_recall_fscore_support(Y_test, predicted,
                                                                   average='weighted')

        output(str(f1), logfilepath)
        output(time.strftime('%Y-%m-%d %H:%M:%S'), logfilepath)


def multilabel_classification(data, model, logfilepath):
    #     node_dict = dp.read_dicts(data.nodedictpath)
    for label_size in range(2, data.group_nums):
        output('*******************************************\nLabel size: ' + str(label_size), logfilepath)

        node_labels = dp.read_node_labels(data, label_size=label_size)
        print('Creating X and Y..')
        X, Y = [], []
        for (node, voc) in node_labels.items():
            if node in model:
                X.append(model[node])
                Y.append(voc)
        print('Classification..')
        Y_test, predicted = multilabel(X, Y, testsize=0.3)

        auc = roc_auc_score(Y_test, predicted)
        percision, recall, f1, s = precision_recall_fscore_support(Y_test, predicted, average='weighted')
        output('AUC score: ' + str(auc) + '\t f1:' + str(f1), logfilepath)
        output(time.strftime('%Y-%m-%d %H:%M:%S'), logfilepath)

if __name__ == '__main__':
    logfilepath = '../logs/test-20240927.txt'
    output(logfilepath, logfilepath)
    data = dp.Amazon()
    output('*******************************************\nDataset: ' + data.dataname, logfilepath)
    method = SpreadGram()
    model = method.loadModel(data)
    # link_prediction(data, model, logfilepath)
    # item_classification(data, model, logfilepath)
    multilabel_classification(data, model, logfilepath)


