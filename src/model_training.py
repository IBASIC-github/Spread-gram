#-*- encoding: utf-8 -*-
'''
Created on 2017年5月6日

@author: Baijie
'''

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import dataprepare as dp
import random, copy, time, math, os, copy
from scipy.special import expit
from gensim.models import Word2Vec
import numpy as np
import networkx as nx
from networkx.classes.function import neighbors

def construct_graph(edges):
    def add_dict(s,d,dic):
        if s in dic:
            dic[s].append(d)
        else:
            dic[s] = [d]
    item_nodes = {}
    node_items = {}
    for (i,n) in edges:
        add_dict(i, n, item_nodes)
        add_dict(n, i, node_items)
    return item_nodes, node_items

          
def writeModel(dic, path):
#     print len(dic)
    print("写入文件")
    entityVectorFile = open(path, 'w', encoding='utf-8')
    for (key,value) in dic.items():
        entityVectorFile.write(str(key) + '\t' + str(value.tolist()))
        entityVectorFile.write("\n")
    entityVectorFile.close()


def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=int)

    norm_const = sum(probs)
    probs = [float(u_prob) / norm_const for u_prob in probs]

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):

        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

def spreadgram_main(data, min_num=3, emb_size=128, repeat_time=5, k_negative_sampling=5, learning_rate=0.05):
    def add_t(s, t):
        #         global neighbor_t
        neighbor_tt = neighbor_t.setdefault(s, set())
        neighbor_tt.add(t)

    def add_dict(s, count_t):
        if s not in node_dict:
            node_dict[s] = count_t
            count_t += 1
        return count_t

    neighbor_t = {}
    node_dict = {}
    count_t = 0
    if data.isdirected:
        for n1, n2 in data:
            add_t(n1, n2)
            count_t = add_dict(n1, count_t)
            count_t = add_dict(n2, count_t)
    elif data.isheterogeneous:
        for _, n1, n2_list in data:
            for n2 in n2_list:
                add_t(n1, n2)
                add_t(n2, n1)
                count_t = add_dict(n1, count_t)
                count_t = add_dict(n2, count_t)
    else:
        for n1, n2 in data:
            add_t(n1, n2)
            add_t(n2, n1)
            count_t = add_dict(n1, count_t)
            count_t = add_dict(n2, count_t)

    node_neighbor = [[]] * count_t
    for (k, v) in neighbor_t.items():
        node_neighbor[node_dict[k]] = [node_dict[i] for i in v]
    del neighbor_t

    node_dict = dict((v, k) for k, v in node_dict.items())
    dp.write_dicts(node_dict, '../model/' + data.dataname + '/node_dict.txt')
    for it, node_vecs in spreadgram_vec(node_neighbor, dim=emb_size, repeat_time=repeat_time,
                                    k_negative_sampling=k_negative_sampling, learning_rate=learning_rate):
        node_vecs_new = dict((v, node_vecs[k]) for k, v in node_dict.items() if v)
        yield it, node_vecs_new


def spreadgram_vec(node_neighbor, dim=128, repeat_time=5, k_negative_sampling=5, learning_rate=0.05):
    node_count = len(node_neighbor)
    freq = [0] * node_count  # the number of freq lists varies according to the node types

    for (node, neighbor) in enumerate(node_neighbor):
        freq[node] = len(neighbor) * 1.0
    J, q = alias_setup(freq)

    node_vecs = np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=(node_count, dim)) / dim

    string = '0/' + str(repeat_time) + ', learning rate: ' + str(learning_rate)
    print(string)
    yield 0, node_vecs

    if repeat_time >= 1:
        for r in range(1, repeat_time + 1):
            string = str(r) + '/' + str(repeat_time) + ', learning rate: ' + str(learning_rate)
            print(string)

            node_list = []
            nodes = range(node_count)
            unused_nodes = set(nodes)
            node_queue = [random.choice(nodes)]
            while unused_nodes != set():
                while True:
                    if len(node_queue) != 0:
                        n_t = node_queue.pop(0)
                        if n_t in unused_nodes:
                            break
                    else:
                        n_t = random.choice(list(unused_nodes))
                        break
                node_list.append(n_t)
                context = node_neighbor[n_t]
                node_queue += [c for c in context if c in unused_nodes]
                unused_nodes.remove(n_t)

            sum_error = 0.0
            for nid in node_list:
                context = node_neighbor[nid]
                for c in context:
                    neu1e = np.zeros(dim)
                    ns = {nid: 1}
                    for i in range(k_negative_sampling):
                        while True:
                            #                         n_neg = random.randint(0, node_count-1)
                            n_neg = alias_draw(J, q)
                            if n_neg not in node_neighbor[c] and n_neg not in ns:
                                ns[n_neg] = 0
                                break
                    for n, y in ns.items():
                        z = np.dot(node_vecs[c], node_vecs[n])
                        p = expit(z)
                        g = learning_rate * (y - p)
                        neu1e += g * node_vecs[n]  # Error to back propagate to nn0
                        node_vecs[n] += g * node_vecs[c]  # Update nn1
                        sum_error += y - p
                    node_vecs[c] += neu1e

            learning_rate = learning_rate

            #             yield node_vecs
            print('Sum_error: ' + str(sum_error))
            #             yield r, np.array([node_vecs[i, :]/np.sqrt((node_vecs[i, :] ** 2).sum(-1)) for i in range(node_count)])
            yield r, np.array([node_vecs[i, :] for i in range(node_count)])


class SpreadGram(object):
    def __init__(self):
        self.methodname = 'spreadgram'
        
    def loadModel(self, data):
        file_path = '../model/' + data.dataname + '/' + self.methodname + 'model'
        fr = open(file_path)
        dic = {}
        for line in fr:
            line = line.strip().split("\t")
            dic[line[0]] = np.array(list(map(float, line[1][1:-1].split(', '))))
        print(len(dic))
        return dic
    def buildModel(self, data, dimensions = 128):
        repeat_time = 100
        for it, model in spreadgram_main(data, min_num = 5, emb_size = dimensions, repeat_time = repeat_time, k_negative_sampling = 2, learning_rate = 0.05):
            if it == repeat_time:
                file_path = '../model/' + data.dataname + '/' + self.methodname + 'model'
                print("模型写入")
                writeModel(model, file_path)
                print('finish')
                return model


if __name__ == '__main__':
    data = dp.Amazon()
    method = SpreadGram()
    print(method.methodname)
    model = method.buildModel(data)
    # model = method.loadModel(data)
