# coding: utf-8
import networkx as nx
from node2vec import Node2Vec
import argparse, re, os
from urllib import parse
import scipy.sparse as smat
import numpy as np
from rf_util import *
from tqdm import tqdm
import pickle as pkl
import scipy as sp
from xbert.rf_util import smat_util

def parse_mlc2seq_format(data_path):
    assert(os.path.isfile(data_path))
    with open(data_path) as fin:
        labels, corpus = [], []
        for line in fin:
            tmp = line.strip().split('\t', 1)
            labels.append(tmp[0])
            corpus.append(tmp[1])
    return labels, corpus

class N2V_embedder():
    def __init__(self, ans_list, outfile, dim=512, walk_length=80, num_walks=20):
        self.ans_list=ans_list
        self.dim = dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.outfile = outfile

    def gen_graph(self):
        graph = nx.Graph()
        for ans in tqdm(self.ans_list):
            label_list = list(map(int, ans.split(',')))
            for i in range(len(label_list)):
                for j in range(i+1, len(label_list)):
                    graph.add_edge(label_list[i], label_list[j])

        return graph

    def gen_embedding(self, graph):
        node2vec = Node2Vec(graph, dimensions=self.dim, walk_length=self.walk_length, \
                    num_walks=self.num_walks, workers=1, p=0.25, q=1)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        # return model
        model.wv.save_word2vec_format(self.outfile)

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-ds", "--dataset", default="AmazonCat-13K", type=str, required=True)
    parser.add_argument("-only_g", "--only_graph", default=0, type=int, required=True)
    args = parser.parse_args()
    ds_path = '../datasets/' + args.dataset

    labelfile = ds_path + '/mlc2seq/train.txt'
    outfile = ds_path + '/label_n2v_embedding'
    trn_labels, _ = parse_mlc2seq_format(labelfile)
    n2v = N2V_embedder(trn_labels, outfile)
    print('==============================================================')
    print('Generaing label graph...')
    graphfile = ds_path+'/label_graph'
    if os.path.isfile(graphfile):
        g = nx.read_edgelist(graphfile, create_using=nx.Graph())
    else:
        g = n2v.gen_graph()
        nx.write_edgelist(g, ds_path+'/label_graph')
    if args.only_graph:
        exit()
    print('==============================================================')
    print('Generaing label embeddings')
    n2v.gen_embedding(g)

if __name__ == '__main__':
    main()
