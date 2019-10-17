import networkx as nx
import argparse
import os, sys, time
import pickle as pkl
import scipy.sparse as smat
import numpy as np
from tqdm import tqdm
import pickle as pkl
import scipy as sp

class GraphUtil():
    def __init__(self, ans_list, num_labels, outfile):
        self.ans_list=ans_list
        self.outfile = outfile
        self.nums = np.zeros(num_labels)
        self.adj = np.zeros([num_labels, num_labels])

    def gen_graph(self):
        graph = nx.Graph()
        for ans in tqdm(self.ans_list):
            label_list = list(map(int, ans.split(',')))
            for i in range(len(label_list)):
                for j in range(i+1, len(label_list)):
                    self.adj[i][j] += 1
                    graph.add_edge(label_list[i], label_list[j])

        return graph



def parse_mlc2seq_format(data_path):
    assert(os.path.isfile(data_path))
    with open(data_path) as fin:
        labels, corpus = [], []
        for line in fin:
            tmp = line.strip().split('\t', 1)
            labels.append(tmp[0])
            corpus.append(tmp[1])
    return labels, corpus

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-ds", "--dataset", default="AmazonCat-13K", type=str, required=True)
    parser.add_argument("-only_g", "--only_graph", default=0, type=int, required=True)
    args = parser.parse_args()
    ds_path = '../datasets/' + args.dataset

    labelfile = ds_path + '/mlc2seq/train.txt'
    outfile = ds_path + '/label_n2v_embedding'
    trn_labels, _ = parse_mlc2seq_format(labelfile)






if __name__ == '__main__':
    main()
