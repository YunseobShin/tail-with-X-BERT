from sklearn.cluster import spectral_clustering
import numpy as np
import argparse
import os, sys, time
import pickle as pkl
import scipy.sparse as smat
from xbert.rf_util import smat_util
from tqdm import tqdm, trange
from gensim.models import KeyedVectors as KV
from sklearn.neighbors import NearestNeighbors as NN

def parse_mlc2seq_format(data_path):
    assert(os.path.isfile(data_path))
    with open(data_path) as fin:
        labels, corpus = [], []
        for line in fin:
            tmp = line.strip().split('\t', 1)
            labels.append(tmp[0])
            corpus.append(tmp[1])
    return labels, corpus

class GraphUtil():
    def __init__(self, Y, num_labels):
        self.Y = Y
        self.nums = np.zeros(num_labels)
        self.adj = np.zeros([num_labels, num_labels])

    def gen_graph(self):
        for label_list in tqdm(self.Y):
            for i in range(len(label_list)):
                for j in range(i+1, len(label_list)):
                    self.adj[label_list[i]][label_list[j]] = 1

    def cal_degree(self):
        self.nums = np.sum(self.adj, axis=1)

parser = argparse.ArgumentParser(description='')
parser.add_argument("-ds", "--dataset", default="AmazonCat-13K", type=str, required=True)
parser.add_argument("-k", "--num_clusters", default=32, type=int, required=True)
args = parser.parse_args()
ds_path = '../datasets/' + args.dataset

labelfile = ds_path + '/mlc2seq/train.txt'
trn_labels, _ = parse_mlc2seq_format(labelfile)
trn_labels = [list(map(int, x.split(','))) for x in trn_labels]
label_space = smat.load_npz(ds_path+'/L.elmo.npz')
gutil = GraphUtil(trn_labels, label_space.shape[0])
gutil.gen_graph()
graph = gutil.adj

print('=====Start Cluerstering=====')
clustering = SpectralClustering(n_clusters = args.num_clusters,
    assign_labels="discretize", random_state=0).fit(graph)

clusters = clustering.labels_

np.save(ds_path+'/label_cluster-'+str(args.num_clusters), clusters)
