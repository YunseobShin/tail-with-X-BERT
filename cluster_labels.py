from sklearn.cluster import MiniBatchKMeans
import numpy as np
import argparse
import os, sys, time
import pickle as pkl
import scipy.sparse as smat
from xbert.rf_util import smat_util
from tqdm import tqdm, trange
from gensim.models import KeyedVectors as KV
from sklearn.neighbors import NearestNeighbors as NN

parser = argparse.ArgumentParser(description='')
parser.add_argument("-ds", "--dataset", default="AmazonCat-13K", type=str, required=True)
parser.add_argument("-k", "--num_clusters", default=32, type=int, required=True)
args = parser.parse_args()
ds_path = '../datasets/' + args.dataset
label_space = smat.load_npz(ds_path+'/L.elmo.npz')
print('=====Start Cluerstering=====')
clustering = MiniBatchKMeans(n_clusters = args.num_clusters, random_state=0,
        batch_size=8, max_iter=10).fit(label_space)

clusters = clustering.labels_

np.save(ds_path+'/label_cluster-'+str(args.num_clusters), clusters)
