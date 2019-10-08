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

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-ds", "--dataset", default="AmazonCat-13K", type=str, required=True)
    parser.add_argument("-t", "--threshold", default=100, type=int, required=True)
    parser.add_argument("-train", "--is_train", default=1, type=int, required=True)
    args = parser.parse_args()
    ds_path = '../datasets/' + args.dataset
    threshold = args.threshold
    tail_instances = []
    tail_Y = []
    with open(ds_path+'/mlc2seq/tails-'+str(threshold), 'rb') as g:
        tails = pkl.load(g)
        tails = list(tails)
    if args.is_train:
        trn_labels, trn_corpus = parse_mlc2seq_format(ds_path + '/mlc2seq/train.txt')
        for idx, labels in tqdm(enumerate(trn_labels)):
            labels = np.array(list(map(int, labels.split(','))))
            for label in list(set(labels) & set(tails)):
                tail_instances.append(trn_corpus[idx])
                tail_Y.append(label)
                break
        with open(ds_path+'/mlc2seq/train_tails_X-'+str(threshold), 'wb') as g:
            pkl.dump(tail_instances, g)
        with open(ds_path+'/mlc2seq/train_tails_Y-'+str(threshold), 'wb') as g:
            pkl.dump(tail_Y, g)
    else:
        test_labels, test_corpus = parse_mlc2seq_format(ds_path + '/mlc2seq/test.txt')
        for idx, labels in tqdm(enumerate(test_labels)):
            labels = np.array(list(map(int, labels.split(','))))
            for label in list(set(labels) & set(tails)):
                tail_instances.append(test_corpus[idx])
                tail_Y.append(label)
                break
        with open(ds_path+'/mlc2seq/test_tails_X-'+str(threshold), 'wb') as g:
            pkl.dump(tail_instances, g)
        with open(ds_path+'/mlc2seq/test_tails_Y-'+str(threshold), 'wb') as g:
            pkl.dump(tail_Y, g)

if __name__ == '__main__':
    main()
