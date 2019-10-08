import argparse
from urllib import parse
import re, os
import scipy.sparse as smat
import numpy as np
from rf_util import *
from tqdm import tqdm, trange
import pickle as pkl
import scipy as sp
from xbert.rf_util import smat_util

parser = argparse.ArgumentParser(description='')
parser.add_argument("-ds", "--dataset", default="AmazonCat-13K", type=str, required=True)
parser.add_argument("-t", "--tail_threshold", default=10, type=int)
args = parser.parse_args()
dataset_list = ['AmazonCat-13K', 'Eurlex-4K', 'Wiki-500K', 'Wiki10-31K']
if args.dataset not in dataset_list:
    print('invalid dataset')
    exit()

t = args.tail_threshold
ds_path = '../datasets/' + args.dataset

with open(ds_path + '/mlc2seq/label_vocab.txt', 'r') as fin:
    label_list = [line.strip().split('\t') for line in fin]

label_list = np.array(label_list)
for i, label in tqdm(enumerate(label_list)):
    if int(label[0]) <= t:
        tails = range(i, len(label_list))
        break;

with open(ds_path+'/mlc2seq/tails-'+str(t), 'wb') as g:
    pkl.dump(tails, g)
