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
parser.add_argument("-t", "--head_threshold", default=10000, type=int)
args = parser.parse_args()
dataset_list = ['AmazonCat-13K', 'Eurlex-4K', 'Wiki-500K', 'Wiki10-31K']
if args.dataset not in dataset_list:
    print('invalid dataset')
    exit()

h = args.head_threshold
ds_path = '../datasets/' + args.dataset

with open(ds_path + '/mlc2seq/label_vocab.txt', 'r') as fin:
    label_list = [line.strip().split('\t') for line in fin]

q=1
label_list = np.array(label_list)
for i, label in tqdm(enumerate(label_list)):
    if int(label[0]) <= h:
        heads = range(0, i)
        q=0
        break;
if q==1:
    heads = range(0, len(label_list))

print(list(heads))
with open(ds_path+'/mlc2seq/heads-'+str(h), 'wb') as g:
    pkl.dump(heads, g)
