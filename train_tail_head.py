import argparse
import os, sys, time
import pickle as pkl
import scipy.sparse as smat
from xbert.rf_util import smat_util
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from gensim.models import KeyedVectors as KV
from sklearn.neighbors import NearestNeighbors as NN












def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-ds", "--dataset", default="AmazonCat-13K", type=str, required=True)
    parser.add_argument("-t", "--tail_threshold", default=100, type=int)
    parser.add_argument("-gpu", "--device_num", default='1', type=str)
    parser.add_argument("-train", "--is_train", default=1, type=int)
    parser.add_argument("-ep", "--epochs", default=8, type=int)
    args = parser.parse_args()

    ds_path = '../datasets/' + args.dataset
    label_space = smat.load_npz(ds_path+'/L.elmo.npz')
    tail_threshold = args.tail_threshold
    device_num = args.device_num



if __name__ == '__main__':
    main()




#
