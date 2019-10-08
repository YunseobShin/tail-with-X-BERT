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

def parse_mlc2seq_format(data_path):
    assert(os.path.isfile(data_path))
    with open(data_path) as fin:
        labels, corpus = [], []
        for line in fin:
            tmp = line.strip().split('\t', 1)
            labels.append(tmp[0])
            corpus.append(tmp[1])
    return labels, corpus

def sorted_csr(csr, only_topk=None):
    assert isinstance(csr, smat.csr_matrix)
    row_idx = sp.repeat(sp.arange(csr.shape[0], dtype=sp.uint32), csr.indptr[1:] - csr.indptr[:-1])
    return smat_util.sorted_csr_from_coo(csr.shape, row_idx, csr.indices, csr.data, only_topk)

def convert_label_to_Y(labels, K_in=None):
    rows, cols ,vals = [], [], []
    for i, label in enumerate(labels):
        label_list = list(map(int, label.split(',')))
        rows += [i] * len(label_list)
        cols += label_list
        vals += [1.0] * len(label_list)

    K_out = max(cols) + 1 if K_in is None else K_in
    Y = smat.csr_matrix( (vals, (rows,cols)), shape=(len(labels),K_out) )
    return Y

def get_tails(x, y, t):
    tails = np.where(y[:,1]<=t)[0]
    return tails

def get_in_tails(v, tails):
    return np.array([x for x in v if x in tails])

def evaluate(ans, preds, topk = 1):
    total_matched = sp.zeros(topk, dtype=sp.uint64)
    recall = sp.zeros(topk, dtype=sp.float64)
    for i in range(ans.shape[0]):
        truth = ans.indices[ans.indptr[i]:ans.indptr[i+1]]
        matched = sp.isin(preds.indices[preds.indptr[i]:preds.indptr[i + 1]][:topk], truth)
        cum_matched = sp.cumsum(matched, dtype=sp.uint64)
        total_matched[:len(cum_matched)] += cum_matched
        recall[:len(cum_matched)] += cum_matched / len(truth)
        if len(cum_matched) != 0:
            total_matched[len(cum_matched):] += cum_matched[-1]
            recall[len(cum_matched):] += cum_matched[-1] / len(truth)
    prec = total_matched / ans.shape[0] / sp.arange(1, topk + 1)
    recall = recall / ans.shape[0]
    return np.round(prec, 4), np.round(recall, 4)

def evaluate_tails(ans, preds, tails, topk = 1):
    total_matched = sp.zeros(topk, dtype=sp.uint64)
    t_total_matched = sp.zeros(topk, dtype=sp.uint64)
    r_total_matched = sp.zeros(topk, dtype=sp.uint64)
    recall = sp.zeros(topk, dtype=sp.float64)
    t_recall = sp.zeros(topk, dtype=sp.float64)
    r_recall = sp.zeros(topk, dtype=sp.float64)
    q = 0
    p = 0
    r = 0
    for i in trange(ans.shape[0]):
        truth = ans.indices[ans.indptr[i]:ans.indptr[i+1]]
        tail_truth = get_in_tails(truth, tails)
        if not len(tail_truth):
            p += 1
            t_preds = preds.indices[preds.indptr[i]:preds.indptr[i + 1]][:topk]
            matched = sp.isin(t_preds, truth)
            cum_matched = sp.cumsum(matched, dtype=sp.uint64)
            total_matched[:len(cum_matched)] += cum_matched
            recall[:len(cum_matched)] += cum_matched / len(truth)
            if len(cum_matched) != 0:
                total_matched[len(cum_matched):] += cum_matched[-1]
                recall[len(cum_matched):] += cum_matched[-1] / len(truth)
                continue
        q += 1
        t_preds = preds.indices[preds.indptr[i]:preds.indptr[i + 1]][:topk]
        t_matched = sp.isin(t_preds, tail_truth)
        r_matched = sp.isin(t_preds, truth)
        t_cum_matched = sp.cumsum(t_matched, dtype=sp.uint64)
        r_cum_matched = sp.cumsum(r_matched, dtype=sp.uint64)
        t_total_matched[:len(t_cum_matched)] += t_cum_matched
        r_total_matched[:len(r_cum_matched)] += r_cum_matched
        t_recall[:len(t_cum_matched)] += t_cum_matched / len(tail_truth)
        r_recall[:len(r_cum_matched)] += r_cum_matched / len(truth)
        if len(t_cum_matched) != 0:
            t_total_matched[len(t_cum_matched):] += t_cum_matched[-1]
            t_recall[len(t_cum_matched):] += t_cum_matched[-1] / len(tail_truth)

        if len(r_cum_matched) != 0:
            r_total_matched[len(r_cum_matched):] += r_cum_matched[-1]
            r_recall[len(r_cum_matched):] += r_cum_matched[-1] / len(truth)

    t_prec = t_total_matched / q / sp.arange(1, topk + 1)
    t_recall = t_recall / q
    r_prec = r_total_matched / q / sp.arange(1, topk + 1)
    r_recall = r_recall / q
    prec = total_matched / p / sp.arange(1, topk + 1)
    recall = recall / p
    print('preds in tails:', q)
    print('preds in non-tails:', p)
    return np.round(t_prec, 4), np.round(t_recall, 4), np.round(prec, 4), np.round(recall, 4), np.round(r_prec, 4), np.round(r_recall, 4)

def get_one(x):
    return x[1]

def get_num_labels(trn_labels, tails):
    dt = []
    st = set(tails)
    tail_instances = []
    for i, tl in tqdm(enumerate(trn_labels)):
        t = list(map(int, tl.split(',')))
        d=(i, t, len(t))
        num_tails = len(set(d[1]) & st)
        if num_tails:
            tail_instances.append((d[2], num_tails))

    return np.array(tail_instances)

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-ds", "--dataset", default="AmazonCat-13K", type=str, required=True)
    parser.add_argument("-t", "--tail_threshold", default=10, type=int)

    args = parser.parse_args()
    dataset_list = ['AmazonCat-13K', 'Eurlex-4K', 'Wiki-500K', 'Wiki10-31K']
    if args.dataset not in dataset_list:
        print('invalid dataset')
        exit()
    x = y = []
    ds_path = '../datasets/' + args.dataset
    num_label_file = ds_path + '/num_label.pkl'
    num_ins_file = ds_path + '/num_ins.pkl'
    trn_labels, trn_corpus = parse_mlc2seq_format(ds_path+'/mlc2seq/train.txt')
    Y_ret_trn = convert_label_to_Y(trn_labels, K_in=None)

    if os.path.isfile(num_label_file):
        with open(num_label_file, 'rb') as f:
            x = pkl.load(f)
    else:
        num_labels = []
        for i in tqdm(range(0, Y_ret_trn.shape[0])):
            v = Y_ret_trn[i].todense()
            num_labels.append( (i, len(np.where(v)[1])) )
        x = sorted(num_labels, reverse=True, key=get_one)
        with open(num_label_file, 'wb') as f:
            pkl.dump(x, f)

    if os.path.isfile(num_ins_file):
        with open(num_ins_file, 'rb') as f:
            y = pkl.load(f)
    else:
        num_instances = []
        YT = Y_ret_trn.transpose()
        for i in tqdm(range(0, YT.shape[0])):
            v = YT[i].todense()
            num_instances.append((i, len(np.where(v)[1])))
            # pos = YT.indices[YT.indptr[i]:YT.indptr[i+1]]
            # num_instances.append((i, len(pos)))
        y = np.array(sorted(num_instances, reverse=True, key=get_one))
        with open(num_ins_file, 'wb') as f:
            pkl.dump(y, f)

    tail_threshold = args.tail_threshold
    tailsfile = ds_path + '/mlc2seq/tails-'+ str(tail_threshold)
    if os.path.isfile(tailsfile):
        with open(tailsfile, 'rb') as f:
            tails = pkl.load(f)
            tails = np.array(tails)
    else:
        tails = get_tails(x, y, tail_threshold)
        with open(tailsfile, 'wb') as f:
            pkl.dump(tails, f)
    num_labels = get_num_labels(trn_labels, tails)
    avg_label_per_tail = round(np.mean(num_labels[:,0]), 4)
    avg_label = round(np.mean(np.array(x)[:,1]), 4)
    avg_ins_per_tail = round(len(num_labels)/tails.shape[0], 4)
    avg_ins_per_label = round(len(x)/len(y), 4)
    print('Number of tails:', tails.shape[0])
    print('Number of tail instances:', len(num_labels))
    print('Average labels per tails instance:', avg_label_per_tail)
    print('Average labels per instance:', avg_label)
    # print('Average instances per tails label:',avg_ins_per_tail)
    # print('Average instances per label:',avg_ins_per_label)
    exit()
    preds = smat.load_npz('../pretrained_models/'+args.dataset+'/pifa-a5-s2/ranker/tst.pred.xbert.npz')
    preds = sorted_csr(preds, 10)
    ans = smat.load_npz(ds_path+'/Y.tst.npz')
    topk=10
    t_precs, t_recalls, precs, recalls, r_precs, r_recalls= evaluate_tails(ans, preds, tails, topk)
    print('T-Precision:', t_precs[0], t_precs[2], t_precs[4], t_precs[9])
    print('T-Recall:', t_recalls[0], t_recalls[2], t_recalls[4], t_recalls[9])
    print('Precision:', precs[0], precs[2], precs[4], precs[9])
    print('Recall:', recalls[0], recalls[2], recalls[4], recalls[9])
    print('R-Precision:', r_precs[0], r_precs[2], r_precs[4], r_precs[9])
    print('R-Recall:', r_recalls[0], r_recalls[2], r_recalls[4], r_recalls[9])


    # precs, recalls = evaluate(ans, preds, topk)
    # print('Precision:', precs)
    # print('Recall:', recalls)


if __name__ == '__main__':
    main()











    #
