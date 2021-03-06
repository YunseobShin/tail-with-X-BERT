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
from torch.nn import Parameter
from pytorch_pretrained_bert import BertTokenizer, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from gensim.models import KeyedVectors as KV
from sklearn.neighbors import NearestNeighbors as NN
import random
# from mather.bert import *
import xbert.data_utils as data_utils
import xbert.rf_linear as rf_linear
import xbert.rf_util as rf_util
from Hyperparameters import Hyperparameters
from GCN import gen_A, gen_adj
from GAT_layers import GraphAttentionLayer, SpGraphAttentionLayer

class GraphUtil():
    def __init__(self, Y, num_labels):
        self.Y = Y
        self.nums = np.zeros(num_labels)
        self.adj = np.zeros([num_labels, num_labels])

    def gen_graph(self):
        for label_list in tqdm(self.Y):
            for i in range(len(label_list)):
                for j in range(i+1, len(label_list)):
                    self.adj[label_list[i]][label_list[j]] += 1

    def cal_degree(self):
        self.nums = np.sum(self.adj, axis=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nclass, nhid=1024, dropout=0.5, alpha=0.2, nheads=8):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        # self.attention = SpGraphAttentionLayer(nfeat,
        #                                          nhid,
        #                                          dropout=dropout,
        #                                          alpha=alpha,
        #                                          concat=False)
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nfeat,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.attention(x, adj, dv)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

class BertGAT(BertModel):
    def __init__(self, config, ft, num_labels, res, H, device_num):
        super(BertGAT, self).__init__(config)
        self.device = torch.device('cuda:' + device_num)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.5)
        self.ft = ft
        self.num_labels = num_labels
        self.gat = SpGAT(nfeat=H.shape[1], nclass=num_labels).to(self.device)
        self.FCN = nn.Linear(768, num_labels)
        self.softmax = nn.Softmax()
        self.H = torch.tensor(H).float().to(self.device)
        A = torch.tensor(gen_A(num_labels, res)).float().to(self.device)
        self.adj = gen_adj(A).detach()
        # self.FCN2 = nn.Linear(num_labels, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if self.ft:
            with torch.no_grad():
                _, pooled_output = self.bert(input_ids, output_all_encoded_layers=False)
        else:
            _, pooled_output = self.bert(input_ids, output_all_encoded_layers=False)

        bert_logits = self.dropout(pooled_output)
        bert_logits = self.FCN(bert_logits) # 768, num_labels

        x = self.gat(self.H, self.adj)
        x = x.transpose(1, 0)
        x = torch.matmul(bert_logits, x)
        # logits = self.FCN2(x)
        # return self.softmax(logits)
        return logits


def get_binary_vec(label_list, output_dim):
    res = np.zeros([len(label_list), output_dim])
    for i, label in tqdm(enumerate(label_list)):
        res[i][label]=1
    return res

def get_score(logits, truth, k=5):
    num_corrects = np.zeros(k)
    preds = np.argsort(-logits.cpu().detach().numpy())[:k]
    # print(logits)
    # print(preds.shape)
    truth = np.where(truth)[0]

    precision=np.zeros(k)
    recall=np.zeros(k)
    for i, pred in enumerate(preds):
        if pred in truth:
            num_corrects[i:] += 1

    for i in range(k):
        precision[i] = num_corrects[i]/(i+1)
        recall[i] = num_corrects[i]/len(truth)
    return precision, recall


class BertGATClassifier():
    def __init__(self, hypes, heads, t, device_num, ft, epochs, gutil, label_space, max_seq_len=512):
        self.max_seq_len = max_seq_len
        self.ds_path = '../datasets/' + hypes.dataset
        self.hypes = hypes
        self.epochs = epochs
        self.t = t
        self.H = label_space.todense()
        self.heads = heads
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.criterion =  nn.MultiLabelSoftMarginLoss()
        self.device = torch.device('cuda:' + device_num)
        self.model = BertGAT.from_pretrained('bert-base-uncased', ft, len(heads), gutil, self.H, device_num)

    def train(self, X, Y, val_X, val_Y, model_path=None, ft_from=0):
        if model_path: # fine tuning
            self.model.load_state_dict(torch.load(model_path))
            epohcs_range = range(ft_from+1, ft_from+self.epochs+1)
        else:
            epohcs_range = range(1, self.epochs+1)

        all_input_ids = torch.tensor(X)
        all_Ys = get_binary_vec(Y, len(self.heads))
        all_Ys = torch.tensor(all_Ys)
        bs = 1
        self.model.train()
        self.model.to(self.device)
        total_run_time = 0.0

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.hypes.learning_rate,
                             warmup=self.hypes.warmup_rate)

        for epoch in tqdm(epohcs_range):
            eval_t = 0
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            num_corrects = 0
            total = 0
            start_time = time.time()
            precisions=np.zeros(5)
            recalls=np.zeros(5)
            for step in range(int(len(X)/bs)-1):
                # if random.randint(1, 5) != 2:
                #     continue
                input_ids = all_input_ids[step*bs:(step+1)*bs].to(self.device)
                labels = all_Ys[step*bs:(step+1)*bs].to(self.device).float()
                c_pred = self.model(input_ids)
                loss = self.criterion(c_pred, labels)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % self.hypes.log_interval == 0:
                    for i in range(len(c_pred)):
                        eval_t += 1
                        truth = labels.cpu().detach().numpy().astype(int)[i]
                        logit = c_pred[i]
                        precision, recall = get_score(logit, truth)
                        precisions += precision
                        recalls += recall
                    elapsed = time.time() - start_time
                    start_time = time.time()
                    cur_loss = tr_loss / nb_tr_steps
                    print("| {:4d}/{:4d} batches | ms/batch {:5.2f}".format(step, int(len(X)/bs), elapsed * 1000))
                    print('Precision:', np.round(precisions/eval_t, 4))
                    print('Recall:', np.round(recalls/eval_t, 4))

            # if epoch % 10 == 0:
            output_dir = '../save_models/head_classifier/'+self.hypes.dataset+'/t-'+str(self.t)+'_ep-'+str(epoch)+'/'
            self.save(output_dir)

            eval_idx = np.random.choice(range(0, len(val_X)), int(len(val_X)/10))
            val_inputs = np.array(val_X)[eval_idx]
            val_labels = np.array(val_Y)[eval_idx]
            acc = self.evaluate(val_inputs, val_labels)
            self.model.train()


    def evaluate(self, X, Y, model_path=''):
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        all_input_ids = torch.tensor(X)
        all_Ys = get_binary_vec(Y, len(self.heads))
        bs = self.hypes.eval_batch_size
        self.model.eval()
        self.model.to(self.device)

        num_corrects = 0
        start_time = time.time()
        precisions = np.zeros(5)
        recalls = np.zeros(5)
        eval_t=0
        print('Inferencing...')
        for step in trange(int(len(X)/bs)-1):
            input_ids = all_input_ids[step*bs:(step+1)*bs].to(self.device)
            labels = all_Ys[step*bs:(step+1)*bs]

            with torch.no_grad():
                c_pred = self.model(input_ids)

            for i in range(len(c_pred)):
                eval_t += 1
                truth = labels[i]
                logit = c_pred[i]
                precision, recall = get_score(logit, truth)
                precisions += precision
                recalls += recall

        print('Test Precision:', np.round(precisions/eval_t, 4))
        print('Test Recall:', np.round(recalls/eval_t, 4))

    def get_bert_token(self, trn_text, only_CLS=False):
        X = []
        # self.model.cuda(1)
        print('========================================================')
        print('getting sentence embedding...')
        for text in tqdm(trn_text):
            marked_text = "[CLS] " + text + " [SEP]"
            tokenized_text = self.tokenizer.tokenize(marked_text)
            if len(tokenized_text) > self.max_seq_len:
                tokenized_text = tokenized_text[:self.max_seq_len-1] + ['[SEP]']

            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            if len(indexed_tokens) < self.max_seq_len:
                padding = [0] * (self.max_seq_len - len(indexed_tokens))
                indexed_tokens += padding
            else:
                indexed_tokens = indexed_tokens[:self.max_seq_len]

            X.append(indexed_tokens)
        print('========================================================')
        return X

    def save(self, output_dir):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

def load_data(X_path, head_X, bert):
    if os.path.isfile(X_path):
        with open(X_path, 'rb') as g:
            X = pkl.load(g)
    else:
        X = bert.get_bert_token(head_X)
        with open(X_path, 'wb') as g:
            pkl.dump(X, g)

    return X

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-ds", "--dataset", default="AmazonCat-13K", type=str, required=True)
    parser.add_argument("-t", "--head_threshold", default=10000, type=int)
    parser.add_argument("-gpu", "--device_num", default='0', type=str)
    parser.add_argument("-train", "--is_train", default=1, type=int)
    parser.add_argument("-ep", "--epochs", default=8, type=int)
    parser.add_argument("-ft", "--fine_tune", default=0, type=int)
    parser.add_argument("-from", "--ft_from", default=0, type=int)
    args = parser.parse_args()

    ft = (args.fine_tune == 1)
    ft_from = args.ft_from
    hypes = Hyperparameters(args.dataset)
    ds_path = '../datasets/' + args.dataset
    device_num = args.device_num
    head_threshold = args.head_threshold
    with open(ds_path+'/mlc2seq/train_heads_X-'+str(head_threshold), 'rb') as g:
        trn_head_X = pkl.load(g)
    with open(ds_path+'/mlc2seq/train_heads_Y-'+str(head_threshold), 'rb') as g:
        trn_head_Y = pkl.load(g)
    with open(ds_path+'/mlc2seq/test_heads_X-'+str(head_threshold), 'rb') as g:
        test_head_X = pkl.load(g)
    with open(ds_path+'/mlc2seq/test_heads_Y-'+str(head_threshold), 'rb') as g:
        test_head_Y = pkl.load(g)
    answer = smat.load_npz(ds_path+'/Y.tst.npz')
    with open(ds_path+'/mlc2seq/heads-'+str(head_threshold), 'rb') as g:
        heads = pkl.load(g)
    label_space = smat.load_npz(ds_path+'/L.elmo.npz')
    label_space = smat.lil_matrix(label_space)
    label_space = label_space[:len(heads)]
    output_dim = len(heads)
    gutil = GraphUtil(trn_head_Y, output_dim)
    gutil.gen_graph()
    gutil.cal_degree()
    bert = BertGATClassifier(hypes, heads, head_threshold, device_num, ft, args.epochs, gutil, label_space, max_seq_len=256)
    trn_X_path = ds_path+'/head_data/trn_X-' + str(head_threshold)
    test_X_path = ds_path+'/head_data/test_X-' + str(head_threshold)
    trn_X = load_data(trn_X_path, trn_head_X, bert)
    test_X = load_data(test_X_path, test_head_X, bert)
    print('Number of labels:', len(heads))
    print('Number of trn instances:', len(trn_X))
    if args.is_train:
        if ft:
            print('======================Start Fine-Tuning======================')
            model_path = '../save_models/GAT_classifier/'+hypes.dataset+'/t-'+str(head_threshold)+'_ep-' + str(ft_from)+'/pytorch_model.bin'
            bert.train(trn_X, trn_head_Y, test_X, test_head_Y, model_path, ft_from)
        else:
            print('======================Start Training======================')
            bert.train(trn_X, trn_head_Y, test_X, test_head_Y)
            # bert.save()
    else:
        model_path = '../save_models/GAT_classifier/'+args.dataset+'/t-'+str(head_threshold)+'_ep-'+str(ft_from)+'/pytorch_model.bin'
        print('======================Start Testing======================')
        bert.evaluate(test_X, test_head_Y, model_path)



if __name__ == '__main__':
    main()

#
