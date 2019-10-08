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
# from mather.bert import *
import xbert.data_utils as data_utils
import xbert.rf_linear as rf_linear
import xbert.rf_util as rf_util
from Hyperparameters import Hyperparameters

class BertModelforRegression_elmo(BertModel):
    def __init__(self, config, ft, output_dim=3072):
        super(BertModelforRegression_elmo, self).__init__(config)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.5)
        self.FCN1 = nn.Linear(768, 1024)
        self.FCN2 = nn.Linear(1024, output_dim)
        self.ft = ft
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if self.ft:
            with torch.no_grad():
                _, pooled_output = self.bert(input_ids, output_all_encoded_layers=False)
        else:
            _, pooled_output = self.bert(input_ids, output_all_encoded_layers=False)

        pooled_output = self.dropout(pooled_output)
        LF1 = torch.tanh(self.dropout(self.FCN1(pooled_output)))
        logits = self.FCN2(LF1)
        return logits

class BertEncodingsforRegression_elmo(BertModel):
    def __init__(self, config, ft, batch_size, max_seq_len=256, output_dim=3072):
        super(BertEncodingsforRegression_elmo, self).__init__(config)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.batch_size = batch_size
        self.dropout = nn.Dropout(0.5)
        self.FCN1 = nn.Linear(12*max_seq_len, 1)
        self.FCN2 = nn.Linear(768, output_dim)
        self.ft = ft
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if self.ft:
            encoded_layers, _ = self.bert(input_ids)
        else:
            with torch.no_grad():
                encoded_layers, _ = self.bert(input_ids)
        encodings = torch.cat(encoded_layers) # 12*bs, 22, 768
        new_shape = [self.batch_size, int(encodings.shape[1]*encodings.shape[0]/self.batch_size), encodings.shape[2]]
        encodings = torch.reshape(encodings, new_shape)
        encodings = torch.transpose(encodings, 2, 1) # bs, 768, 12*22
        LF1 = torch.tanh(self.FCN1(self.dropout(encodings))) # bs, 768, 1
        LF1 = torch.reshape(LF1, [LF1.shape[0], LF1.shape[1]])
        logits = self.FCN2(LF1)
        return logits

class BertRegressor():
    def __init__(self, hypes, emb_type, device_num, epochs, ft=False, max_seq_len=512):
        self.max_seq_len = max_seq_len
        self.ds_path = '../datasets/' + hypes.dataset
        self.hypes = hypes
        self.epochs = epochs
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.criterion = nn.MSELoss()
        if device_num in ['0', '1', '2', '3']:
            self.device = torch.device('cuda:'+device_num)
        elif device_num=='p':
            self.device = torch.device('cuda')
        self.device_num = device_num
        self.emb_type = emb_type
        if emb_type == 'n2v':
            self.model = BertModelforRegression_n2v.from_pretrained('bert-base-uncased', ft)
        elif emb_type == 'elmo':
            self.model = BertModelforRegression_elmo.from_pretrained('bert-base-uncased', ft)
        elif emb_type == 'en':
            self.model = BertEncodingsforRegression_elmo.from_pretrained('bert-base-uncased', ft, batch_size=self.hypes.train_batch_size, max_seq_len=max_seq_len, output_dim=3072)
    # def get_bert_emb(self, input_ids):
    #     _, pooled_output = self.bert(input_ids, output_all_encoded_layers=False)
    #     pooled_output = self.dropout(pooled_output)
    #     logits = self.model(pooled_output)
    #     return logits

    # def reg_loss(self, x, y):


    def train(self, X, Y, Y_nums, label_space, nbrs, model_path=None, ft_from=0):
        self.model.train()
        if self.device_num == 'p':
            self.model = torch.nn.DataParallel(self.model)
        else:
            self.model.to(self.device)

        if model_path: # fine tuning
            self.model.load_state_dict(torch.load(model_path))
            epohcs_range = range(ft_from+1, ft_from+self.epochs+1)
        else:
            epohcs_range = range(1, self.epochs+1)

        all_input_ids = torch.tensor(X)
        all_label_emb = torch.tensor(Y)
        all_Y_nums = Y_nums
        bs = self.hypes.train_batch_size
        # self.model = torch.nn.DataParallel(self.model)
        total_run_time = 0.0
        n_gpu = torch.cuda.device_count()

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
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            num_corrects = np.zeros(5)
            start_time = time.time()
            eval_t = 0
            for step in range(int(len(X)/bs)-1):
                input_ids = all_input_ids[step*bs:(step+1)*bs].to(self.device)
                label_embs = all_label_emb[step*bs:(step+1)*bs].to(self.device).float()
                lable_nums = np.array(all_Y_nums[step*bs:(step+1)*bs])

                c_pred = self.model(input_ids)
                logits = c_pred.cpu().detach().numpy()
                # loss = self.criterion(c_pred, label_embs)
                diff = np.abs(np.sum(logits -label_embs.cpu().detach().numpy()))
                if diff < 1:
                    loss = self.criterion(c_pred, label_embs)
                else:
                    loss = torch.abs(torch.sum(c_pred -label_embs)) - 0.5

                if self.device_num == 'p':
                    loss = loss.mean()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                total_run_time += time.time() - start_time

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 200 == 0:
                    eval_t += bs
                    distances, preds = nbrs.kneighbors(logits)
                    for i, pred in tqdm(enumerate(preds)):
                        truth = lable_nums[i]
                        if len(set(truth) & set(pred)):
                            for j in range(len(pred)):
                                if pred[j] in truth:
                                    num_corrects[j:] += 1


                    for i in range(1, len(num_corrects)+1):
                        num_corrects[i-1] /= i

                    print('Precisions:', np.round(num_corrects/eval_t, 4))
                    elapsed = time.time() - start_time
                    start_time = time.time()
                    cur_loss = tr_loss / nb_tr_steps
                    print("| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:5.4f} | train_loss {:e}".format(
                        epoch, step, int(len(X)/bs), elapsed * 1000, cur_loss))

            # output_dir = '../save_models/tail_regressor/'+self.hypes.dataset+'/t-'+str(self.t)+'_ep-' + str(epoch)+'-'+self.emb_type+'/'
            # self.save(output_dir)

    def evaluate(self, X, Y, Y_nums, label_space, nbrs, model_path = ''):
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        all_input_ids = torch.tensor(X)
        all_label_embs = torch.tensor(Y)
        all_Y_nums = torch.tensor(np.float64(Y_nums))
        bs=self.hypes.eval_batch_size
        self.model.eval()
        self.model.to(self.device)
        num_corrects = np.zeros(5)
        # for batch in eval_dataloader:
        print('Inferencing...')
        logits = np.zeros(all_label_embs.shape)
        for step in trange(int(len(X)/bs)-1):
            input_ids = all_input_ids[step*bs:(step+1)*bs].to(self.device)
            # label_embs = all_label_embs[step*bs:(step+1)*bs].to(self.device).float()
            # lable_nums = all_Y_nums[step*bs:(step+1)*bs]
            # start_time = time.time()

            with torch.no_grad():
                logit = self.model(input_ids)
            logits[step*bs:(step+1)*bs] = logit.cpu().detach().numpy()

        print(logits.shape)
        print('Calculating Precisions...')
        distances, preds = nbrs.kneighbors(logits)
        for i, pred in tqdm(enumerate(preds)):
            truth = int(all_Y_nums[i])
            if truth in pred:
                for j in range(len(pred)):
                    if pred[j] == truth:
                        num_corrects[j:] += 1
                        break

        Precisions = num_corrects/len(X)
        print('Test Precisions:', Precisions)
        return Precisions

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

    def get_label_elmo(self, labels_list, elmo):
        Y = []
        Y_nums = []
        print('getting elmo label embedding...')
        for labels in tqdm(labels_list):
            Y.append(np.mean(elmo[labels].toarray(), axis=0))
            Y_nums.append(labels)
        return Y,  Y_nums

    def save(self, model_path):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        output_model_file = os.path.join(model_path, WEIGHTS_NAME)
        output_config_file = os.path.join(model_path, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)


def load_data(X_raw, Y_list, X_path, Y_path, bertReg, embs, emb_type):
    if os.path.isfile(X_path):
        with open(X_path, 'rb') as g:
            X = pkl.load(g)
    else:
        X = bertReg.get_bert_token(X_raw)
        with open(X_path, 'wb') as g:
            pkl.dump(X, g)
            # np.savez(g, sp.csr_matrix(X))

    if os.path.isfile(Y_path):
        with open(Y_path, 'rb') as g:
            Y = pkl.load(g)
        with open(Y_path + 'nums', 'rb') as g:
            Y_nums = pkl.load(g)
    else:
        if emb_type == 'elmo' or emb_type == 'en':
            Y, Y_nums = bertReg.get_label_elmo(Y_list, embs)

        with open(Y_path, 'wb') as g:
            pkl.dump(Y, g)
        with open(Y_path + 'nums', 'wb') as g:
            pkl.dump(Y_nums, g)


    return X, Y, Y_nums

def load_raw(ds_path, is_train=1):
    if is_train:
        trn_labels, trn_corpus = parse_mlc2seq_format(ds_path + '/mlc2seq/train.txt')
    else:
        trn_labels, trn_corpus = parse_mlc2seq_format(ds_path + '/mlc2seq/test.txt')
    trn_X_raw_path = ds_path + '/raw_X'
    trn_Y_list_path = ds_path + '/Y-list'
    if os.path.isfile(trn_X_raw_path):
        trn_X_raw = pkl.load(open(trn_X_raw_path, 'rb'))
        trn_Y = pkl.load(open(trn_Y_list_path, 'rb'))
    else:
        trn_X_raw = []
        trn_Y = []
        for idx, labels in tqdm(enumerate(trn_labels)):
            labels = np.array(list(map(int, labels.split(','))))
            trn_X_raw.append(trn_corpus[idx])
            trn_Y.append(labels)
        pkl.dump(trn_X_raw, open(trn_X_raw_path, 'wb'))
        pkl.dump(trn_Y, open(trn_Y_list_path, 'wb'))
    return trn_X_raw, trn_Y

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
    parser.add_argument("-le", "--label_embs", default='elmo', type=str)
    parser.add_argument("-gpu", "--device_num", default='1', type=str)
    parser.add_argument("-train", "--is_train", default=1, type=int)
    parser.add_argument("-ep", "--epochs", default=8, type=int)
    parser.add_argument("-ft", "--fine_tune", default=0, type=int)
    parser.add_argument("-from", "--ft_from", default=0, type=int)
    args = parser.parse_args()

    hypes = Hyperparameters(args.dataset)
    ds_path = '../datasets/' + args.dataset
    device_num = args.device_num
    label_embs = args.label_embs
    fine_tune = args.fine_tune
    epochs = args.epochs
    is_train = args.is_train
    ft = (args.fine_tune == 1)
    ft_from = args.ft_from
    trn_X_raw, trn_Y = load_raw(ds_path, 1)
    test_X_raw, test_Y = load_raw(ds_path, 0)

    bertReg = BertRegressor(hypes, label_embs, device_num, epochs, ft, max_seq_len=256)
    trn_X_path = ds_path+'/trn_X'
    test_X_path = ds_path+'/test_X'
    if label_embs == 'elmo' or label_embs == 'en':
        label_space = smat.load_npz(ds_path+'/L.elmo.npz')
        trn_Y_path = ds_path+'/trn_elmo'
        test_Y_path = ds_path+'/test_elmo_Y'
        trn_X, trn_Y, trn_Y_nums = load_data(trn_X_raw, trn_Y, trn_X_path, trn_Y_path, bertReg, label_space, 'elmo')
        test_X, test_Y, test_Y_nums = load_data(test_X_raw, test_Y, test_X_path, test_Y_path, bertReg, label_space, 'elmo')

    else:
        print('invalid label embedding type')
        exit()
    # print('Number of instnaces:', len(trn_X))
    # print('Number of labels:', len(trn_Y_nums))

    knn_path = ds_path + 'nbrs'
    if os.path.isfile(knn_path):
        with open(knn_path, 'rb') as f:
            nbrs = pkl.load(f)
    else:
        with open(knn_path, 'wb') as f:
            nbrs = NN(n_neighbors=5, algorithm='auto').fit(label_space)
            pkl.dump(nbrs, f)

    if is_train:
        if ft:
            print('======================Start Fine-tuning======================')
            model_path = '../save_models/regressor/'+hypes.dataset+'/ep-' + str(ft_from)+'-elmo/pytorch_model.bin'
            bertReg.train(trn_X, trn_Y, trn_Y_nums, label_space, nbrs, model_path, ft_from)
        else:
            print('======================Start Training======================')
            bertReg.train(trn_X, trn_Y, trn_Y_nums, label_space, nbrs)
        output_dir = '../save_models/regressor/'+hypes.dataset+'/ep-' + str(epochs + ft_from) +'-'+ label_embs+'/'
        bertReg.save(output_dir)
    else:
        model_path = '../save_models/regressor/'+hypes.dataset+'/ep-' + str(ft_from)+'/pytorch_model.bin'
        print('======================Start Testing======================')
        accs =  bertReg.evaluate(test_X, test_Y, test_Y_nums, label_space, nbrs, model_path)



if __name__ == '__main__':
    main()

#
