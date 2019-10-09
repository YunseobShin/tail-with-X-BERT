import argparse
import os, sys, time
import pickle as pkl
import scipy.sparse as smat, networkx as nx
from xbert.rf_util import smat_util
from collections import Counter
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
from tail_eval import evaluate_tails, get_in_tails
from Hyperparameters import Hyperparameters


class BertModelforRegression_n2v(BertModel):
    def __init__(self, config, ft, output_dim=512):
        super(BertModelforRegression_n2v, self).__init__(config)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.FCN1 = nn.Linear(768, 512)
        self.FCN2 = nn.Linear(512, output_dim)
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
    def __init__(self, hypes, tails, emb_type, t, device_num, epochs, ft=False, max_seq_len=512):
        self.max_seq_len = max_seq_len
        self.ds_path = '../datasets/' + hypes.dataset
        self.hypes = hypes
        self.epochs = epochs
        self.tails = tails
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.criterion = nn.MSELoss()
        self.t = t
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
    def predict_heads_by_graph(self, X, Y_nums, label_graph, heads, tail_space, head_space, model_path=None):
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model.to(self.device)
        tail_nbrs = NN(n_neighbors=5, algorithm='auto').fit(tail_space)
        all_input_ids = torch.tensor(X)
        logits = np.zeros([len(X), tail_space.shape[1]])
        bs=self.hypes.eval_batch_size
        for step in trange(int(len(X)/bs)-1):
            input_ids = all_input_ids[step*bs:(step+1)*bs].to(self.device)
            with torch.no_grad():
                logit = self.model(input_ids)
            logits[step*bs:(step+1)*bs] = logit.cpu().detach().numpy()

        # print('Calculating Acccuracy...')
        preds_path = self.ds_path + '/knn_preds'
        if not os.path.exists(preds_path):
            _, preds = tail_nbrs.kneighbors(logits)
            with open(preds_path, 'wb') as f:
                pkl.dump(preds, f)
        else:
            with open(preds_path, 'rb') as f:
                preds = pkl.load(f)

        head_subs = {}
        num_corrects = np.zeros(5)
        for i, pred_tails in tqdm(enumerate(preds)):
            truth = Y_nums[i]
            for pred_tail in pred_tails:
                head_neis = [int(x) for x in list(label_graph.neighbors(str(pred_tail))) if int(x) in heads]
                if i in head_subs:
                    head_subs[i] += head_neis
                else:
                    head_subs[i] = head_neis

            if i in head_subs:
                voted_list = np.array(sorted(Counter(head_subs[i]).items(), reverse=True, key=lambda kv: kv[1]))[:,0]
                head_preds = voted_list[:5]
                # head_embs = head_space[head_subs[i]]
                # head_nbrs = NN(n_neighbors=5, algorithm='auto').fit(head_embs)
                # _, head_preds = head_nbrs.kneighbors([logits[i]])
                # head_preds = head_preds[0]
                if len(set(truth) & set(head_preds)):
                    for j in range(len(head_preds)):
                        if head_preds[j] in truth:
                            num_corrects[j:] += 1
            else:
                print(i, 'does not have heads')

        for i in range(1, 6):
            num_corrects[i-1] /= i

        head_precision = np.round(num_corrects / len(X), 4)
        print('Head Precisions:', head_precision)
        return head_precision

    def train(self, X, Y, Y_nums, label_space, val_X, val_Y, val_Y_nums, model_path=None, ft_from=0):
        self.model.train()
        if self.device_num == 'p':
            self.model = torch.nn.DataParallel(self.model)
        else:
            self.model.to(self.device)
        print('building KNN...')
        nbrs = NN(n_neighbors=5, algorithm='auto').fit(label_space)
        if model_path: # fine tuning
            self.model.load_state_dict(torch.load(model_path))
            epohcs_range = range(ft_from+1, ft_from+self.epochs+1)
        else:
            epohcs_range = range(1, self.epochs+1)

        bs = self.hypes.train_batch_size
        all_input_ids = torch.tensor(X)
        all_label_emb = torch.tensor(Y)
        all_Y_nums = torch.tensor(np.float64(Y_nums))
        train_data = TensorDataset(all_input_ids, all_label_emb, all_Y_nums)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)
        val_X = np.array(val_X)
        val_Y = np.array(val_Y)
        val_Y_nums = np.array(val_Y_nums)
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
            for step, batch in enumerate(train_dataloader):
                input_ids, label_embs, label_nums = batch
                input_ids = input_ids.to(self.device)
                label_embs = label_embs.to(self.device).float()
            # for step in range(int(len(X)/bs)-1):
            #     input_ids = all_input_ids[step*bs:(step+1)*bs].to(self.device)
            #     label_embs = all_label_emb[step*bs:(step+1)*bs].to(self.device).float()
            #     label_nums = all_Y_nums[step*bs:(step+1)*bs]

                c_pred = self.model(input_ids)
                loss = self.criterion(c_pred, label_embs)

                if self.device_num == 'p':
                    loss = loss.mean()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                total_run_time += time.time() - start_time

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % self.hypes.log_interval == 0:
                    logits = c_pred.cpu().detach().numpy()
                    eval_t += bs
                    distances, preds = nbrs.kneighbors(logits)
                    for i, pred in tqdm(enumerate(preds)):
                        truth = int(label_nums[i])
                        if truth in pred:
                            for j in range(len(pred)):
                                if pred[j] == truth:
                                    num_corrects[j:] += 1
                                    break

                    print('Acccuracy:', np.round(num_corrects/eval_t, 4))
                    elapsed = time.time() - start_time
                    start_time = time.time()
                    cur_loss = tr_loss / nb_tr_steps
                    print("| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:5.4f} | train_loss {:e}".format(
                        epoch, step, len(train_dataloader), elapsed * 1000, cur_loss))


            if epoch % 10 == 0:
                eval_idx = np.random.choice(range(0, len(val_X)), int(len(val_X)/10))
                val_inputs = val_X[eval_idx]
                val_labels = val_Y[eval_idx]
                val_nums = val_Y_nums[eval_idx]
                acc = self.evaluate(val_inputs, val_labels, val_nums, label_space)
                self.model.train()

            # output_dir = '../save_models/tail_regressor/'+self.hypes.dataset+'/t-'+str(self.t)+'_ep-' + str(epoch)+'-'+self.emb_type+'/'
            # self.save(output_dir)

    def evaluate(self, X, Y, Y_nums, label_space, model_path = '', ft_from=0):
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        nbrs = NN(n_neighbors=5, algorithm='auto').fit(label_space)
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
        # print('Calculating Acccuracy...')
        preds_path = self.ds_path + '/knn_preds-ep-' + str(ft_from)
        if not os.path.exists(preds_path):
            _, preds = nbrs.kneighbors(logits)
            with open(preds_path, 'wb') as f:
                pkl.dump(preds, f)
        else:
            with open(preds_path, 'rb') as f:
                preds = pkl.load(f)

        for i, pred in tqdm(enumerate(preds)):
            if i >= len(all_Y_nums):
                break
            truth = int(all_Y_nums[i])
            if truth in pred:
                for j in range(len(pred)):
                    if pred[j] == truth:
                        num_corrects[j:] += 1
                        break

        Acccuracy = num_corrects/len(X)
        print('Test Acccuracy:', Acccuracy)
        return Acccuracy

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

    def get_label_n2v(self, tail_labels, lv):
        Y = []
        Y_nums = []
        print('getting n2v label embedding...')
        # lv = KV.load_word2vec_format(self.ds_path+'/label_n2v_embedding', binary=False)
        OOK = 0
        for label in tqdm(tail_labels):
            Y_nums.append(label)
            if str(label) in lv:
                Y.append(lv[str(label)])
            else:
                OOK += 1
                print(label)
                Y.append(np.zeros(lv.vector_size))
        print(OOK, 'instances have no n2v label embedding')
        return Y, Y_nums

    def get_label_elmo(self, tail_labels, elmo):
        Y = []
        Y_nums = []
        print('getting elmo label embedding...')
        for label in tqdm(tail_labels):
            Y.append(elmo[label].toarray()[0])
            Y_nums.append(label)
        return Y,  Y_nums

    def save(self, model_path):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        output_model_file = os.path.join(model_path, WEIGHTS_NAME)
        output_config_file = os.path.join(model_path, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)


def load_data(X_path, Y_path, tail_X, tail_Y, bertReg, embs, emb_type):
    if os.path.isfile(X_path):
        with open(X_path, 'rb') as g:
            X = pkl.load(g)
    else:
        X = bertReg.get_bert_token(tail_X)
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
            Y, Y_nums = bertReg.get_label_elmo(tail_Y, embs)
        elif emb_type == 'n2v':
            Y, Y_nums = bertReg.get_label_n2v(tail_Y, embs)

        with open(Y_path, 'wb') as g:
            pkl.dump(Y, g)
        with open(Y_path + 'nums', 'wb') as g:
            pkl.dump(Y_nums, g)

    return X, Y,  Y_nums

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-ds", "--dataset", default="AmazonCat-13K", type=str, required=True)
    parser.add_argument("-t", "--tail_threshold", default=100, type=int)
    parser.add_argument("-le", "--label_embs", default='elmo', type=str)
    parser.add_argument("-gpu", "--device_num", default='1', type=str)
    parser.add_argument("-train", "--is_train", default=1, type=int)
    parser.add_argument("-ep", "--epochs", default=8, type=int)
    parser.add_argument("-ft", "--fine_tune", default=0, type=int)
    parser.add_argument("-from", "--ft_from", default=0, type=int)
    args = parser.parse_args()

    hypes = Hyperparameters(args.dataset)
    ds_path = '../datasets/' + args.dataset
    tail_threshold = args.tail_threshold
    device_num = args.device_num
    label_embs = args.label_embs
    fine_tune = args.fine_tune
    epochs = args.epochs
    is_train = args.is_train
    ft = (args.fine_tune == 1)
    ft_from = args.ft_from
    graphfile = ds_path+'/label_graph'
    label_graph = nx.read_edgelist(graphfile, create_using=nx.Graph())
    with open(ds_path+'/mlc2seq/heads-'+str(tail_threshold), 'rb') as g:
        heads = pkl.load(g)
    with open(ds_path+'/mlc2seq/train_heads_Y-'+str(tail_threshold), 'rb') as g:
        trn_head_Y = pkl.load(g)
    with open(ds_path+'/mlc2seq/test_heads_Y-'+str(tail_threshold), 'rb') as g:
        test_head_Y = pkl.load(g)
    answer = smat.load_npz(ds_path+'/Y.tst.npz')
    with open(ds_path+'/mlc2seq/tails-'+str(tail_threshold), 'rb') as g:
        tails = pkl.load(g)
    bertReg = BertRegressor(hypes, tails, label_embs, tail_threshold, device_num, epochs, ft, max_seq_len=hypes.max_seq_len)
    trn_X_path = ds_path+'/tail_data/trn_X-' + str(tail_threshold)
    test_X_path = ds_path+'/tail_data/test_X-' + str(tail_threshold)

    if label_embs == 'elmo' or label_embs == 'en':
        label_space = smat.load_npz(ds_path+'/L.elmo.npz')
        label_space = smat.lil_matrix(label_space)
        label_space[:tails[0]] = 9999
        trn_Y_path = ds_path+'/tail_data/trn_elmo_Y-' + str(tail_threshold)
        test_Y_path = ds_path+'/tail_data/test_elmo_Y-' + str(tail_threshold)
        with open(ds_path+'/mlc2seq/train_tails_X-'+str(tail_threshold), 'rb') as g:
            trn_tail_X = pkl.load(g)
        with open(ds_path+'/mlc2seq/train_tails_Y-'+str(tail_threshold), 'rb') as g:
            trn_tail_Y = pkl.load(g)
        trn_X, trn_Y, trn_Y_nums = load_data(trn_X_path, trn_Y_path, trn_tail_X, trn_tail_Y, bertReg, label_space, 'elmo')
        del(trn_tail_X, trn_tail_Y)
        with open(ds_path+'/mlc2seq/test_tails_X-'+str(tail_threshold), 'rb') as g:
            test_tail_X = pkl.load(g)
        with open(ds_path+'/mlc2seq/test_tails_Y-'+str(tail_threshold), 'rb') as g:
            test_tail_Y = pkl.load(g)
        test_X, test_Y, test_Y_nums = load_data(test_X_path, test_Y_path, test_tail_X, test_tail_Y, bertReg, label_space, 'elmo')
        del(test_tail_X, test_tail_Y)
    elif label_embs == 'n2v':
        lv = KV.load_word2vec_format(ds_path+'/label_n2v_embedding', binary=False)
        trn_Y_path = ds_path+'/tail_data/trn_n2v_Y-' + str(tail_threshold)
        test_Y_path = ds_path+'/tail_data/test_n2v_Y-' + str(tail_threshold)
        trn_X, trn_Y, trn_Y_nums = load_data(trn_X_path, trn_Y_path, trn_tail_X, trn_tail_Y, bertReg, lv, 'n2v')
        label_space = smat.csr_matrix(np.matrix(trn_Y))
        test_X, test_Y, test_Y_nums = load_data(test_X_path, test_Y_path, test_tail_X, test_tail_Y, bertReg, lv, 'n2v')
    else:
        print('invalid label embedding type')
        exit()
    print('Number of instnaces:', len(trn_X))
    print('Number of tail labels:', len(tails))

    if is_train:
        if ft:
            print('======================Start Fine-tuning======================')
            model_path = '../save_models/tail_regressor/'+hypes.dataset+'/t-'+str(tail_threshold)+'_ep-' + str(ft_from)+'-elmo/pytorch_model.bin'
            bertReg.train(trn_X, trn_Y, trn_Y_nums, label_space, test_X, test_Y, test_Y_nums, model_path, ft_from)
        else:
            print('======================Start Training======================')
            bertReg.train(trn_X, trn_Y, trn_Y_nums, label_space, test_X, test_Y, test_Y_nums)
        output_dir = '../save_models/tail_regressor/'+hypes.dataset+'/t-'+str(tail_threshold)+'_ep-' + str(epochs + ft_from) +'-'+ label_embs+'/'
        bertReg.save(output_dir)
    else:
        model_path = '../save_models/tail_regressor/' + hypes.dataset + '/t-' +str(tail_threshold)+'_ep-' + str(ft_from)+'-'+label_embs+'/pytorch_model.bin'
        print('======================Start Testing======================')
        accs =  bertReg.evaluate(test_X, test_Y, test_Y_nums, label_space, model_path, ft_from)

        ## predict head labels with predicted tail labels and evaluate
        # head_space = smat.load_npz(ds_path+'/L.elmo.npz')
        # head_space[tails[0]:] = 9999
        # bertReg.predict_heads_by_graph(test_X, test_head_Y, label_graph, heads, label_space, head_space, model_path)

        ## test with training data
        # accs =  bertReg.evaluate(trn_X, trn_Y, trn_Y_nums, label_space, model_path)


if __name__ == '__main__':
    main()

#
