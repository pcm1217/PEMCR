import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import os
import math
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
import json
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm
from torch.utils.data import DataLoader

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

def computeRePos(time_seq, time_span):
    
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i]-time_seq[j])
            if span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix

def Relation(user_train, usernum, maxlen, time_span):
    data_train = dict()
    for user in tqdm(range(1, usernum+1), desc='Preparing relation matrix'):
        time_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        data_train[user] = computeRePos(time_seq, time_span)
    return data_train

def computedisPos(dis_seq, dis_span):
    dis_span = dis_span
    size = len(dis_seq)
    dis_matrix = np.zeros([size, size], dtype=np.float64)
    for i in range(size):
        for j in range(size):
            # print(time_seq)
            lon1 = float(dis_seq[i].split(',')[0])
            lat1 = float(dis_seq[i].split(',')[1])
            lon2 = float(dis_seq[j].split(',')[0])
            lat2 = float(dis_seq[j].split(',')[1])
            span = int(abs(haversine(lon1, lat1, lon2, lat2)))
            if dis_seq[i] == '0,0' or dis_seq[j] == '0,0':
                dis_matrix[i][j] = dis_span
            elif span > dis_span:
                dis_matrix[i][j] = dis_span
            else:
                dis_matrix[i][j] = span
    return dis_matrix

def Relation_dis(user_train, usernum, maxlen, dis_span):
    data_train = dict()
    for user in tqdm(range(1, usernum + 1), desc='Preparing dis relation matrix'):
        dis_seq = ['0,0'] * maxlen
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            dis_seq[idx] = i[2]
            idx -= 1
            if idx == -1: break
        data_train[user] = computedisPos(dis_seq, dis_span)
    return data_train


def timeSlice(time_set):
    time_min = min(time_set)
    time_map = dict()
    for time in time_set:
        time_map[time] = int(round(float(time-time_min)))
    return time_map

def cleanAndsort(User, time_map):
    User_filted = dict()
    user_set = set()
    item_set = set()
    for user, items in User.items():
        user_set.add(user)
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u+1
    for i, item in enumerate(item_set):
        item_map[item] = i+1
    
    for user, items in User_filted.items():
        User_filted[user] = sorted(items, key=lambda x: x[1])

    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(map(lambda x: [item_map[x[0]], time_map[x[1]], x[2]], items))

    time_max = set()
    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))
        time_diff = set()
        for i in range(len(time_list)-1):
            if time_list[i+1]-time_list[i] != 0:
                time_diff.add(time_list[i+1]-time_list[i])
        if len(time_diff)==0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        time_min = min(time_list)
        User_res[user] = list(map(lambda x: [x[0], int(round((x[1]-time_min)/time_scale)+1), x[2]], items))
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))

    return User_res, len(user_set), len(item_set), max(time_max)

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_test = {}
    
    print('Preparing data...')
    f = open('/data/s2019020849/Mydata/GB_TKY/%s.csv' % fname, 'r')
    time_set = set()

    #user_count = defaultdict(int)
    #item_count = defaultdict(int)
    
    f = open('/data/s2019020849/Mydata/GB_TKY/%s.csv' % fname, 'r')

    for line in f:
        try:
            u, i, location, timestamp = line.rstrip().split('\t')
            if i == '':  # 如果i为空字符串，则跳过此行
                continue
            i = int(float(i))  # 将浮点数物品ID转换为整数
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(float(i))
        timestamp = float(timestamp)
        #if user_count[u]<5 or item_count[i]<5:
        #    continue
        time_set.add(timestamp)
        User[u].append([i, timestamp,location])
    f.close()
    time_map = timeSlice(time_set)
    User, usernum, itemnum, timenum = cleanAndsort(User, time_map)

    for user in User:
        nfeedback = len(User[user])
        split_index = int(0.8 * nfeedback)  # 前80%划分到训练集，剩下的20%划分到测试集
        user_train[user] = User[user][:split_index]
        user_test[user] = User[user][split_index:]
    print('Preparing done...')
    return [user_train, user_test, usernum, itemnum, timenum]


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')

def neg_sample(item_set, item_size):  
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item

class EarlyStopping:

    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):

        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
        
            if score[i] > self.best_score[i]+self.delta:
                return False
        return True

    def __call__(self, score, model):
        # score HIT@10 NDCG@10

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0]*len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation score increased.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)

def avg_pooling(x, dim):
    return x.sum(dim=dim)/x.size(dim)


def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def get_user_seqs(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    num_users = len(lines)
    num_items = max_item + 2

    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    return user_seq, max_item, valid_rating_matrix, test_rating_matrix

def get_user_seqs_long(data_file,Elist):
    lines = open(data_file).readlines()
    user_seq = []
    long_sequence = []
    item_set = set()

    def id_dict(fname):                                       
        itemdict = {}
        with open(fname,'r') as f:
            items =  f.readlines()                 
        for item in items:
            item = item.strip().split('\t')
            itemdict[item[1]] = int(item[0])+1
        return itemdict

    item_A = id_dict(Elist)

    for line in lines:
        user, items = line.strip().split('\t', 1)
        items = items.split('\t')

        new_items=[]

        for item in items:
            new_items.append(item_A[item])
        long_sequence.extend(new_items) 
        user_seq.append(new_items)
        item_set = item_set | set(new_items)
    max_item = max(item_set)

    return user_seq, max_item, long_sequence

def get_user_seqs_and_sample(data_file, sample_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    lines = open(sample_file).readlines()
    sample_seq = []
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        sample_seq.append(items)

    assert len(user_seq) == len(sample_seq)

    return user_seq, max_item, sample_seq

def get_item2attribute_json(data_file):
    item2attribute = json.loads(open(data_file).readline())
    attribute_set = set()
    for item, attributes in item2attribute.items():
        attribute_set = attribute_set | set(attributes)
    attribute_size = max(attribute_set) # 331
    return item2attribute, attribute_size

def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT /len(pred_list), NDCG /len(pred_list), MRR /len(pred_list)

def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)

def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):

    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))



def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


import torch

def get_eval(predlist, truelist, klist):
    recall = []
    mrr = []
    ndcg = []

    predlist = predlist.argsort()
    
    for k in klist:
        recall.append(0)
        mrr.append(0)
        ndcg.append(0)

        templist = predlist[:, -k:]
        
        i = 0
        while i < len(truelist):
            pos = torch.argwhere(templist[i] == (truelist[i] - 1))
            
            if len(pos) > 0:
                recall[-1] += 1
                
                mrr[-1] += 1 / (k - pos[0][0])

                rank = k - pos[0][0]
                dcg = 1 / torch.log2(rank + 2)
                idcg = 1 / torch.log2(torch.tensor(2.0))
                ndcg[-1] += dcg / idcg
            else:
                recall[-1] += 0
                mrr[-1] += 0
                ndcg[-1] += 0
            
            i += 1
    
    return recall, mrr, ndcg


def get_eval1(predlist, truelist, klist):
    recall = []
    mrr = []
    predlist = predlist.argsort()
    for k in klist:
        recall.append(0)

        mrr.append(0)

        templist = predlist[:,-k:]
        i = 0
        while i < len(truelist):
            pos = torch.argwhere(templist[i]==(truelist[i]-1))
        
            if len(pos) > 0:
                recall[-1] += 1
                mrr[-1] += 1/(k-pos[0][0])
            else:
                recall[-1] += 0
                mrr[-1] += 0
            i += 1
    return recall, mrr