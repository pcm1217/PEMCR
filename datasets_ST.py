import json
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import torch
import geohash

class TVdatasets_all(Dataset):
    def __init__(self, Elist, Vlist, train_A, train_B, A_image, A_text, A_meta, B_image, B_text, B_meta, args, domain, offsets):
        print('Loading training set data......')
        self.Elist = Elist
        self.Vlist = Vlist
        self.train_A = train_A
        self.train_B = train_B
        self.maxlen = args.max_len
        self.min_len = args.min_len
        self.domain = domain
        self.offsets = offsets
        self.modal_dim = 768  

        self.A_image_embeddings = self.load_embeddings(A_image)
        self.A_text_embeddings = self.load_embeddings(A_text)
        self.A_meta_embeddings = self.load_embeddings(A_meta)
        self.B_image_embeddings = self.load_embeddings(B_image)
        self.B_text_embeddings = self.load_embeddings(B_text)
        self.B_meta_embeddings = self.load_embeddings(B_meta)

        if self.domain == 'A':
            user_data, user_target, user_locs, user_times = self.getdict(self.Elist, self.train_A, 0)
        elif self.domain == 'B':
            user_data, user_target, user_locs, user_times = self.getdict(self.Vlist, self.train_B, 0)
        else:
            user_data_A, user_target_A, user_locs_A, user_times_A = self.getdict(self.Elist, self.train_A, 0)
            start_id = len(user_data_A)
            user_data_B, user_target_B, user_locs_B, user_times_B = self.getdict(self.Vlist, self.train_B, start_id)
            user_data = {**user_data_A, **user_data_B}
            user_target = {**user_target_A, **user_target_B}
            user_locs = {**user_locs_A, **user_locs_B}
            user_times = {**user_times_A, **user_times_B}

        self.user_data, self.user_target, self.user_locs, self.user_times = self.sample_seq(user_data, user_target, user_locs, user_times, self.maxlen)

    def load_embeddings(self, file_path):
        embeddings = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                obj = json.loads(line)
                embeddings[obj["poi_id"]] = obj["embedding"] 
        return embeddings

    def getdict(self, E_list, train, start_id):
        User_all = defaultdict(list)
        User_target = defaultdict(list)
        User_locs = defaultdict(list)
        User_times = defaultdict(list)

        with open(train, 'r') as f:
            for line in f.readlines():
                line = line.strip().split('\t')
                user_id = int(line[0])
                item_id = int(line[1])
                #print(type(line[3]))
                loc_info = int(float(line[3]))
                time_info = float(line[2]) 
                User_all[user_id].append(item_id)
                User_locs[user_id].append(loc_info)
                User_times[user_id].append(time_info)

        for user in list(User_all.keys()):
            User_target[user] = User_all[user][-1]
            User_all[user] = User_all[user][:-1]
            User_locs[user] = User_locs[user][:-1]
            User_times[user] = User_times[user][:-1]
            if len(User_all[user]) < self.min_len:
                del User_all[user]
                del User_target[user]
                del User_locs[user]
                del User_times[user]

        return User_all, User_target, User_locs, User_times

    def sample_seq(self, user_train, user_target, user_locs, user_times, maxlen):
        new_user_id = 0
        finally_user_train = []
        finally_user_target = []
        finally_user_locs = []
        finally_user_times = []

        for user in user_train.keys():
            seq = np.zeros([maxlen], dtype=np.int32)
            times = np.zeros([maxlen], dtype=np.float32)
            loc = np.zeros([maxlen], dtype=np.int32)
            idx = maxlen - 1

            for i, (item_id, location, time) in enumerate(zip(user_train[user], user_locs[user], user_times[user])):
                seq[idx] = item_id
                times[idx] = time
                loc[idx] = location
                idx -= 1
                if idx == -1:
                    break
            finally_user_train.append(seq)
            finally_user_target.append(user_target[user])
            finally_user_times.append(times)
            finally_user_locs.append(loc)
            new_user_id += 1

        return finally_user_train, finally_user_target, finally_user_locs, finally_user_times

    def __getitem__(self, index):
        user_seq = self.user_data[index]
        user_target = self.user_target[index]
        user_locs = self.user_locs[index]
        user_times = self.user_times[index]

        img_emb = []
        text_emb = []
        meta_emb = []
        for item_id in user_seq:
            if item_id < self.offsets:  
                img_emb.append(self.A_image_embeddings.get(item_id, [0] * self.modal_dim))
                text_emb.append(self.A_text_embeddings.get(item_id, [0] * self.modal_dim))
                meta_emb.append(self.A_meta_embeddings.get(item_id, [0] * self.modal_dim))
            else:  # B 域
                img_emb.append(self.B_image_embeddings.get(item_id, [0] * self.modal_dim))
                text_emb.append(self.B_text_embeddings.get(item_id, [0] * self.modal_dim))
                meta_emb.append(self.B_meta_embeddings.get(item_id, [0] * self.modal_dim))

        img_emb = torch.tensor(img_emb, dtype=torch.float)  # [seq_len, img_dim]
        text_emb = torch.tensor(text_emb, dtype=torch.float)  # [seq_len, text_dim]
        meta_emb = torch.tensor(meta_emb, dtype=torch.float)  # [seq_len, meta_dim]
        user_locs = torch.tensor(user_locs, dtype=torch.long)  # [seq_len, 2]
        user_times = torch.tensor(user_times, dtype=torch.float)  # [seq_len]

        return torch.tensor(user_seq, dtype=torch.long), user_locs, user_times, img_emb, text_emb, meta_emb, torch.tensor(user_target, dtype=torch.long)

    def __len__(self):
        return len(self.user_data)
