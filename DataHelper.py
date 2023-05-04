# -*- Encoding:UTF-8 -*-

import pandas as pd
import numpy as np
from numpy import random
import tensorflow as tf

class Data:
    def __init__(self, batch_size, num_steps):
        # events_df = pd.read_csv('./data/retailrocket/events.csv')
        # category_tree_df = pd.read_csv('./data/retailrocket/category_tree.csv')
        # item_properties_1_df = pd.read_csv('./data/retailrocket/item_properties_part1.csv')
        # item_properties_2_df = pd.read_csv('./data/retailrocket/item_properties_part2.csv')
        # item_properties_df = pd.concat([item_properties_1_df, item_properties_2_df])
        # events_df = events_df.drop(['event','transactionid'],axis=1)
        # users = events_df['visitorid']
        # users = users.value_counts()
        # users = users.iloc[users.values > 10].index
        # users = users.sort_values()
        self.dataset = pd.read_csv('./data/retailrocket/events.csv')
        self.df,self.item_size = self.make_df()
        self.data_iter = self.dataset_loader(self.df, batch_size, num_steps)

    def make_df(self):
        events_df = self.dataset.sort_values(['visitorid', 'timestamp'])
        events_df = events_df.reset_index(drop=True)

        # 筛选出其中行动序列长度大于2的
        users = events_df['visitorid']
        users = users.value_counts()
        users = users.iloc[users.values > 2].index
        users = users.sort_values()

        group = events_df.groupby('visitorid',group_keys=False).groups
        # user_list=[]
        # action_list=[]
        # for user in self.dataset.visitorid.unique().tolist():
        #     item_list=[]
        #     for item in group[user]:
        #         item = self.dataset['itemid'][item]
        #         item_list.append(item)
        #     user_list.append(user)
        #     action_list.append(item_list)
        # data = {'user':user_list, 'actions':action_list}
        # df = pd.DataFrame(data)
        # return df
    
        user_list=[]
        action_list=[]
        all_items=[]
        for user in events_df.visitorid.unique().tolist():
            item_list=[]
            for item in group[user]:  # 将user访问过的不重复的item加入到item列表中
                item = events_df['itemid'][item]
                #print(item not in item_list)
                if item not in item_list:
                    item_list.append(item)
            if len(item_list) > 3: # 如果访问过的不重复的物品序列长度大于k则加入到最终的数据集中
                user_list.append(user)
                action_list.append(item_list)
                all_items += item_list
        item_set = set(all_items)
        item2id = dict(zip(item_set, range(len(item_set))))
        id_action_list=[]
        for lst in action_list:
            ids=[]
            for item in lst:
                item = item2id[item]
                ids.append(item)
            id_action_list.append(ids)   
        data = {'user':user_list, 'actions':id_action_list}
        df = pd.DataFrame(data)
        return df, len(item2id)

    def seq_data_iter_random(self, corpus, batch_size, num_steps):  #@save
        """使用随机抽样生成一个小批量子序列"""
        # 减去1，是因为我们需要考虑标签
        num_subseqs = (len(corpus) - 1) // num_steps
        # 长度为num_steps的子序列的起始索引
        initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
        # 在随机抽样的迭代过程中，
        # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
        random.shuffle(initial_indices)

        def data(pos):
            # 返回从pos位置开始的长度为num_steps的序列
            return corpus[pos: pos + num_steps]

        num_batches = num_subseqs // batch_size
        for i in range(0, batch_size * num_batches, batch_size):
            # 在这里，initial_indices包含子序列的随机起始索引
            initial_indices_per_batch = initial_indices[i: i + batch_size]
            X = [data(j) for j in initial_indices_per_batch]
            Y = [data(j + 1) for j in initial_indices_per_batch]
            yield tf.constant(X), tf.constant(Y)
            #yield X, Y

    def dataset_loader(self, df, batch_size, num_steps):
        for index in df.index:
            seq_data = df.loc[index].actions
            action_and_label = self.seq_data_iter_random(seq_data, batch_size, num_steps)
            yield index, action_and_label      