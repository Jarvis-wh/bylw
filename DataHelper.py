# -*- Encoding:UTF-8 -*-

import pandas as pd
import numpy as np
from numpy import random
import tensorflow as tf

class Data:
    def __init__(self):
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
        self.dataset = pd.read_csv('./data/retailrocket/SLdata.csv')

    def make_action_list(self):
        group = self.dataset.groupby('visitorid',group_keys=False).groups
        user_list=[]
        action_list=[]
        for user in self.dataset.visitorid.unique().tolist():
            item_list=[]
            for item in group[user]:
                item = self.dataset['itemid'][item]
                item_list.append(item)
            user_list.append(user)
            action_list.append(item_list)
        data = {'user':user_list, 'actions':action_list}
        df = pd.DataFrame(data)
        return df

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