# coding=utf-8

import numpy as np


class DataInput:
    def __init__(self, data, batch_size):
        """data input 初始化

        Args:
            data ([list]): 数据集
            batch_size ([int]): batch大小
        """
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        """TODO 似乎这个类本身就是迭代器

        Returns:
            [self]: 自身
        """
        return self

    def __next__(self):
        """自身 next迭代器

        Raises:
            StopIteration: 所有数据遍历一遍了

        Returns:
            i [int]: epoch num
            u, y 应该是user_id, label
        """
        if self.i == self.epoch_size:
            raise StopIteration
        start = self.i * self.batch_size
        end = min((self.i+1) * self.batch_size, len(self.data))
        ts = self.data[start:end]
        self.i += 1

        u, i, y, sl = [], [], [], []
        for t in ts:
            u.append(t[0])
            i.append(t[2])
            y.append(t[3])
            sl.append(len(t[1]))
        # 获取最长的行为序列长度
        max_sl = max(sl)
        # 存储行为历史
        hist_i = np.zeros([len(ts), max_sl], np.int64)

        k = 0
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
            k += 1
        return self.i, (u, i, y, hist_i, sl)


class DataInputTest:
    def __init__(self, data, batch_size):
        """data input 初始化

        Args:
            data ([list]): 数据集
            batch_size ([int]): batch大小
        """
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        start = self.i * self.batch_size
        end = min((self.i+1) * self.batch_size, len(self.data))
        ts = self.data[start:end]

        self.i += 1

        u, i, j, sl = [], [], [], []
        for t in ts:
            u.append(t[0])
            i.append(t[2][0])
            j.append(t[2][1])
            sl.append(len(t[1]))
        max_sl = max(sl)

        hist_i = np.zeros([len(ts), max_sl], np.int64)

        k = 0
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
            k += 1
        return self.i, (u, i, j, hist_i, sl)
