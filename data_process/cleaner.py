# ！/usr/bin/env python
# -*- coding:utf8 -*-
"""
本模块主要对数据进行清洗
"""
import os
import pandas

from utils.constant import DATA_DIR


def read_data(path):
    data = pandas.read_csv(path, delimiter='\t', header=None)
    print(data.head())


if __name__ == '__main__':
    path = os.path.join(DATA_DIR, 'ccks_4_1 Data/event_entity_train_data_label.csv')
    read_data(path)