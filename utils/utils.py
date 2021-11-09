# -*- coding: utf-8 -*-
# @Time    : 2021/11/8 10:55
# @Author  : yanqun.jiang
# @File    : utils.py

import time


def print_run_time(func):
    """
    本模块作用是打印程序运行时间
    :param func: 要计算程序的时间
    :return:
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time-start_time
        print('>>>>>>>%s的程序运行时间为：%s' % (func.__name__, runtime))
    return wrapper


def print_df_head(func):
    """
    打印dataframe的前5行
    :param func:
    :return:
    """
    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)  # 这里需要返回dataframe类型
        print(df.head())
        return df

    return wrapper

