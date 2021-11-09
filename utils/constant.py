# ！/usr/bin/env python
# -*- coding:utf8 -*-
"""
本模块主要存放程序所需要的一些常量
"""

import os.path

# 常用路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前文件夹路径
PARENT_DIR = os.path.join(CURRENT_DIR, os.path.pardir)  # 当前文件夹的父级目录
DATA_DIR = os.path.join(PARENT_DIR, 'data')  # 数据存放路径
MODEL_SAVE_DIR = os.path.join(PARENT_DIR, 'model_save')  # 模型文件保存路径
CONFIG_DIR = os.path.join(PARENT_DIR, 'config')  # 模型文件保存路径

# 编码
ENCODING_UTF8 = 'utf-8'
