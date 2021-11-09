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
LOG_DIR = os.path.join(PARENT_DIR, 'log')   # 日志文件路径
LOG_PATH = os.path.join(LOG_DIR, 'runtime.log')   # 日志文件路径

# 编码
ENCODING_UTF8 = 'utf-8'

# 日志配置项
LOG_ENABLED = True  # 是否开启日志
LOG_TO_CONSOLE = True  # 是否输出到控制台
LOG_TO_FILE = True  # 是否输出到文件
LOG_TO_ES = True  # 是否输出到 Elasticsearch
LOG_LEVEL = 'DEBUG'  # 日志级别
LOG_FORMAT = '%(levelname)s - %(asctime)s - process: %(process)d - %(filename)s - %(name)s - %(lineno)d - %(module)s - %(message)s'  # 每条日志输出格式
ELASTIC_SEARCH_HOST = 'eshost'  # Elasticsearch Host
ELASTIC_SEARCH_PORT = 9200  # Elasticsearch Port
ELASTIC_SEARCH_INDEX = 'runtime'  # Elasticsearch Index Name
APP_ENVIRONMENT = 'dev'  # 运行环境，如测试环境还是生产环境
