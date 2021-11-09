# -*- coding: utf-8 -*-
# @Time    : 2021/11/8 17:25
# @Author  : yanqun.jiang
# @File    : logger.py


import logging
import sys

from os import makedirs
from os.path import dirname, exists

from constant import *

loggers = {}


def get_logger(name=None):
    """
    日志模块
    :param name: 日志模块名称
    :return: 日志记录器
    """
    global loggers

    if not name:
        name = __name__

    if loggers.get(name):
        return loggers.get(name)

    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # 输出到控制台
    if LOG_ENABLED and LOG_TO_CONSOLE:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=LOG_LEVEL)
        formatter = logging.Formatter(LOG_FORMAT)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # 输出到文件
    if LOG_ENABLED and LOG_TO_FILE:
        # 如果路径不存在，创建日志文件文件夹
        log_dir = dirname(LOG_PATH)
        if not exists(log_dir):
            makedirs(log_dir)
        # 添加 FileHandler
        file_handler = logging.FileHandler(LOG_PATH, encoding=ENCODING_UTF8)
        file_handler.setLevel(level=LOG_LEVEL)
        formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 保存到全局 loggers
    loggers[name] = logger
    return logger


if __name__ == '__main__':
    logger1 = get_logger('logger1')
    logger1.debug('测试！')
