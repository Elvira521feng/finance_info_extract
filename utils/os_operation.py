# ！/usr/bin/env python
# -*- coding:utf8 -*-

"""
本模块是一些常用工具模块
"""

import os


def get_filenames_in_folder(folder_name, ext_name=True, hidden_file=False, attach_folder_name=True):
    """
    获取指定文件夹下的所有文件
    :param folder_name: 指定文件夹
    :param ext_name: 是否保留拓展文件名
    :param hidden_file: 是否列出隐藏文件
    :param attach_folder_name: 是否显示文件夹路径
    :return:
    """
    filenames = []
    if not os.path.exists(folder_name):
        raise Exception('folder is not existed.')
    for filename in os.listdir(folder_name):
        if hidden_file:
            if filename.startswith('.') and filename not in {'.', '..'}:
                filenames.append(filename)
        elif not filename.startswith('.'):
            filenames.append(filename)
    if attach_folder_name:
        filenames = [os.path.join(folder_name, name) for name in filenames]
    if not ext_name:
        filenames = [name[:name.rindex('.')] for name in filenames]
    return filenames


def mk_dir(dir_path):
    """
    创建文件夹
    :param dir_path:
    :return:
    """
    if os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        print(dir_path, '已存在！')

