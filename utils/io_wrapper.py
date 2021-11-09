# ！/usr/bin/env python
# -*- coding:utf8 -*-

"""
本模块主要是一些文件读写操作
"""
import json
from collections import Iterable

from utils.constant import ENCODING_UTF8


def write_file(file_path, text, write_mode='w', encoding=ENCODING_UTF8):
    """
    写入文件
    :param file_path: 要写入的文件路径
    :param text: 要写入的文本
    :param write_mode: 写文件的模式，默认'w'
    :param encoding: 编码，默认utf8
    :return:
    """
    with open(file_path, write_mode, encoding=encoding) as fw:
        fw.write(text)


def write_lines_to_file(filename, lines, encoding=ENCODING_UTF8, is_filter_empty=False, is_strip=False):
    """
    按行写入文件
    :param filename: 要写入的文件路径
    :param lines: 要写入的句子列表
    :param encoding: 编码，默认utf8
    :param is_filter_empty: 是否过滤空行，默认False
    :param is_strip: 是否去掉两头的空格，默认False
    :return:
    """
    if not isinstance(lines, Iterable):
        raise Exception('data can\'t be iterated')

    if is_strip:
        if is_filter_empty:
            lines = [l.strip() for l in lines if l.strip()]
        else:
            lines = [l.strip() for l in lines]
    else:
        if is_filter_empty:
            lines = [l for l in lines if l]

    if not lines:
        raise Exception('lines are empty')

    with open(filename, 'w', encoding=encoding) as f:
        f.write('\n'.join(lines))


def read_lines_in_file(filename, encoding=ENCODING_UTF8, is_filter_empty=False, is_strip=False):
    """
    按行读文件
    :param filename: 要读的文件路径
    :param encoding: 编码，默认utf8
    :param is_filter_empty: 是否过滤空行，默认False
    :param is_strip: 是否去掉两头的空格，默认False
    :return: 句子列表
    """
    with open(filename, encoding=encoding) as f:
        if is_strip:
            if is_filter_empty:
                return [l.strip() for l in f.read().splitlines() if l.strip()]
            else:
                return [l.strip() for l in f.read().splitlines()]
        else:
            if is_filter_empty:
                return [l for l in f.read().splitlines() if l]
            else:
                return [l for l in f.read().splitlines()]


def read_file(filename, encoding=ENCODING_UTF8):
    """
    读文件
    :param filename: 要读的文件路径
    :param encoding: 编码，默认utf8
    :return: 文本内容
    """
    with open(filename, encoding=encoding) as f:
        return f.read()


def write_json_file(dest_filename, data):
    """
    将json打包写入json文件
    :param data: 要保存的内容
    :param dest_filename: 要保存的路径
    :return: None
    """
    with open(dest_filename, 'w', encoding=ENCODING_UTF8) as f:
        json.dump(data, f, ensure_ascii=False)


def read_json_file(src_filename):
    """
    读取json文件
    :param src_filename: 要读取的文件路径
    :return: json内容
    """
    with open(src_filename, encoding=ENCODING_UTF8) as f:
        return json.load(f)


def read_conll_file(path, encoding='utf-8',sep=None, indexes=None, dropna=True):
    r"""
    Construct a generator to read conll items.

    :param path: file path
    :param encoding: file's encoding, default: utf-8
    :param sep: seperator
    :param indexes: conll object's column indexes that needed, if None, all columns are needed. default: None
    :param dropna: weather to ignore and drop invalid data,
            :if False, raise ValueError when reading invalid data. default: True
    :return: generator, every time yield (line number, conll item)
    """

    def parse_conll(sample):
        sample = list(map(list, zip(*sample)))
        sample = [sample[i] for i in indexes]
        for f in sample:
            if len(f) <= 0:
                raise ValueError('empty field')
        return sample

    with open(path, 'r', encoding=encoding) as f:
        sample = []
        start = next(f).strip()
        if start != '':
            sample.append(start.split(sep)) if sep else sample.append(start.split())
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if line == '':
                if len(sample):
                    try:
                        res = parse_conll(sample)
                        sample = []
                        yield line_idx, res
                    except Exception as e:
                        if dropna:
                            logger.warning('Invalid instance which ends at line: {} has been dropped.'.format(line_idx))
                            sample = []
                            continue
                        raise ValueError('Invalid instance which ends at line: {}'.format(line_idx))
            elif line.startswith('#'):
                continue
            else:
                sample.append(line.split(sep)) if sep else sample.append(line.split())
        if len(sample) > 0:
            try:
                res = parse_conll(sample)
                yield line_idx, res
            except Exception as e:
                if dropna:
                    return
                logger.error('invalid instance ends at line: {}'.format(line_idx))
                raise e
