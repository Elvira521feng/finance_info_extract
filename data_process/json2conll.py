# -*- coding: UTF-8 -*-
# @Time    : 2021/10/6 5:03 下午
# @Author  : JiangYanQun
"""
本模块主要是将json格式的标注数据转换成conll格式
"""
import copy
import random
import re
from collections import Counter
from intervaltree import IntervalTree
from itertools import accumulate
from brat2json import *


def token_json2label(paragraph_list, dest_filename, conll_delimiter='\t'):
    paragraph_list = [p for p in paragraph_list if p]
    if not paragraph_list:
        return None
    for key in {'tokens', 'labels'}:
        if key not in paragraph_list[0]:
            raise KeyError('key {0} must be in paragraph dict'.format(key))

    if not isinstance(paragraph_list[0]['tokens'][0], dict):
        raise TypeError('token must be in dict')

    token_keys = set(paragraph_list[0]['tokens'][0])
    extra_keys = set(token_keys).difference({'text', 'start', 'end', 'pos_tag'})
    conll_segments = []
    for paragraph in paragraph_list:
        conll_segment_lines = []
        for token, label in zip(paragraph['tokens'], paragraph['labels']):
            base_items = [token['text']]
            if 'pos_tag' in token:
                base_items.append(token['pos_tag'])
            if extra_keys:
                base_items.append([token[key] for key in extra_keys])
            base_items.append(label)
            conll_segment_lines.append(conll_delimiter.join(base_items))
        conll_segments.append('\n'.join(conll_segment_lines))
    write_file(dest_filename, '\n\n'.join(conll_segments))


def json2label(paragraph_list, label_schema='BILOU', pos_key='pos_tags'):
    """
    （多个）转换实体和文本为标签序列
    :param paragraph_list: json列表，包含text、entity等信息
    :param label_schema: 标签体系, 只允许'BILOU'、'BIO'、'BMEOS', and None（实体type大写作为标签）.
    :param pos_key: 词性的列名
    :return:
    """
    for item in paragraph_list:
        entity2label(item, label_schema, pos_key=pos_key)
    return paragraph_list


def entity2label(data, label_schema='BILOU', pos_key='pos_tags'):
    """
    （单个）将实体转换成标签序列
    :param data: 单个json，包含text、entity等信息
    :param label_schema: 标签体系, 只允许 'BILOU', 'BIO' , 'BMEOS', and None（实体type大写作为标签）.
    :param pos_key: 词性的列名
    :return: None
    """
    entities = data['entities']
    text = data['text']
    token_acc_len = list(accumulate(len(t) for t in data['tokens']))
    token_starts = [0] + token_acc_len[:-1]
    token_ends = token_acc_len
    token_spans = pos_spans = [(s, e) for s, e in zip(token_starts, token_ends) if s != e]
    split_indices = []

    for e in entities:
        if e['start'] not in token_starts:
            split_indices.append(e['start'])
        if e['end'] not in token_ends:
            split_indices.append(e['end'])

    split_indices = sorted(set(split_indices))
    for split_index in split_indices:
        tree = IntervalTree.from_tuples(token_spans)
        interval = list(tree.search(split_index))
        if len(interval) != 1:
            raise Exception('interval count error')
        interval = interval[0]
        if interval.begin == split_index:
            raise Exception('split index start error')
        elif interval.end == split_index:
            raise Exception('split index end error')
        token_spans.extend([(interval.begin, split_index), (split_index, interval.end)])
        if (interval.begin, interval.end) in token_spans:
            token_spans.remove((interval.begin, interval.end))

    token_spans = sorted(token_spans, key=lambda s: s[0])
    tokens = [text[s:e] for s, e in token_spans]

    if len(token_spans) != 0:
        token_starts, token_ends = list(zip(*token_spans))
        labels = ['O'] * len(token_spans)
    else:
        labels = []

    if len(set([s for s, e in token_spans])) < len(token_spans):
        dup = Counter([s for s, e in token_spans]).most_common(1)
        print(dup)
        raise Exception('have duplicate span')

    tree = IntervalTree.from_tuples(token_spans)
    for entity in entities:
        entity_type = entity['type']
        start_intervals = list(tree.search(entity['start']))
        if len(start_intervals) != 1:
            raise Exception('start interval count error')
        end_intervals = list(tree.search(entity['end'] - 1))
        if len(end_intervals) != 1:
            raise Exception('end interval count error')
        start_index = token_starts.index(start_intervals[0].begin)
        end_index = token_ends.index(end_intervals[0].end) + 1
        if labels[start_index] != 'O':
            continue
        if label_schema == 'BILOU':
            if end_index - start_index == 1:
                labels[start_index] = 'U-' + entity_type
            else:
                if end_index - start_index > 2:
                    labels[start_index + 1:end_index - 1] = ['I-' + entity_type] * (end_index - start_index - 2)
                labels[start_index] = 'B-' + entity_type
                labels[end_index - 1] = 'L-' + entity_type
        elif label_schema == 'BMEOS':
            if end_index - start_index == 1:
                labels[start_index] = 'S-' + entity_type
            else:
                if end_index - start_index > 2:
                    labels[start_index + 1:end_index - 1] = ['I-' + entity_type] * (end_index - start_index - 2)
                labels[start_index] = 'B-' + entity_type
                labels[end_index - 1] = 'E-' + entity_type
        elif label_schema == 'BIO':
            labels[start_index] = 'B-' + entity_type
            if end_index - start_index > 1:
                labels[start_index + 1:end_index] = ['I-' + entity_type] * (end_index - start_index - 1)
        elif not label_schema:
            labels[start_index:end_index] = [entity['type'].upper()] * (end_index - start_index)
        else:
            raise Exception('label schema is not supported.')
    pos_tags = []
    if pos_key not in data:
        print('don\'t have pos tags')
    for pos_tag, (start, end) in zip(data[pos_key], pos_spans):
        pos_start = token_starts.index(start)
        pos_end = token_ends.index(end)
        if pos_start == pos_end:
            pos_tags.append(pos_tag)
        elif pos_end - pos_start > 0:
            pos_tags.extend([pos_tag] * (pos_end - pos_start + 1))

    data['tokens'] = tokens
    data[pos_key] = pos_tags
    data['labels'] = labels

    return labels


def output_conll(paragraph_list, dest_filename, dest_dir_path=None, delimiter='\t', pos_key=None, has_context=True):
    """
    输出成conll格式的文件
    :param paragraph_list: 包含text、entity、label等信息的列表
    :param dest_filename: 保存文件夹
    :param delimiter: 分隔符号
    :param pos_key: 词性标签的关键词
    :return: None
    """
    conll_data = []
    i = 0
    text_set = set()
    valid_last_para_idx = -1  # 有效的上一句idx，相对于当前位置
    valid_last_paragraph = None
    valid_next_paragraph = None
    paragraphs_length = len(paragraph_list)  # 段落数量
    max_len = 0
    length_dict = {}
    for idx in range(paragraphs_length):
        paragraph = copy.deepcopy(paragraph_list[idx])
        if len(paragraph['text']) > 1000:
            print()
        # 有重复的句子则不进行处理
        if paragraph['text'] in text_set:
            continue

        # 生成随机数
        rand_num = random.randint(0, 9)

        # 句子为空则不作处理
        # 句子为表格也不作处理（如：TAB0）
        if len(paragraph['labels']) == 0 or re.match(r'TAB[\d]+', paragraph['text']):
            valid_last_para_idx -= 1  # 不作处理，但有效上一句要发生变化；显然上一句为空无效
            continue

        # 句子中没有实体则按10%的比例随机选择
        if set(paragraph['labels']) == set('O') and rand_num != 5:
            valid_last_para_idx = -1  # 不进行选择时，有效句要变成当前句
            continue

        if 0 < idx + valid_last_para_idx < paragraphs_length:
            valid_last_paragraph = copy.deepcopy(paragraph_list[idx + valid_last_para_idx])  # 拿到有效的上一句
        print("当前句:", paragraph['text'])

        if has_context and valid_last_paragraph and paragraph['file'] == valid_last_paragraph['file']:
            print("前一句:", valid_last_paragraph['text'])
            # print("valid_last_paragraph['labels']：", valid_last_paragraph['labels'])
            # print("valid_last_paragraph['tokens']:", valid_last_paragraph['tokens'])
            paragraph['text'] = ''.join([valid_last_paragraph['text'], paragraph['text']])  # 将上一句和当前句子拼接
            valid_last_paragraph['tokens'].extend(paragraph['tokens'])
            paragraph['tokens'] = copy.deepcopy(valid_last_paragraph['tokens'])
            valid_last_paragraph['labels'].extend(paragraph['labels'])
            paragraph['labels'] = copy.deepcopy(valid_last_paragraph['labels'])

        if 0 < idx + 1 < paragraphs_length:
            valid_next_paragraph = copy.deepcopy(paragraph_list[idx + 1])  # 拿到有效的下一句

        if has_context and valid_next_paragraph and paragraph['file'] == valid_next_paragraph['file'] and len(paragraph['text']) < 801:
            print("后一句:", valid_next_paragraph['text'])
            # print("valid_last_paragraph['labels']：", valid_next_paragraph['labels'])
            # print("valid_last_paragraph['tokens']:", valid_next_paragraph['tokens'])
            paragraph['text'] = ''.join([paragraph['text'], valid_next_paragraph['text']])  # 将上一句和当前句子拼接
            paragraph['tokens'].extend(valid_next_paragraph['tokens'])
            paragraph['labels'].extend(valid_next_paragraph['labels'])

        # print('paragraph:', paragraph)
        paragraph['tokens'].append('。')
        paragraph['labels'].append('O')
        valid_last_para_idx = -1

        if pos_key:
            lines = [delimiter.join((t, p, l)) for t, p, l in zip(paragraph['tokens'], paragraph[pos_key], paragraph['labels'])]
        else:
            lines = [delimiter.join((t, l)) for t, l in zip(paragraph['tokens'], paragraph['labels'])]

        # print(paragraph['file'])
        # print('-------line:', lines)
        i += 1
        tokens_num = len(paragraph['tokens'])

        print(tokens_num)
        if tokens_num in length_dict.keys():
            length_dict[tokens_num] += 1
        else:
            length_dict[tokens_num] = 1

        if tokens_num > max_len:
            max_len = tokens_num
        conll_data.append('\n'.join(lines))

        if dest_dir_path:
            write_file(os.path.join(dest_dir_path, str(i)) + '.txt',
                       '\n'.join(lines).replace('\n\tO\n', '').replace('\n\n\tO', '').
                       replace('\n		O', '').replace('		O\n', '').replace('\n\t\tO', ''))

        text_set.add(paragraph['text'])
    print('total num:', i)
    print("max_len:", max_len)
    print("length_dict:")
    for k, v in length_dict.items():
        print(k, ":", v)
    conll_data_len = len(conll_data)
    random.shuffle(conll_data)
    # dev_data = conll_data[:conll_data_len//10]
    # test_data = conll_data[conll_data_len // 10:conll_data_len // 10*2]
    # train_data = conll_data[conll_data_len // 10*2:]
    #
    # dev_filename = os.path.join(dest_filename, 'dev.txt')
    # test_filename = os.path.join(dest_filename, 'test.txt')
    # train_filename = os.path.join(dest_filename, 'train.txt')
    #
    # write_file(dev_filename, '\n\n'.join(dev_data))
    # write_file(test_filename, '\n\n'.join(test_data))
    # write_file(train_filename, '\n\n'.join(train_data))
    write_file(dest_filename, '\n\n'.join(conll_data))


if __name__ == '__main__':
    src_filename = os.path.join(DATA_DIR, 'vote/json_data/event_entity_train_data_label_new.json')
    dest_filename = os.path.join(DATA_DIR, 'vote/conll_data/event_entity_train_data_label_new.conll')
    tmp_paragraph_list = read_json_file(src_filename)
    tmp_paragraph_list = json2label(tmp_paragraph_list, label_schema='BILOU', pos_key='pos_tags')
    output_conll(tmp_paragraph_list, dest_filename, has_context=False)

