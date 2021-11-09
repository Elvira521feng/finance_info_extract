# -*- coding: utf-8 -*-
# @Time    : 2021/11/8 10:55
# @Author  : yanqun.jiang
# @File    : brat2json.py

"""
本模块主要是将brat标注好的ann文件转化成json格式
"""
import json
import os

from utils.constant import DATA_DIR
from utils.io_wrapper import write_json_file, read_file, write_file, read_lines_in_file, read_json_file
from utils.os_operation import get_filenames_in_folder


def check_entities(entities, text):
    """
    检查实体是否合法，即检查通过【start，end】是否能够在原文中拿到实体的文本内容，有问题则抛出异常
    :param entities: 实体列表
    :param text: 原文
    :return: None
    """
    for entity in entities:
        start = entity['start']
        end = entity['end']
        if start < 0 or end < 0:
            raise Exception('offset is negative')
        if text[start:end] != entity['entity']:
            print('=====================')
            print(text)
            print(entity)
            print(text[start:end])
            # raise Exception('entity text and offset don\'t correspond.')
            print('=====================')
        else:
            print('**********************')
            print(text)
            print(entity)
            print(text[start:end])
            print('*********************')


def adjust_entities_offsets(entity_list, offset, start=None, end=None):
    """
    调整实体的位移
    :param entity_list: 实体列表
    :param offset: 位移
    :param start: 区间的开始位置
    :param end: 区间的结束位置
    :return:
    """
    for entity in entity_list:
        not_restrict = not start and not end
        restrict_start = start and start <= entity['start']
        restrict_end = end and end >= entity['end']
        restrict_all = start and end and start <= entity['start'] < entity['end'] <= end
        if not_restrict or restrict_all or restrict_start or restrict_end:
            print('entity:', entity, 'offset:', offset)
            print('old:', entity['start'], entity['end'])
            entity['start'] += offset
            entity['end'] += offset
            print('new:', entity['start'], entity['end'])
    return entity_list


def brat2json_dir(dirname, dest_filename=None):
    """
    转换指定文件夹中的标注数据，ann-->json
    :参数 dirname: brat标注数据文件夹
    :参数 dest_filename: 目标文件夹, 为None则不输出
    :返回: 解析后的段落列表
    """
    paragraphs = []
    for filename in get_brat_filename(dirname):
        print(filename)
        paragraphs.extend(brat2json_file(filename))

    if dest_filename:
        write_json_file(dest_filename, paragraphs)

    return paragraphs


def brat2json_file(src_filename, dest_filename=None):
    """
    转换单个（brat）标注数据文件为json文件
    :参数 dirname: brat标注数据文件夹
    :参数 dest_filename: 目标文件夹, 为None则不输出
    :返回: 解析的段落列表
    """
    paragraphs = []
    entities = parse_ann_file(src_filename + '.ann')
    text = read_file(src_filename + '.txt')
    meta_info = None
    # tokens = None

    # if os.path.exists(src_filename + '.meta'):
    #     meta_info = read_meta_info(src_filename)

    # if os.path.exists(src_filename + '.tok'):
    # tokens = parse_tok_file(src_filename)

    for line_index, (line, (start, end)) in enumerate(zip(*split_into_lines(text))):
        line_entities = select_entity_by_offset(entities, start, end, line)
        line_entities = adjust_entities_offsets(line_entities, -start)
        check_entities(line_entities, line)
        dirname = os.path.dirname(src_filename)
        basename = os.path.basename(src_filename)
        paragraph = {'text': line, 'entities': line_entities, 'dir': dirname, 'file': basename}
        tokens = list(line)
        paragraph['tokens'] = tokens
        paragraph['pos_tags'] = ['o'] * len(tokens)

        if meta_info:
            paragraph.update(meta_info[line_index])
        # if tokens:
        #     paragraph['tokens'] = tokens['tokens'][line_index]
        #     if 'pos_tags' in tokens:
        #         paragraph['pos_tags'] = tokens['pos_tags'][line_index]
        paragraphs.append(paragraph)
    if dest_filename:
        write_json_file(dest_filename, paragraphs)
    return paragraphs


def brat2jsonl(filenames, dest_filename):
    """
    将brat文件转换成jsonl文件
    :param filenames: brat文件名, 不带后缀
    :param dest_filename: jsonl的保存路径
    :return: 返回jsonl内容
    """
    metadata = read_meta_infos(filenames)
    paragraphs = []
    for filename in filenames:
        entities = parse_ann_file(filename + '.ann')
        text = read_file(filename + '.txt')

        for line_index, (line, (start, end)) in enumerate(zip(*split_into_lines(text))):
            line_entities = select_entity_by_offset(entities, start, end, line)
            line_entities = adjust_entities_offsets(line_entities, -start)
            paragraph = metadata[filename][line_index]
            paragraph['text'] = line
            paragraph['entities'] = line_entities
            paragraphs.append(paragraph)
    write_file(dest_filename, '\n'.join(json.dumps(para) for para in paragraphs))
    return paragraphs


def parse_ann_file(filename):
    """
    解析ann文件
    :param filename: ann文件名
    :return: 返回解析后的实体列表
    """
    entities = []
    for line in read_lines_in_file(filename):
        print(line)
        index, metainfo, text = line.split('\t')
        if index.startswith('T'):
            entity_type, start, end = metainfo.split(' ')
            start = int(start)
            end = int(end)
            entity = {'entity': text, 'start': start, 'end': end, 'type': entity_type, 'id': index}
            entities.append(entity)
    return entities


def parse_tok_file(filename):
    """
    解析分词文件
    :param filename: 分词文件名，不带后缀
    :return: 解析后的分词和词性标签
    """
    tok_lines = read_json_file(filename + '.tok')
    raw_lines = read_lines_in_file(filename + '.txt', is_filter_empty=False)

    if len(raw_lines) != len(tok_lines):
        raise Exception('line count not equal')
    if not tok_lines:
        return [[]]
    is_pos = all('_' in tok for tok in tok_lines[0].split(' ') if tok)

    tokens = []
    pos_tags = []
    for tok_line in tok_lines:
        sent_tokens = []
        sent_pos_tags = []
        for token in tok_line.rstrip().split(' '):
            if not token:
                continue
            if is_pos:
                split_index = token.rindex('_')
                sent_tokens.append(token[:split_index])
                sent_pos_tags.append(token[split_index + 1:])
            else:
                sent_tokens.append(token)
        if is_pos:
            pos_tags.append(sent_pos_tags)
        tokens.append(sent_tokens)

    result = {'tokens': tokens}
    if is_pos:
        result['pos_tags'] = pos_tags

    return result


def parse_ann_folder(dir_name, has_text=False):
    """
    解析多个ann文件
    :param dir_name: 存放ann文件的文件夹
    :param has_text: 是否要保存ann对应的文本内容
    :return: 解析后的实体列表
    """
    filenames = set()
    data = []
    for filename in os.listdir(dir_name):
        if not filename.startswith('.') and filename.endswith('.ann'):
            filenames.add(dir_name + '/' + filename[:filename.rindex('.')])
    for filename in filenames:
        entities = parse_ann_file(filename)
        if has_text:
            text = read_file(filename + '.txt')
            data.append({'text': text, 'entities': entities, 'index': filename[filename.rindex('/') + 1:]})
        else:
            data.append(entities)
    return data


def get_brat_filename(dirname):
    """
    获取文件夹中的（brat）标注数据, 不带前缀
    :参数 dirname: （brat）标注数据所在文件夹
    :返回: （brat）标注数据文件名列表
    """
    filenames = set()

    for filename in get_filenames_in_folder(dirname):
        if filename.endswith(('.txt', '.ann')):
            filenames.add(filename[:filename.rindex('.')])

    for filename in filenames:
        if not os.path.exists(filename + '.txt') or not os.path.exists(filename + '.ann'):
            raise Exception('brat file does not exist.')

    return list(filenames)


def split_into_lines(text, split_tag='\n', len_upper_limit=800, is_skip_empty=True):
    """
    使用指定分隔符来分句
    :param text: 原文本
    :param split_tag: 分隔符
    :param len_upper_limit: 句子长度的最大上限，并且保证句子的完整性
    :param is_skip_empty: 是否跳过空行
    :return: 句子列表
    """
    start = 0
    spans = []
    lines = []
    text_splits = text.split(split_tag)

    text_splits_ = []
    text_splits_str = ''
    for idx, text_split in enumerate(text_splits):
        if not len(text_split.strip('\n')) and is_skip_empty:
            continue

        if idx == 0:
            text_splits_str = text_split
        elif len(text_splits_str) + len(text_split) + len(split_tag) < len_upper_limit:
            text_splits_str += split_tag + text_split
        else:
            text_splits_.append(text_splits_str)
            text_splits_str = text_split

    if len(text_splits_str) != 0:
        text_splits_.append(text_splits_str)

    text_splits = text_splits_

    for line in text_splits:
        lines.append(line)
        end = start + len(line)
        spans.append((start, end))
        start = end + len(split_tag)
        print('----', line, (start, end))

    return lines, spans


def select_entity_by_offset(entity_list, start, end, line):
    """
    判断实体的开始结束位置是否在句子的开始结束位置区间中，满足的实体放入结果集合；
    判断实体列表的内容和文本中的内容是否一直一致
    :param entity_list: 实体列表
    :param start: 句子开始下标
    :param end: 句子结束下标
    :param line: 句子内容
    :return: 实体结果集
    """
    new_entity_list = []
    for entity in entity_list:
        entity_start = entity['start']
        entity_end = entity['end']
        if start <= entity_start < entity_end <= end:
            new_entity_list.append(entity)
            print('------------')
            print('is?', entity['entity'] == line[entity_start:entity_end])
            print('>>>>>>>>>>entity:', entity)
            print('>>>>>>>>>>line[entity_start:entity_end]:', line[entity_start:entity_end])
    return new_entity_list


def read_meta_info(filename):
    """
    从单个文件读元数据信息
    :param filename: 文件路径，没有后缀
    :return: parsed meta info, flatten into list
    """
    paragraph_infos = []
    if not os.path.exists(filename + '.meta'):
        raise Exception('meta info file doesn\'t existed')
    for line in read_lines_in_file(filename + '.meta'):
        print(line)
        patent_id, section, index = line.split('\t')
        index = int(index)
        dirname, basename = os.path.split(filename)
        item = {'patent_id': patent_id, 'section': section, 'index': index,
                'dirname': dirname, 'filename': basename}
        paragraph_infos.append(item)

    return paragraph_infos


def read_meta_infos(filenames):
    """
    read meta info files
    :param filenames: meta info file paths
    :return: parsed meta info
    """
    data = {}
    for filename in filenames:
        data[filename] = read_meta_info(filename)
    return data


if __name__ == '__main__':
    source_dir_path = os.path.join(DATA_DIR, 'vote/annotated_data')
    dest_dir_path = os.path.join(DATA_DIR, 'vote/json_data/event_entity_train_data_label_new.json')
    brat2json_dir(source_dir_path, dest_dir_path)
