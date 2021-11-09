# -*- coding: utf-8 -*-
# @Time    : 2021/10/6 5:03 下午
# @Author  : JiangYanQun
"""
本模块主要通过代码对数据进行预标注
"""
import os
import re

import pandas
from flashtext import KeywordProcessor

from utils.constant import DATA_DIR
from utils.io_wrapper import write_file


def test_flashtext():
    """
    测试flashtext
    :return:
    """
    keyword_processor = KeywordProcessor()
    keyword_processor.add_keywords_from_list(['你好', '再见'])
    sample_sentence = '你好，我叫琳达，再见！'
    keywords_found = keyword_processor.extract_keywords(sample_sentence, span_info=True)
    print(keywords_found)


def annotate(text, entity, entity_type):
    """
    代码标注实体
    :param text:
    :param entity:
    :param entity_type:
    :return:
    """
    ann_res = []
    keyword_processor = KeywordProcessor()
    keyword_processor.add_keyword(entity)
    keywords_found = keyword_processor.extract_keywords(text, span_info=True)
    print('>>>>>>>>>>keywords_found:', keywords_found)
    for i, kf in enumerate(keywords_found):
        print(kf)
        start = str(kf[1])
        end = str(kf[2])
        ann = '\t'.join(['T' + str(i), ' '.join([entity_type, start, end]), entity])
        print('>>>>>>>>>ann', ann)
        ann_res.append(ann)

    return ann_res


def annotate_file(file_path, dest_dir):
    """
    标注并生成brat格式的文件
    :param file_path:
    :param dest_dir:
    :return:
    """
    data = pandas.read_csv(file_path, delimiter='\t', header=None)
    data = data.dropna(axis=0, how='all')
    # print(data.shape)

    new_file_path = file_path.replace('.csv', '_new.csv')
    data = data.dropna(axis=0, how='any')
    # print(data.shape)

    data.to_csv(new_file_path, index=None, header=['news_id', 'text', 'subject', 'class'], sep='\t')
    entity_type = 'subject'

    # text_dict = {}  # 文本去重后的集合
    text_set = []

    for index, items in data.iterrows():
        # print(items)
        news_id = items[0]
        text = items[1]
        text = re.sub('[\s\t]+', '', text)
        subject = items[3]
        txt_path = os.path.join(dest_dir, str(index + 1) + '_' + str(news_id) + '.txt')
        write_file(txt_path, text)
        ann_res = annotate(text, subject, entity_type)
        ann_path = os.path.join(dest_dir, str(index + 1) + '_' + str(news_id) + '.ann')
        write_file(ann_path, '\n'.join(ann_res))

        if text not in text_set:
            text_set.append(text)


if __name__ == '__main__':
    path = os.path.join(DATA_DIR, 'ccks_4_1_Data/event_entity_train_data_label.csv')
    dest_dir = os.path.join(DATA_DIR, 'ccks_4_1_Data/annotated_data')
    annotate_file(path, dest_dir)
