# -*- coding: utf-8 -*-
# @Time    : 2020/5/17 15:18


import os
from collections import defaultdict
import random


def get_random(start, end):
    assert start <= end - 1
    return random.randint(start, end - 1)


class NerArgument(object):
    def __init__(self, ner_dir_name: str, ignore_tag_list: list,
                 data_argument_tag_list: list,
                 argument_size: int = 3,
                 seed: int = 0):
        random.seed(seed)
        self.ignore_tag_list = ignore_tag_list
        self.size = argument_size
        self.data_argument_tag_list = data_argument_tag_list
        self.tag_map = self.__get_all_tag_map(ner_dir_name)

    def __get_random_ner(self, tag: str):
        assert tag in self.tag_map
        max_size = len(self.tag_map[tag])
        assert max_size > 1
        select_idx = get_random(0, max_size)
        new_sene = self.tag_map[tag][select_idx]
        return new_sene

    def __get_all_tag_map(self, dir_name: str):
        """
        得到目录下全部标注文件的，各种实体，ignore_tag_list 里面的不要
        :param dir_name:
        :return:
        """
        tag_map = defaultdict(list)
        print(dir_name)
        for name in os.listdir(dir_name):
            file_path = os.path.join(dir_name, name)
            data_iter = self.__get_file_data_iter(file_path)
            for char_tag in data_iter:
                t_tag, t_ner_sentence = char_tag[0], char_tag[1]
                if t_tag in self.ignore_tag_list:
                    continue
                tag_map[t_tag].append(t_ner_sentence)
        print('tag_map:', tag_map)
        return tag_map

    @staticmethod
    def __get_file_data_iter(file_path: str):
        """
        NER的BIO标注文本 xxx.txt的file_path，要对它做增强
        :param file_path: 路径path
        :return:
        """
        with open(file_path, 'r', encoding='utf-8') as r_f:
            pre_tag = ''
            ner_sentence = ''
            for line in r_f:
                print(line)
                print(line.replace('\n', '').split('\t'))
                print(file_path)
                t_char, t_label = line.replace('\n', '').split('\t')
                tp_tag = 'O'
                if 'O' != t_label:
                    tp_tag = t_label.split('-')[1]
                if pre_tag == '':
                    pre_tag = tp_tag
                    ner_sentence += t_char
                elif pre_tag == tp_tag and t_label[0] != 'B':
                    ner_sentence += t_char
                else:
                    yield [pre_tag, ner_sentence]
                    pre_tag = tp_tag
                    ner_sentence = t_char
            if ner_sentence != '':
                print('pre_tag:', pre_tag)
                yield [pre_tag, ner_sentence]

    def __data_augment_one(self, org_data):

        new_data = []
        for di in org_data:
            t_tag, t_ner_sentence = di[0], di[1]
            if t_tag in self.data_argument_tag_list and t_tag in self.tag_map:
                rdm_select_ner = self.__get_random_ner(t_tag)
                new_data.append([t_tag, rdm_select_ner])
            else:
                new_data.append([t_tag, t_ner_sentence])
        return new_data

    def __data_augment(self, org_data, size=3):
        """
        对原始数据做增强
        :param org_data:
        :param size: 增强/最多/数量
        :return:
        """
        new_data = []
        org_sent = ''.join([di[1] for di in org_data])
        for i in range(size):
            o_new_data = self.__data_augment_one(org_data)
            new_sent = ''.join([di[1] for di in o_new_data])
            if org_sent != new_sent:
                new_data.append(o_new_data)
        return new_data

    @staticmethod
    def __parser_ner(ner_data):
        """
        数据还原成NER数组，字数组，标签数组
        :param ner_data:
        :return:
        """
        sentence_arr = []
        label_arr = []
        for i in range(len(ner_data)):
            entity_text_length = len(ner_data[i][1])
            for j in range(entity_text_length):
                if ner_data[i][0] == 'O':
                    label_arr.append(ner_data[i][0])
                else:
                    if j == 0:
                        label_arr.append('B-' + ner_data[i][0])
                    elif j == entity_text_length - 1:
                        label_arr.append('L-' + ner_data[i][0])
                    else:
                        label_arr.append('I-' + ner_data[i][0])
                # if entity_text_length == 1:
                #     label_arr.append('U-' + ner_data[i][0])
                sentence_arr.append(ner_data[i][1][j])
        return sentence_arr, label_arr

    def augment(self, file_name) -> tuple:
        """
        对文件做增强，输出文件路径，返回size个增强好的数据对 [sentence_arr, label_arr]
        :param file_name:
        :return:
        """
        org_data = list(self.__get_file_data_iter(file_name))
        new_data = self.__data_augment(org_data, self.size)
        data_sentence_arrs = []
        data_label_arrs = []
        for ndi in new_data:
            sentence_arr, label_arr = self.__parser_ner(ndi)
            data_sentence_arrs.append(sentence_arr)
            data_label_arrs.append(label_arr)
        return data_sentence_arrs, data_label_arrs

