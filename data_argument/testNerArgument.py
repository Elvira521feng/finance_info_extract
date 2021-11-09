# -*- coding: utf-8 -*-
# @Time    : 2021/3/1 5:52 下午
# @Author  : JiangYanQun
# @File    : testNerArgument.py
import glob
import os

from data_argument.NerArgument import NerArgument
from utils.io_wrapper import read_file, write_file

ner_dir_name = '/data/total_data/conll_train'
data_augument_tag_list = ['ContractAmountUppercase', 'ContractName', 'ProjectName', 'ResponsibleDepartment',
                          'InstitutionName', 'StartTime', 'EndTime', 'ContractAmount', 'ContractSingleAmount',
                          'ConnectPerson', 'Phone', 'ContractSingleAmountUppercase']

ner = NerArgument(ner_dir_name=ner_dir_name,
                  ignore_tag_list=['O'],
                  data_augument_tag_list=data_augument_tag_list,
                  augument_size=6, seed=0)

data_sentence_arrs = []
data_label_arrs = []

files = glob.glob(os.path.join(ner_dir_name, '*.txt'))
txts = []
for file in files:
    txt = read_file(file)
    txts.append(txt)
    data_sentence_arrs_tmp, data_label_arrs_tmp = ner.augment(file_name=file)
    data_sentence_arrs.extend(data_sentence_arrs_tmp)
    data_label_arrs.extend(data_label_arrs_tmp)

ori_data = '\n\n'.join(txts)

# 3条增强后的句子、标签 数据，len(data_sentence_arrs)==3
# 你可以写文件输出函数，用于写出，作为后续训练等
# print(data_sentence_arrs, data_label_arrs)
lines = ''
for data_sentence_arr, data_label_arr in zip(data_sentence_arrs, data_label_arrs):
    for word, tag in zip(data_sentence_arr, data_label_arr):
        line = '\t'.join([word, tag])
        lines += line + '\n'

    lines += '\n'

# new_data = '\n'.join(lines)
new_data = lines
final_data = '\n\n'.join([ori_data, new_data])
print(new_data, end='')
print('原始文本的数量：', len(txts))
print('数据增强的数量：', len(data_sentence_arrs))
write_file('/data/total_data/train_argue.txt', final_data)
