import glob
import os
import re

from fastNLP import DataSet

from dataset_loader import DataSetLoader
from utils.constant import DATA_DIR

from utils.io_wrapper import read_lines_in_file, write_file, read_file


def generate_tag2label():
    """
    :return:
    """
    tag2label = {}
    tags = ['add', 'minus', 'inAdd', 'inMinus']

    for i in range(len(tags)):
        tag = tags[i]
        tag2label[tag] = len(tag2label)

    tag2label['inInAdd'] = tag2label['inAdd']
    tag2label['inInMinus'] = tag2label['inMinus']
    tag2label['subtotal'] = tag2label['add']
    tag2label['total'] = tag2label['add']
    tag2label['undefine'] = tag2label['add']

    return tag2label


tag2label =generate_tag2label()
# label2tag = {value: key for key, value in tag2label.items()}
# label2tag = {0: 'add', 1: 'minus', 2: 'inAdd', 3: 'inMinus'}
label2tag = {0: 'add', 1: 'minus', 2: 'inAdd', 3: 'inMinus', 4:'undefine'}
# print(tag2label)
# print(label2tag)


def txt2DataSet(file_path):
    """
    :param file_path:
    :return:
    """
    lines = read_lines_in_file(file_path)
    pattern = re.compile("\[BEG\] (.*) \[END\]	\[BEG\] (.*) \[END\]")
    labels_list = []
    sentence_list = []
    words_list = []
    seq_len_list = []

    for line in lines:
        res = pattern.search(line)
        labels = res.group(1).split()
        print(labels)
        if "discarded" in labels:
            continue
        sentence = res.group(2)
        words = sentence.split()

        seq_len = len(words)
        labels_list.append([tag2label[label] for label in labels])
        sentence_list.append(sentence)
        words_list.append(words)
        seq_len_list.append(seq_len)

    txt_dict = {"sentence": sentence_list, "words": words_list, "seq_len": seq_len_list, "target": labels_list}
    data_set = DataSet(txt_dict)

    return data_set


def Conll2DataSet(file_path):
    """
    :param file_path:
    :return:
    """
    loader = DataSetLoader()
    data_bundle = loader.load(file_path)
    data_set = data_bundle.datasets['train']
    # vocab = data_bundle.get_vocab('words')
    # tag_vocab = data_bundle.get_vocab('target')

    labels = data_set.get_field("target")
    tags = []
    # print(data_set.get_field('target'))
    for label in labels:
        # print(label)
        label = [tag2label[l] for l in label]
        # print(label)
        tags.append(label)

    seq_len_list = [len(words) for words in data_set.get_field('words')]
    data_set.add_field('seq_len', seq_len_list)
    data_set.add_field('target', tags)
    # print(data_set.get_field('target'))

    return data_bundle


def build_word_list(dir_path, dest_path, words=None):
    file_list = glob.glob(os.path.join(dir_path, '*/*.txt'))

    if words:
        word_list = words
    else:
        word_list = set()

    for file in file_list:
        print(file)
        tmp_words = set(read_file(file).split('<>'))
        word_list.update(tmp_words)

    word_list_str = ' '.join(word_list)
    print(len(word_list))
    write_file(dest_path, word_list_str)

if __name__ == '__main__':
    path =DATA_DIR + '4_annotated_data\\2019_07_29\\'
    dest_path = DATA_DIR +  '5_data_for_train/v2/word_list.txt'
    txt_path = DATA_DIR + '5_data_for_train/v_test/word_list.txt'
    words = set(read_file(txt_path).split())
    word_list = build_word_list(path, dest_path, words=words)
    write_file(dest_path, " ".join(word_list))
    print('finish')
    path = DATA_DIR + 'data_for_trainning_v1/data.conll'
    print(Conll2DataSet(path))

