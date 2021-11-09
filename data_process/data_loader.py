# -*- coding: utf-8 -*-
# @Time    : 2021/3/1 10:13 上午
# @Author  : JiangYanQun
# @File    : data_loader.py
from fastNLP import DataSet, Instance
from fastNLP.io import ConllLoader, DataBundle
from fastNLP.io.file_reader import _read_conll
from fastNLP.io.pipe.conll import _NERPipe


class Conll2003NERPipe(_NERPipe):
    r"""
    Conll2003的NER任务的处理Pipe, 该Pipe会（1）复制raw_words列，并命名为words; (2）在words, target列建立词表
    (创建 :class:`fastNLP.Vocabulary` 对象，所以在返回的DataBundle中将有两个Vocabulary); (3）将words，target列根据相应的
    Vocabulary转换为index。
    经过该Pipe过后，DataSet中的内容如下所示

    .. csv-table:: Following is a demo layout of DataSet returned by Conll2003Loader
       :header: "raw_words", "target", "words", "seq_len"

       "[Nadim, Ladki]", "[1, 2]", "[2, 3]", 2
       "[AL-AIN, United, Arab, ...]", "[3, 4,...]", "[4, 5, 6,...]", 6
       "[...]", "[...]", "[...]", .

    raw_words列为List[str], 是未转换的原始数据; words列为List[int]，是转换为index的输入数据; target列是List[int]，是转换为index的
    target。返回的DataSet中被设置为input有words, target, seq_len; 设置为target有target。

    dataset的print_field_meta()函数输出的各个field的被设置成input和target的情况为::

        +-------------+-----------+--------+-------+---------+
        | field_names | raw_words | target | words | seq_len |
        +-------------+-----------+--------+-------+---------+
        |   is_input  |   False   |  True  |  True |   True  |
        |  is_target  |   False   |  True  | False |   True  |
        | ignore_type |           | False  | False |  False  |
        |  pad_value  |           |   0    |   0   |    0    |
        +-------------+-----------+--------+-------+---------+

    """

    def process_from_file(self, paths) -> DataBundle:
        r"""

        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.ConllLoader` 的load函数。
        :return: DataBundle
        """
        # 读取数据
        data_bundle = BaoWuNERLoader().load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle


class BaoWuNERLoader(ConllLoader):
    r"""
    用于读取宝武合同的NER数据。每一行有2列内容，空行意味着隔开两个句子

    支持读取的内容如下
    Example::

        甲 O
        方 O
        ： O
        源 B-ResponsibleDepartment
        美 I-ResponsibleDepartment
        智 I-ResponsibleDepartment
        能 I-ResponsibleDepartment
        ...

    返回的DataSet的内容为

    .. csv-table:: 下面是BaoWuNERLoader加载后数据具备的结构, target是BIO2编码
       :header: "raw_words", "target"

       "[Nadim, Ladki]", "[B-PER, I-PER]"
       "[AL-AIN, United, Arab, ...]", "[B-LOC, B-LOC, I-LOC, ...]"
       "[...]",  "[...]"

    """

    def __init__(self):
        headers = [
            'raw_words', 'target',
        ]
        super().__init__(headers=headers, indexes=[0, 1])

    def _load(self, path):
        r"""
        传入的一个文件路径，将该文件读入DataSet中，field由ConllLoader初始化时指定的headers决定。

        :param str path: 文件的路径
        :return: DataSet
        """
        ds = DataSet()
        for idx, data in _read_conll(path, indexes=self.indexes, dropna=self.dropna):
            doc_start = False
            for i, h in enumerate(self.headers):
                field = data[i]
                if str(field[0]).startswith('-DOCSTART-'):
                    doc_start = True
                    break
            if doc_start:
                continue
            ins = {h: data[i] for i, h in enumerate(self.headers)}
            ds.append(Instance(**ins))
        if len(ds) == 0:
            raise RuntimeError("No data found {}.".format(path))
        return ds

    def download(self):
        raise RuntimeError("conll2003 cannot be downloaded automatically.")