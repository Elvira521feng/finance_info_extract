# -*- coding: utf-8 -*-
# @Time    : 2021/1/6 4:48 下午
# @Author  : JiangYanQun
# @File    : BasicNER.py

"""
使用Bert进行英文命名实体识别
"""
import argparse
import os
import sys

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')

import fitlog
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from fastNLP.io import DataBundle, ConllLoader, Pipe, ModelLoader
from fastNLP.io.file_reader import _read_conll
from fastNLP.io.pipe.utils import _indexize
from fastNLP import Trainer, Const
from fastNLP import BucketSampler, GradientClipCallback, Instance, DataSet, ClassifyFPreRecMetric, Vocabulary
from fastNLP.core.callback import SaveModelCallback, FitlogCallback
from fastNLP.core.optimizer import AdamW, Adam
from fastNLP import EvaluateCallback

from SequenceLabeling.data_process import generate_tag2label
from pufa.bert_embedding import BertEmbedding
from pufa.bert_sever import Tester
from bert_crf import BertCnnCRF
from utils.constant import MODEL_DIR, DATA_DIR


def setup(rank, world_size):
    """
    初始化DDP
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def _add_words_field(data_bundle, lower=False):
    r"""
    给data_bundle中的dataset中复制一列words. 并根据lower参数判断是否需要小写化

    :param data_bundle:
    :param bool lower:是否要小写化
    :return: 传入的DataBundle
    """
    data_bundle.copy_field(field_name=Const.RAW_WORD, new_field_name=Const.INPUT, ignore_miss_dataset=True)
    data_bundle.apply_field(lambda sentences: [list(s) for s in sentences], field_name=Const.INPUT, new_field_name=Const.INPUT)

    if lower:
        for name, dataset in data_bundle.datasets.items():
            dataset[Const.INPUT].lower()
    return data_bundle


class SeqPipe(Pipe):
    r"""
    NER任务的处理Pipe, 该Pipe会（1）复制raw_words列，并命名为words; (2）在words, target列建立词表
    (创建 :class:`fastNLP.Vocabulary` 对象，所以在返回的DataBundle中将有两个Vocabulary); (3）将words，target列根据相应的
    Vocabulary转换为index。

    raw_words列为List[str], 是未转换的原始数据; words列为List[int]，是转换为index的输入数据; target列是List[int]，是转换为index的
    target。返回的DataSet中被设置为input有words, target, seq_len; 设置为target有target, seq_len。
    """

    def __init__(self, lower: bool = False):
        r"""

        :param: str encoding_type: target列使用什么类型的encoding方式，支持bioes, bio两种。
        :param bool lower: 是否将words小写化后再建立词表，绝大多数情况都不需要设置为True。
        """
        self.lower = lower
        self.convert_tag = lambda words: [tag2label.get(word) for word in words]

    def process(self, data_bundle: DataBundle) -> DataBundle:
        r"""
        支持的DataSet的field为

        .. csv-table::
           :header: "raw_words", "target"

           "[Nadim, Ladki]", "[B-PER, I-PER]"
           "[AL-AIN, United, Arab, ...]", "[B-LOC, B-LOC, I-LOC, ...]"
           "[...]", "[...]"

        :param ~fastNLP.DataBundle data_bundle: 传入的DataBundle中的DataSet必须包含raw_words和ner两个field，且两个field的内容均为List[str]在传入DataBundle基础上原位修改。
        :return DataBundle:
        """
        # 转换tag
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(self.convert_tag, field_name=Const.TARGET, new_field_name=Const.TARGET)

        _add_words_field(data_bundle, lower=self.lower)

        # index
        _indexize(data_bundle)

        input_fields = [Const.TARGET, Const.INPUT, Const.INPUT_LEN]
        target_fields = [Const.TARGET, Const.INPUT_LEN]

        for name, dataset in data_bundle.datasets.items():
            if dataset.has_field(field_name=Const.INPUT):
                dataset.apply_field(len, Const.INPUT, new_field_name='seq_len')
                dataset.apply_field(lambda words: [len(word) for word in words], Const.INPUT, new_field_name='sentence_num')
            else:
                raise KeyError(f"Field:{Const.INPUT} not found.")

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle


class PuFaNERLoader(ConllLoader):
    r"""
    用于读取conll2003任务的NER数据。每一行有4列内容，空行意味着隔开两个句子

    支持读取的内容如下
    Example::

        Nadim NNP B-NP B-PER
        Ladki NNP I-NP I-PER

        AL-AIN NNP B-NP B-LOC
        United NNP B-NP B-LOC
        Arab NNP I-NP I-LOC
        Emirates NNPS I-NP I-LOC
        1996-12-06 CD I-NP O
        ...

    返回的DataSet的内容为

    .. csv-table:: 下面是Conll2003Loader加载后数据具备的结构, target是BIO2编码
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


class PuFaSequenceTagging(SeqPipe):
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
        data_bundle = PuFaNERLoader().load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle


def load_data(paths):
    # 替换路径

    # paths = 'data'
    data = PuFaSequenceTagging().process_from_file(paths)
    return data


def train(args):
    """
    训练
    :return:
    """
    # 获取数据路径
    data_path = os.path.join(DATA_DIR, args.data_path)

    # 加载数据
    print('>>>>>开始加载数据！')
    data = load_data(data_path)
    print('>>>>>数据加载完毕！')
    word_vocab = data.get_vocab(Const.INPUT)
    tag_vocab = data.get_vocab('target')
    print('>>>>>词表：', word_vocab)
    print('>>>>>标签映射表：：', tag_vocab)

    # 保存
    word_vocab_path = os.path.join(args.word_vocab_path, 'word.vocab')
    word_vocab.save(word_vocab_path)
    tag_vocab_path = os.path.join(args.word_vocab_path, 'tag.vocab')
    tag_vocab.save(tag_vocab_path)

    train_data = data.datasets['train']  # 训练集
    train_data.set_target('target', 'seq_len', 'sentence_num')
    dev_data = data.datasets['dev']  # 验证集
    dev_data.set_target('target', 'seq_len', 'sentence_num')
    test_data = data.datasets['test']  # 测试集
    test_data.set_target('target', 'seq_len', 'sentence_num')

    # 设置词向量
    embed = BertEmbedding(word_vocab, model_dir_or_name='cn-base', pool_method='first',
                          requires_grad=False, layers='11', include_cls_sep=True, dropout=0.5, word_dropout=0.01)

    # 设置callbacks
    callbacks = [
        GradientClipCallback(clip_type='norm', clip_value=1),
        EvaluateCallback(data.get_dataset('test')),
        SaveModelCallback(MODEL_DIR),
        # FitlogCallback(data.datasets['dev'])
    ]

    # 构建模型
    model = BertCnnCRF(embed, tag_vocab=data.get_vocab('target'))

    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=4e-5)

    if args.is_use_ddp:  # 是否使用DistributedDataParallel（多卡）
        print(">>>>>>>>>>>>>>>>> rank:", args.local_rank)
        rank = 0
        world_size = 1
        # setup(rank, world_size)  # 启动
        dist.init_process_group('nccl')
        torch.cuda.set_device(args.local_rank)
        model = model.to(args.local_rank)
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)

    # 设置训练器
    trainer = Trainer(train_data=train_data, model=model, optimizer=optimizer,
                      # sampler=BucketSampler(num_buckets=3),
                      device=None, dev_data=dev_data, batch_size=args.batch_size,
                      metrics=ClassifyFPreRecMetric(tag_vocab=data.vocabs[Const.TARGET]),
                      loss=None, callbacks=callbacks, num_workers=2, n_epochs=args.epochs,
                      check_code_level=0, update_every=1, test_use_tqdm=False)
    # 训练
    trainer.train()

    # 测试
    tester = Tester(data=test_data, model=model, metrics=ClassifyFPreRecMetric(tag_vocab=tag_vocab),
                    batch_size=args.batch_size)
    # 测试结果
    res = tester.test()
    print(res)
    if args.is_use_ddp:  # 是否使用DistributedDataParallel（多卡）
        cleanup()


def test(args):
    """
    预测
    :return:
    """
    data_path = os.path.join(DATA_DIR, args.data_path)
    data = load_data(data_path)
    print('=========')
    # print(data)
    print(data.get_vocab(Const.INPUT))
    data.datasets['test'].set_target('target', 'seq_len', 'sentence_num')
    data.datasets['test'].set_input('raw_words', 'words', 'seq_len', 'sentence_num')

    word_vocab = Vocabulary().load(args.word_vocab_path)
    tag_vocab = Vocabulary().load(args.tag_vocab_path)

    model_loader = ModelLoader()
    model = model_loader.load_pytorch_model(os.path.join(MODEL_DIR, args.model_path))
    # print(model)

    tester = Tester(data=data.get_dataset('test'), model=model, device=args.device,
                    metrics=ClassifyFPreRecMetric(tag_vocab=tag_vocab), batch_size=1)
    res = tester.test_predict(save_path='.')

    predict_samples = data.get_dataset('test').get_field('words')
    print(predict_samples)
    # model.predict(predict_samples)
    print(res)


if __name__ == '__main__':
    tag2label = generate_tag2label()

    # 从命令行传入参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', type=int, default=2)
    parser.add_argument('--device', default='cuda:2', help='cuda/cpu')
    parser.add_argument('--operation', type=str, default="train", help='训练/预测')
    parser.add_argument('--model_path', type=str, default="2021-07-08-19-21-33-149592/epoch-1_step-6_f-0.166667.pt", help='模型路径')
    parser.add_argument('--data_path', type=str, default="new", help='数据路径')
    parser.add_argument('--word_vocab_path', type=str, default="", help='词表路径')
    parser.add_argument('--tag_vocab_path', type=str, default="", help='标签表路径')
    parser.add_argument('--batch_size', type=int, default=2, help='')
    parser.add_argument('--epochs', type=int, default=10, help='')
    parser.add_argument('--is_use_ddp', type=bool, default=True, help='是否使用分布式')
    parser.add_argument('--local_rank', type=int, default=-1, help='')

    # fitlog.commit(__file__)             # 自动 commit 你的代码
    fitlog.set_log_dir("../logs/")  # 设定日志存储的目录

    args = parser.parse_args()
    fitlog.add_hyper(args)  # 通过这种方式记录ArgumentParser的参数
    fitlog.add_hyper_in_file(__file__)  # 记录本文件中写死的超参数

    if args.operation == 'train':
        train(args)
    elif args.operation == 'test':
        test(args)




