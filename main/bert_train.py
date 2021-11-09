# -*- coding: utf-8 -*-
# @Time    : 2021/1/6 4:48 下午
# @Author  : JiangYanQun

"""
使用Bert进行命名实体识别
"""
import argparse
import os
import sys
from fastNLP.embeddings import BertEmbedding
from fastNLP import Trainer, Const, EvaluateCallback, Tester
from fastNLP import BucketSampler, SpanFPreRecMetric, GradientClipCallback
from fastNLP.core.callback import WarmupCallback
from fastNLP.core.optimizer import AdamW

sys.path.append('../../../')

from model.bert_crf import BertCRF
from data_process.data_loader import NERPipe
from utils.constant import DATA_DIR

encoding_type = 'bioes'


def load_data():
    # 替换路径
    paths = os.path.join(DATA_DIR, 'ccks_4_1_Data/test/')
    data = NERPipe(encoding_type=encoding_type).process_from_file(paths)
    return data


def main(args):
    data = load_data()
    print('=========')
    print(data)

    embed = BertEmbedding(data.get_vocab(Const.INPUT), model_dir_or_name='cn-wwm-ext',
                          pool_method='first', requires_grad=True, layers='11', include_cls_sep=False,
                          dropout=0.5, word_dropout=0.01)
    print('+++++++++')

    callbacks = [
        GradientClipCallback(clip_type='norm', clip_value=1),
        # WarmupCallback(warmup=0.1, schedule='linear'),
        EvaluateCallback(data.get_dataset('test'))
    ]

    model = BertCRF(embed, tag_vocab=data.get_vocab('target'), encoding_type=encoding_type)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    trainer = Trainer(train_data=data.datasets['train'], model=model, optimizer=optimizer,
                      device=args.device, dev_data=data.datasets['dev'], batch_size=args.batch_size,
                      metrics=SpanFPreRecMetric(tag_vocab=data.vocabs[Const.TARGET], encoding_type=encoding_type),
                      loss=None, callbacks=callbacks, n_epochs=args.epochs,
                      check_code_level=0, update_every=1, test_use_tqdm=False)
    trainer.train()

    # tester = Tester()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='bert train')
    parser.add_argument('--lr', type=int, default=2e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=6, help='batch size')
    parser.add_argument('--epochs', type=int, default=5, help='epochs')
    parser.add_argument('--device', type=str, default='cpu', help='device')

    args = parser.parse_args()
    main(args)
