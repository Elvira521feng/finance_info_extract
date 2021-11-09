import argparse
import os

from fastNLP import Const, GradientClipCallback, EvaluateCallback, SaveModelCallback, AdamW, Trainer, \
    ClassifyFPreRecMetric, Tester, BucketSampler
from fastNLP.embeddings import StaticEmbedding

from model.bilstm_crf import BiLstmCrf
from utils.constant import DATA_DIR, MODEL_SAVE_DIR
from utils.os_operation import mk_dir


def load_data():
    # 替换路径
    paths = os.path.join(DATA_DIR, 'test')
    data = Conll2003NERPipe(encoding_type=encoding_type).process_from_file(paths)
    return data


def main(args):
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
    model_save_dir = os.path.join(MODEL_SAVE_DIR, args.model_save_path)
    mk_dir(model_save_dir)
    word_vocab_path = os.path.join(model_save_dir, 'word.vocab')
    word_vocab.save(word_vocab_path)
    tag_vocab_path = os.path.join(model_save_dir, 'tag.vocab')
    tag_vocab.save(tag_vocab_path)

    train_data = data.datasets['train']  # 训练集
    train_data.set_target('target', 'seq_len', 'sentence_num')
    dev_data = data.datasets['dev']  # 验证集
    dev_data.set_target('target', 'seq_len', 'sentence_num')
    test_data = data.datasets['test']  # 测试集
    test_data.set_target('target', 'seq_len', 'sentence_num')

    # 设置词向量
    embed = StaticEmbedding()

    # 设置callbacks
    callbacks = [
        GradientClipCallback(clip_type='norm', clip_value=1),
        EvaluateCallback(data.get_dataset('test')),
        SaveModelCallback(model_save_dir),
    ]

    # 构建模型
    model = BiLstmCrf(embed, args.num_classes, args.num_layers, args.hidden_size, args.dropout, args.target_vocab,
                      args.encoding_type)

    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=4e-5)

    # 设置训练器
    trainer = Trainer(train_data=train_data, model=model, optimizer=optimizer,
                      sampler=BucketSampler(num_buckets=3),
                      device=args.device, dev_data=dev_data, batch_size=args.batch_size,
                      metrics=ClassifyFPreRecMetric(tag_vocab=tag_vocab),
                      loss=None, callbacks=callbacks, n_epochs=args.epochs,
                      check_code_level=0, update_every=1, test_use_tqdm=False)
    # 训练
    trainer.train()

    # 测试
    tester = Tester(data=test_data, model=model, metrics=ClassifyFPreRecMetric(tag_vocab=tag_vocab),
                    batch_size=args.batch_size)
    # 测试结果
    res = tester.test()
    print(res)


if __name__ == '__main__':
    # 从命令行传入参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', type=int, default=2)
    parser.add_argument('--device', default='cuda:2', help='cuda/cpu')
    parser.add_argument('--model_path', type=str, default="2021-08-01-22-37-46-408514/epoch-10_step-90_f-0.990385.pt",
                        help='模型路径')
    parser.add_argument('--data_path', type=str, default="new", help='数据路径')
    parser.add_argument('--model_save_path', type=str, default="0731", help='词表路径')
    parser.add_argument('--word_vocab_path', type=str, default="", help='词表路径')
    parser.add_argument('--tag_vocab_path', type=str, default="", help='标签表路径')
    parser.add_argument('--bert_model_dir_or_name', type=str, default='cn-base', help='标签表路径')
    parser.add_argument('--batch_size', type=int, default=2, help='')
    parser.add_argument('--epochs', type=int, default=20, help='')

    args = parser.parse_args()
    main()
