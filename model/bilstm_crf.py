# -*- coding: utf-8 -*-
# @Time    : 2021/10/31 15:49 下午
# @Author  : JiangYanQun
"""
本模块包含一些NER常用的模型，如下：
(1)BiLSTM-CRF
"""
import torch.nn as nn
import torch.nn.functional as func
import fastNLP.core.const as C
from fastNLP import seq_len_to_mask
from fastNLP.embeddings.utils import get_embeddings
from fastNLP.modules import LSTM, allowed_transitions, ConditionalRandomField


class BiLstmCrf(nn.Module):
    def __init__(self, char_embed, num_classes, num_layers=1, hidden_size=100, dropout=0.5, target_vocab=None,
                 encoding_type=None):
        super().__init__()

        self.char_embed = get_embeddings(char_embed)
        embed_size = self.char_embed.embedding_dim

        self.lstm = LSTM(embed_size, num_layers=num_layers, hidden_size=hidden_size // 2, bidirectional=True, 
                         batch_first=True, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

        trans = None
        if target_vocab is not None and encoding_type is not None:
            trans = allowed_transitions(target_vocab.idx2word, encoding_type=encoding_type, include_start_end=True)

        self.crf = ConditionalRandomField(num_classes, include_start_end_trans=True, allowed_transitions=trans)

    def _forward(self, chars, seq_len=None, target=None):
        chars = self.char_embed(chars)
        feats, _ = self.lstm(chars, seq_len=seq_len)
        feats = self.fc(feats)
        feats = self.dropout(feats)
        logist = func.log_softmax(feats, dim=-1)
        mask = seq_len_to_mask(seq_len)
        if target is None:
            pred, _ = self.crf.viterbi_decode(logist, mask)
            return {C.OUTPUT: pred}
        else:
            loss = self.crf(logist, target, mask).mean()
            return {C.LOSS: loss}

    def forward(self, chars, target, seq_len=None):
        return self._forward(chars, seq_len, target)

    def predict(self, chars, seq_len=None, bigrams=None, trigrams=None):
        return self._forward(chars,  seq_len)
