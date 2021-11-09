# -*- coding: utf-8 -*-
# @Time    : 2021/10/26 3:05 下午
# @Author  : JiangYanQun
"""
将brat标注的ann文件转成conll格式
"""

from __future__ import print_function

import os
import re
import sys
import argparse
from collections import namedtuple
from io import StringIO
from os import path

options = None

EMPTY_LINE_RE = re.compile(r'^\s*$')
CONLL_LINE_RE = re.compile(r'^\S+\t\d+\t\d+.')


class FormatError(Exception):
    pass


def read_sentence(f):
    """
    读取文件句子，返回一个句子列表
    :param f: 读出的文件内容
    :return: 句子列表
    """
    lines = []
    for l in f:
        lines.append(l)
        if EMPTY_LINE_RE.match(l):  # 条过空格开头的句子
            break
        if not CONLL_LINE_RE.search(l):  # 不满足格式的句子报错
            raise FormatError(
                'Line not in CoNLL format: "%s"' %
                l.rstrip('\n'))
    return lines


def strip_labels(lines):
    """Given CoNLL-format lines, strip the label (first TAB-separated field)
    from each non-empty line.
    Return list of labels and list of lines without labels. Returned
    list of labels contains None for each empty line in the input.
    """

    labels, stripped = [], []

    labels = []
    for l in lines:
        if EMPTY_LINE_RE.match(l):
            labels.append(None)
            stripped.append(l)
        else:
            fields = l.split('\t')
            labels.append(fields[0])
            stripped.append('\t'.join(fields[1:]))

    return labels, stripped


def attach_labels(labels, lines):
    """Given a list of labels and CoNLL-format lines, affix TAB-separated label
    to each non-empty line.
    Returns list of lines with attached labels.
    """

    assert len(labels) == len(
        lines), "Number of labels (%d) does not match number of lines (%d)" % (len(labels), len(lines))

    attached = []
    for label, line in zip(labels, lines):
        empty = EMPTY_LINE_RE.match(line)
        assert (label is None and empty) or (label is not None and not empty)

        if empty:
            attached.append(line)
        else:
            attached.append('%s\t%s' % (label, line))

    return attached


# NERsuite tokenization: any alnum sequence is preserved as a single
# token, while any non-alnum character is separated into a
# single-character token. TODO: non-ASCII alnum.
TOKENIZATION_REGEX = re.compile(r'([0-9a-zA-Z]+|[^0-9a-zA-Z])')

NEWLINE_TERM_REGEX = re.compile(r'(.*?\n)')


def text_to_conll(f):
    """Convert plain text into CoNLL format."""
    global options

    if options.nosplit:
        sentences = f.readlines()
    else:
        sentences = []
        for l in f:
            l = sentencebreaks_to_newlines(l)
            sentences.extend([s for s in NEWLINE_TERM_REGEX.split(l) if s])

    lines = []

    offset = 0
    for s in sentences:
        nonspace_token_seen = False

        tokens = [t for t in TOKENIZATION_REGEX.split(s) if t]

        for t in tokens:
            if not t.isspace():
                lines.append(['O', offset, offset + len(t), t])
                nonspace_token_seen = True
            offset += len(t)

        # sentences delimited by empty lines
        if nonspace_token_seen:
            lines.append([])

    # add labels (other than 'O') from standoff annotation if specified
    if options.annsuffix:
        lines = relabel(lines, get_annotations(f.name))

    lines = [[l[0], str(l[1]), str(l[2]), l[3]] if l else l for l in lines]
    return StringIO('\n'.join(('\t'.join(l) for l in lines)))


def relabel(lines, annotations):
    global options

    # TODO: this could be done more neatly/efficiently
    offset_label = {}

    for tb in annotations:
        for i in range(tb.start, tb.end):
            if i in offset_label:
                print("Warning: overlapping annotations", file=sys.stderr)
            offset_label[i] = tb

    prev_label = None
    for i, l in enumerate(lines):
        if not l:
            prev_label = None
            continue
        tag, start, end, token = l

        # TODO: warn for multiple, detailed info for non-initial
        label = None
        for o in range(start, end):
            if o in offset_label:
                if o != start:
                    print('Warning: annotation-token boundary mismatch: "%s" --- "%s"' % (
                        token, offset_label[o].text), file=sys.stderr)
                label = offset_label[o].type
                break

        if label is not None:
            if label == prev_label:
                tag = 'I-' + label
            else:
                tag = 'B-' + label
        prev_label = label

        lines[i] = [tag, start, end, token]

    # optional single-classing
    if options.singleclass:
        for l in lines:
            if l and l[0] != 'O':
                l[0] = l[0][:2] + options.singleclass

    return lines


def process(f):
    return text_to_conll(f)


def process_files(options):
    files = options.text

    ner_suite_proc = []

    try:
        for fn in files:
            try:
                if fn == '-':
                    lines = process(sys.stdin)
                else:
                    with open(fn, 'rU') as f:
                        lines = process(f)

                # TODO: better error handling
                if lines is None:
                    raise FormatError

                if fn == '-' or not options.outsuffix:
                    sys.stdout.write(''.join(lines))
                else:
                    ofn = path.splitext(fn)[0] + options.outsuffix
                    with open(ofn, 'wt') as of:
                        of.write(''.join(lines))

            except BaseException:
                # TODO: error processing
                raise
    except Exception as e:
        for p in nersuite_proc:
            p.kill()
        if not isinstance(e, FormatError):
            raise

# start standoff processing


TEXTBOUND_LINE_RE = re.compile(r'^T\d+\t')

Textbound = namedtuple('Textbound', 'start end type text')


def parse_textbounds(f):
    """Parse textbound annotations in input, returning a list of Textbound."""

    textbounds = []

    for l in f:
        l = l.rstrip('\n')

        if not TEXTBOUND_LINE_RE.search(l):
            continue

        id_, type_offsets, text = l.split('\t')
        type_, start, end = type_offsets.split()
        start, end = int(start), int(end)

        textbounds.append(Textbound(start, end, type_, text))

    return textbounds


def eliminate_overlaps(textbounds):
    eliminate = {}

    # TODO: avoid O(n^2) overlap check
    for t1 in textbounds:
        for t2 in textbounds:
            if t1 is t2:
                continue
            if t2.start >= t1.end or t2.end <= t1.start:
                continue
            # eliminate shorter
            if t1.end - t1.start > t2.end - t2.start:
                print("Eliminate %s due to overlap with %s" % (
                    t2, t1), file=sys.stderr)
                eliminate[t2] = True
            else:
                print("Eliminate %s due to overlap with %s" % (
                    t1, t2), file=sys.stderr)
                eliminate[t1] = True

    return [t for t in textbounds if t not in eliminate]


def get_annotations(fn):
    global options

    annfn = path.splitext(fn)[0] + options.annsuffix

    with open(annfn, 'rU') as f:
        textbounds = parse_textbounds(f)

    textbounds = eliminate_overlaps(textbounds)

    return textbounds

# end standoff processing


def main(argv=None):
    if argv is None:
        argv = sys.argv

    options = arg_parser().parse_args(argv[1:])

    # 确保输出文件格式包含.，如果不包含则加上
    if options.outsuffix and options.outsuffix[0] != '.':
        options.outsuffix = '.' + options.outsuffix
    if options.annsuffix and options.annsuffix[0] != '.':
        options.annsuffix = '.' + options.annsuffix

    process_files(options)


def arg_parser():
    ap = argparse.ArgumentParser(description='将brat标注的ann文件转成conll格式 ')
    ap.add_argument('-a', '--ann后缀', default="ann",
                    help='Standoff annotation file suffix (默认 "ann")')
    ap.add_argument('-c', '--single_class', default=None,
                    help='Use given single class for annotations')
    ap.add_argument('-n', '--no_split', default=False, action='store_true',
                    help='不进行句子切分')
    ap.add_argument('-o', '--out_suffix', default="conll",
                    help='输出文件后缀 (default "conll")')
    ap.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='Verbose output')
    ap.add_argument('text', metavar='TEXT', nargs='+',
                    help='Text files ("-" for STDIN)')
    return ap


if __name__ == "__main__":
    sys.exit(main(sys.argv))