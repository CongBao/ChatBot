""" Utilities """

from __future__ import print_function, division

import numpy as np

__author__ = 'Cong Bao'

def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    return lines

def load_embedding(path, words=None):
    embedding = {}
    if words is not None:
        oov = list(words)
        with open(path) as f:
            for line in f:
                embed = line.split()
                if embed[0] in oov:
                    embedding[embed[0]] = np.asarray(embed[1:], dtype='float32')
                    oov.remove(embed[0])
        return embedding, oov
    else:
        with open(path) as f:
            for line in f:
                embed = line.split()
                embedding[embed[0]] = np.asarray(embed[1:], dtype='float32')
        return embedding

def text_preprocess(text):
    l1 = ['won’t', 'won\'t', 'wouldn’t', 'wouldn\'t', '’m', '’re', '’ve', '’ll', '’s', '’d', 'n’t', '\'m', '\'re',
          '\'ve', '\'ll', '\'s', '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .',
          '. ,', 'EOS', 'BOS', 'eos', 'bos']
    l2 = ['will not', 'will not', 'would not', 'would not', ' am', ' are', ' have', ' will', ' is', ' had', ' not',
          ' am', ' are', ' have', ' will', ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :',
          '? ', '.', ',', '', '', '', '']
    l3 = ['-', '_', ' *', ' /', '* ', '/ ', '\"', ' \\"', '\\ ', '--', '...', '. . .']
    new_text = []
    for line in text:
        line = line.lower()
        for i, term in enumerate(l1):
            line = line.replace(term, l2[i])
        for term in l3:
            line = line.replace(term, ' ')
        for i in range(30):
            line = line.replace('. .', '')
            line = line.replace('.  .', '')
            line = line.replace('..', '')
        for i in range(10):
            line = line.replace('  ', ' ')
        if line[-1] != '!' and line[-1] != '?' and line[-1] != '.':
            line = line + ' .'
        if line[-2:] != '! ' and line[-2:] != '? ' and line[-2:] != '. ':
            line = line + ' .'
        if line == ' !' or line == ' ?' or line == ' .' or line == ' ! ' or line == ' ? ' or line == ' . ':
            line = 'what ?'
        if line == '  .' or line == ' .' or line == '  . ':
            line = 'i do not want to talk about it .'
        new_text.append(line)
    return new_text