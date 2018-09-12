# -*- coding: utf-8 -*-

""" Utilities """

from __future__ import print_function, division

import re

import numpy as np
from tqdm import tqdm

__author__ = 'Cong Bao'

BAR_FMT = 'Progress: {percentage:3.0f}% {r_bar}'


def load_text(path):
    """
    Load raw texts from disk.
    :param path: path of text
    :return: list of line strings
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    return lines


def load_embedding(path, words=None):
    """
    Load word embeddings from disk.
    :param path: path of embedding file
    :param words: a list of selected words
    :return: (embedding dict, dimensionality, oov list) if words are given,
             (embedding dict, dimensionality) if words are not given
    """
    embedding = {}
    dim = 0
    if words is not None:
        oov = list(words)
        with open(path) as f:
            print('Loading ', path)
            for line in tqdm(f, bar_format=BAR_FMT):
                embed = line.split()
                word = embed[0].lower()
                dim = len(embed[1:])
                if word in oov:
                    embedding[word] = np.asarray(embed[1:], dtype='float32')
                    oov.remove(word)
        return embedding, dim, oov
    else:
        with open(path) as f:
            print('Loading ', path)
            for line in tqdm(f, bar_format=BAR_FMT):
                embed = line.split()
                dim = len(embed[1:])
                embedding[embed[0]] = np.asarray(embed[1:], dtype='float32')
        return embedding, dim


def save_embedding(path, embed):
    """
    Save word embeddings to disk.
    :param path: path to save embeddings
    :param embed: dict of word embeddings
    """
    embed_list = []
    for w, e in embed.items():
        embed_list.append(w + ' ' + ' '.join(str(i) for i in e))
    with open(path, 'w') as f:
        f.writelines('%s\n' % line for line in embed_list)


def preprocess_text(text):
    return [preprocess_line(line) for line in text]


def preprocess_line(line):
    c1 = ['won’t', 'won\'t', 'wouldn’t', 'wouldn\'t', '’m', '’re', '’ve', '’ll', '’s', '’d', 'n’t', '\'m', '\'re',
          '\'ve', '\'ll', '\'s', '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .',
          '. ,', 'EOS', 'BOS', 'eos', 'bos']
    c2 = ['will not', 'will not', 'would not', 'would not', ' am', ' are', ' have', ' will', ' is', ' had', ' not',
          ' am', ' are', ' have', ' will', ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :',
          '? ', '.', ',', '', '', '', '']
    c3 = ['-', '_', ' *', ' /', '* ', '/ ', '\'', ' \'', '\' ', '--', '...', '. . .']
    line = line.strip().lower()
    for i, term in enumerate(c1):
        line = line.replace(term, c2[i])
    for term in c3:
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
    return line


class Corrector(object):
    """ The class used to correct user inputs """

    def __init__(self):
        self.back_slash = re.compile(r'\\+', re.IGNORECASE)
        self.end_slash = re.compile(r'^.*[^/]$', re.IGNORECASE)

    def replace_backslash(self, line):
        """ replace backslashes to slashes
            :param line: the input line
            :return: string after correction
        """
        return self.back_slash.sub('/', line)

    def add_endslash(self, line):
        """ add a slash in the end of line
            :param line: the input line
            :return: string after correction
        """
        if self.end_slash.match(line):
            return line + '/'
        return line

    def correct(self, line):
        """ do all corrections
            :param line: the input line
            :return: string after correction
        """
        line = self.replace_backslash(line)
        line = self.add_endslash(line)
        return line