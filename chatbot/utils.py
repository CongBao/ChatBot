""" Utilities """

from __future__ import print_function, division

import re
import numpy as np
from tqdm import tqdm

__author__ = 'Cong Bao'

BAR_FMT = 'Progress: {percentage:3.0f}% {r_bar}'


def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    return lines


def load_embedding(path, words=None):
    embedding = {}
    if words is not None:
        oov = list(words)
        with open(path) as f:
            print('Loading ', path)
            for line in tqdm(f, bar_format=BAR_FMT):
                embed = line.split()
                word = embed[0].lower()
                if word in oov:
                    embedding[word] = np.asarray(embed[1:], dtype='float32')
                    oov.remove(word)
        return embedding, oov
    else:
        with open(path) as f:
            print('Loading ', path)
            for line in tqdm(f, bar_format=BAR_FMT):
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
    l3 = ['-', '_', ' *', ' /', '* ', '/ ', '\'', ' \'', '\' ', '--', '...', '. . .']
    new_text = []
    for line in text:
        line = line.strip().lower()
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