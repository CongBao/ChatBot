# -*- coding: utf-8 -*-

""" Train word embedding of OOV words """

from __future__ import print_function, division

from keras.layers import Dense, Dot, Embedding, Input, Reshape
from keras.models import Model
from keras.preprocessing.text import Tokenizer

import numpy as np

__author__ = 'Cong Bao'

class OOV(object):

    def __init__(self, raw_text, emb_dict, oov_list, dim, **kwargs):
        self.raw_text = raw_text
        self.emb_dict = emb_dict
        self.oov_list = oov_list
        self.dim = dim

        self.win_size = kwargs.get('win_size', 4)
        self.batch_size = kwargs.get('batch_size', 1024)
        self.epochs = kwargs.get('epoch', 2)

    def build(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(['bos ' + line + ' eos' for line in self.raw_text])
        self.voc_size = len(self.tokenizer.word_index) + 1
        embed_mat = np.zeros((self.voc_size, self.dim))
        for word, idx in self.tokenizer.word_index.items():
            embed_mat[idx] = self.emb_dict.get(word, [0.] * self.dim)

        target = Input(shape=(1,), name='target')
        tar_embed = Embedding(self.voc_size, self.dim, input_length=1, name='target_embedding')(target)
        context = Input(shape=(1,), name='context')
        ctx_embed = Embedding(self.voc_size, self.dim, input_length=1, weights=[embed_mat], trainable=False, name='context_embedding')(context)

        dot_product = Dot([2, 2])([tar_embed, ctx_embed])
        dot_product = Reshape((1,))(dot_product)
        output = Dense(1, activation='sigmoid')(dot_product)
        self.model = Model([target, context], output)

        self.embed_model = Model(target, tar_embed)

    def fit(self):
        self.build()
        if len(self.oov_list) == 0:
            return {}
        ext_dict = {}
        for w in ['bos', 'eos', 'BOS', 'EOS']:
            if w in self.oov_list:
                ext_dict[w] = np.random.normal(0., .0001, (self.dim,))
        pairs = []
        for seq in self.tokenizer.texts_to_sequences(self.raw_text):
            words = [token for token in seq]
            for target in self.tokenizer.texts_to_sequences(self.oov_list):
                target = target[0]
                if target in words:
                    idx = words.index(target)
                    pos = words[max(0, idx - self.win_size):idx] + words[idx + 1:min(idx + self.win_size + 1, len(words))]
                    for w in pos:
                        pairs.append([target, w, 1.])
                    neg = set(words) - set(pos + [target])
                    for w in neg:
                        pairs.append([target, w, 0.])
                else:
                    for w in words:
                        pairs.append([target, w, 0.])
        tar_ipt, ctx_ipt, outputs = zip(*pairs)
        tar_ipt = np.reshape(tar_ipt, (-1, 1))
        ctx_ipt = np.reshape(ctx_ipt, (-1, 1))
        outputs = np.reshape(outputs, (-1, 1))
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop')
        print('Learning OOV word embeddings...')
        self.model.fit([tar_ipt, ctx_ipt], outputs, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.1)
        print('OOV word embeddings learning complete.')
        oov_dict = {}
        for w in self.tokenizer.texts_to_sequences(self.oov_list):
            embed = self.embed_model.predict(w)
            oov_dict[self.tokenizer.index_word[w[0]]] = np.reshape(embed, (self.dim,))
            oov_dict.update(ext_dict)
        return oov_dict
