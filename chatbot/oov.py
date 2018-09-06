""" Train word embedding of OOV words """

from __future__ import print_function, division

from keras.models import Model
from keras.layers import Input, Dot, Dense, Reshape
import keras.backend as K
import numpy as np

__author__ = 'Cong Bao'

class OOV(object):

    def __init__(self, raw_text, emb_dict, oov_list, dim, **kwargs):
        self.raw_text = raw_text
        self.emb_dict = emb_dict
        self.oov_list = oov_list
        self.dim = dim

        self.win_size = kwargs.get('win_size', 4)
        self.batch_size = kwargs.get('batch_size', 64)
        self.epochs = kwargs.get('epoch', 100)

        self.model = None

    def build(self):
        target = Input(shape=(self.dim, 1))
        context = Input(shape=(self.dim, 1))
        dot_product = Dot(1)([target, context])
        dot_product = Reshape((1,))(dot_product)
        output = Dense(1, activation='sigmoid')(dot_product)
        self.model = Model([target, context], output)

    def fit(self):
        if len(self.oov_list) == 0:
            return {}
        ext_dict = {}
        for w in ['bos', 'eos', 'BOS', 'EOS']:
            if w in self.oov_list:
                ext_dict[w] = np.random.normal(0., .0001, (self.dim,))
        var_dict = {}
        for w in self.oov_list:
            var_dict[w] = K.random_normal_variable((self.dim, 1), 0., .01, dtype='float32', name=w)
        self.emb_dict.update(var_dict)
        del var_dict
        pairs = []
        for seq in self.raw_text:
            words = [w.strip().lower() for w in seq.split()]
            for target in self.oov_list:
                if target in words:
                    idx = words.index(target)
                    pos = words[max(0, idx - self.win_size):idx] + words[idx + 1:min(idx + self.win_size + 1, len(words))]
                    for w in pos:
                        pairs.append(([self.emb_dict[target], self.emb_dict[w]], 1.))
                    neg = set(words) - set(pos + [target])
                    for w in neg:
                        pairs.append(([self.emb_dict[target], self.emb_dict[w]], 0.))
                else:
                    for w in words:
                        pairs.append(([self.emb_dict[target], self.emb_dict[w]], 0.))
        inputs, outputs = zip(*pairs)
        self.build()
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop')
        self.model.fit(inputs, outputs, batch_size=self.batch_size, epochs=self.epochs)
        oov_dict = {}
        for w in self.oov_list:
            oov_dict[w] = np.reshape(K.get_value(w), (self.dim,))
        return oov_dict.update(ext_dict)
