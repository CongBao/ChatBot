# -*- coding: utf-8 -*-

""" The model of chat bot """

from __future__ import division, print_function

from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.layers import Activation, Concatenate, Dense, Dot, Embedding, LSTM, Input
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.utils.generic_utils import Progbar

import numpy as np

from .utils import *
from .oov import OOV

__author__ = 'Cong Bao'


class ChatBot(object):

    def __init__(self, **kwargs):
        self.text_dir = kwargs.get('text')
        self.embd_dir = kwargs.get('embd')
        self.ckpt_dir = kwargs.get('ckpt')

        self.dim = kwargs.get('dim', 300)
        self.lr = kwargs.get('lr', 0.01)
        self.bs = kwargs.get('bs', 32)
        self.epoch = kwargs.get('epoch', 100)
        self.tfr = kwargs.get('tfr', 0.7)

    def load_data(self):
        raw_text = preprocess_text(load_text(self.text_dir))
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(['bos ' + line + ' eos' for line in raw_text])
        self.voc_size = len(self.tokenizer.word_index) + 1
        print('Number of tokens: ', self.voc_size)
        self.en_ipt = self.tokenizer.texts_to_sequences(raw_text[0::2]) # [:-1]
        self.de_ipt = self.tokenizer.texts_to_sequences(['bos ' + line for line in raw_text[1::2]]) # [1:]
        self.max_en_seq = max([len(s) for s in self.en_ipt])
        self.max_de_seq = max([len(s) for s in self.de_ipt])
        print('Max input sequence length: ', self.max_en_seq)
        print('Max output sequence length: ', self.max_de_seq)
        self.en_ipt = np.asarray(pad_sequences(self.en_ipt, padding='post'))
        self.de_ipt = np.asarray(pad_sequences(self.de_ipt, padding='post'))
        de_opt_seq = self.tokenizer.texts_to_sequences([line + ' eos' for line in raw_text[1::2]]) # [1:]
        de_opt_seq = pad_sequences(de_opt_seq, padding='post')
        self.de_opt = []
        for seq in de_opt_seq:
            one_hot = to_categorical(seq, num_classes=self.voc_size)
            one_hot[:,0] = 0.
            self.de_opt.append(one_hot)
        self.de_opt = np.asarray(self.de_opt)
        self.embed_dict, oov = load_embedding(self.embd_dir, self.tokenizer.word_index.keys())
        print('Number of OOV words: ', len(oov))
        self.embed_dict.update(OOV(raw_text, self.embed_dict, oov, self.dim).fit())
        save_embedding(self.ckpt_dir + 'embedding.txt', self.embed_dict)
        self.embed_mat = np.zeros((self.voc_size, self.dim))
        self.word2idx = {}
        self.idx2word = [None] * self.voc_size
        for word, idx in self.tokenizer.word_index.items():
            self.embed_mat[idx] = self.embed_dict[word]
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def load_saved_data(self):
        raw_text = preprocess_text(load_text(self.text_dir))
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(['bos ' + line + ' eos' for line in raw_text])
        self.voc_size = len(self.tokenizer.word_index) + 1
        self.max_de_seq = max([len(s.split()) for s in raw_text])
        self.embed_dict = load_embedding(self.ckpt_dir + 'embedding.txt')
        self.embed_mat = np.zeros((self.voc_size, self.dim))
        self.word2idx = {}
        self.idx2word = [None] * self.voc_size
        for word, idx in self.tokenizer.word_index.items():
            self.embed_mat[idx] = self.embed_dict[word]
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def build_model(self, load_weights=False):
        embedding = Embedding(self.voc_size, self.dim, weights=[self.embed_mat], trainable=False, mask_zero=True, name='share_embedding')

        en_input = Input(shape=(None,), name='encoder_input')
        encoder_inputs = embedding(en_input)
        encoder = LSTM(256, return_state=True, return_sequences=True, name='encoder_lstm')
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        de_input = Input(shape=(None,), name='decoder_input')
        decoder_inputs = embedding(de_input)
        decoder = LSTM(256, return_state=True, return_sequences=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=[state_h, state_c])

        attention = Dot([2, 2])([decoder_outputs, encoder_outputs])
        attention = Activation('softmax')(attention)
        context = Dot([2, 1])([attention, encoder_outputs])
        decoder_combined = Concatenate()([context, decoder_outputs])
        decoder_dense = Dense(self.voc_size, activation='softmax')
        outputs = decoder_dense(decoder_combined)

        self.model = Model([en_input, de_input], outputs)

        self.encoder_model = Model(en_input, [encoder_outputs, state_h, state_c])

        de_state_input_h = Input(shape=(256,), name='state_h')
        de_state_input_c = Input(shape=(256,), name='state_c')
        de_state_input = [de_state_input_h, de_state_input_c]
        de_outputs, st_h, st_c = decoder(decoder_inputs, initial_state=de_state_input)
        en_outputs = Input(shape=(None, 256), name='encoder_output')
        atten = Dot([2, 2])([de_outputs, en_outputs])
        atten = Activation('softmax')(atten)
        cotxt = Dot([2, 1])([atten, en_outputs])
        de_comb = Concatenate()([cotxt, de_outputs])
        out = decoder_dense(de_comb)
        self.decoder_model = Model([en_outputs, de_input] + de_state_input, [out, st_h, st_c])
        if load_weights:
            self.model.load_weights(self.ckpt_dir + 'weights.hdf5')
        else:
            self.model.compile(optimizer='adam', loss='categorical_crossentropy')

    def train_model(self):
        cbs = []
        cbs.append(EarlyStopping(patience=2))
        cbs.append(LearningRateScheduler(lambda e: self.lr * 0.999 ** (e / 10)))
        cb = CallBacks(cbs)
        cb.set_model(self.model)

        print('Start training chatbot...')
        train_num = len(self.en_ipt)
        cb.on_train_begin()
        for itr in range(self.epoch):
            print('Epoch %s/%s' % (itr + 1, self.epoch))
            cb.on_epoch_begin(itr)
            indexes = np.random.permutation(train_num)
            progbar = Progbar(train_num)
            losses = []
            for idx in range(int(0.8 * train_num / self.bs)):
                batch_idx = indexes[idx * self.bs : (idx + 1) * self.bs]
                en_ipt_bc = self.en_ipt[batch_idx]
                de_ipt_bc = self.de_ipt[batch_idx]
                de_opt_bc = self.de_opt[batch_idx]
                if np.random.rand() < self.tfr:
                    bc_loss = self.model.train_on_batch([en_ipt_bc, de_ipt_bc], de_opt_bc)
                else:
                    ipt_len = [sum(i) for i in np.any(de_opt_bc, axis=-1)]
                    de_ipt_nt = np.zeros((self.max_de_seq, self.bs), dtype='int64')
                    en_out, h, c = self.encoder_model.predict(en_ipt_bc, batch_size=self.bs)
                    de_in = np.asarray([[self.word2idx['bos']]] * self.bs)
                    for i in range(self.max_de_seq):
                        de_out, h, c = self.decoder_model.predict([en_out, de_in, h, c], batch_size=self.bs)
                        sampled_idxs = np.argmax(de_out[:, -1, :], axis=-1)
                        de_ipt_nt[i] = sampled_idxs
                        de_in = sampled_idxs.reshape((-1, 1))
                    de_ipt_nt = de_ipt_nt.T
                    for i in range(self.bs):
                        de_ipt_nt[i, ipt_len[i]:] = 0
                    bc_loss = self.model.train_on_batch([en_ipt_bc, de_ipt_nt], de_opt_bc)
                losses.append(bc_loss)
                progbar.add(self.bs, [('loss', np.mean(losses))])
            val_idx = indexes[-int(0.2 * train_num):]
            val_loss = self.model.evaluate([self.en_ipt[val_idx], self.de_ipt[val_idx]], self.de_opt[val_idx], batch_size=self.bs, verbose=0)
            progbar.update(train_num, [('val_loss', np.mean(val_loss))])
            cb.on_epoch_end(itr, logs={'loss': np.mean(losses), 'val_loss': np.mean(val_loss)})
            self.model.save_weights(self.ckpt_dir + 'weights.hdf5')
        cb.on_train_end()
        print('Chatbot training complete.')

    def dialogue(self, input_text, mode='beam', k=5):
        input_text = preprocess_line(input_text)
        input_seq = self.tokenizer.texts_to_sequences([input_text])
        en_outputs, st_h, st_c = self.encoder_model.predict(input_seq)
        en_outputs = np.reshape(en_outputs, (1, -1, 256))
        states = [st_h, st_c]
        target_seq = np.asarray([[self.word2idx['bos']]])
        answer = ''
        if mode == 'beam':
            answers = []
            output_tokens, h, c = self.decoder_model.predict([en_outputs, target_seq] + states)
            top_k_idx = np.argpartition(output_tokens[0, -1, :], -k)[-k:]
            for idx in top_k_idx:
                answers.append([(idx, output_tokens[0, -1, idx], [h, c])])
            count = 0
            while True:
                count += 1
                for _ in range(k):
                    seq = answers.pop(0)
                    if self.idx2word[seq[-1][0]] == 'eos':
                        answers.append(seq)
                        continue
                    target_seq = np.asarray([[seq[-1][0]]])
                    output_tokens, h, c = self.decoder_model.predict([en_outputs, target_seq] + seq[-1][-1])
                    top_k_idx = np.argpartition(output_tokens[0, -1, :], -k)[-k:]
                    for idx in top_k_idx:
                        answers.append(seq + [(idx, output_tokens[0, -1, idx], list([h, c]))])
                dead_list = [sum([x[1] for x in seq]) for seq in answers] # seq[:, 1]
                top_k_idx = np.argpartition(dead_list, -k)[-k:]
                answers = [answers[i] for i in top_k_idx] # answers[top_k_idx]
                if all([self.idx2word[s[-1][0]] == 'eos' or len(s) > self.max_de_seq for s in answers]):
                    break
            dead_list = [sum([x[1] for x in seq]) for seq in answers]
            best_answer = answers[np.argmax(dead_list)]
            for token in best_answer[:-1]:
                answer += self.idx2word[token[0]] + ' '
        elif mode == 'greedy':
            while True:
                output_tokens, h, c = self.decoder_model.predict([en_outputs, target_seq] + states)
                sampled_token_idx = np.argmax(output_tokens[0, -1, :])
                sampled_word = self.idx2word[sampled_token_idx]
                if sampled_word == 'eos' or len(answer) > self.max_de_seq:
                    break
                answer += sampled_word + ' '
                target_seq = np.asarray([[sampled_token_idx]])
                states = [h, c]
        return answer.strip().capitalize() + '.'


class CallBacks(object):

    def __init__(self, cb_list):
        self.cb_list = cb_list

    def set_model(self, model):
        for cb in self.cb_list:
            cb.set_model(model)

    def on_train_begin(self, logs=None):
        for cb in self.cb_list:
            cb.on_train_begin(logs)

    def on_epoch_begin(self, epoch, logs=None):
        for cb in self.cb_list:
            cb.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        for cb in self.cb_list:
            cb.on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        for cb in self.cb_list:
            cb.on_train_end(logs)
