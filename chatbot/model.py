""" The model of chat bot """

from __future__ import division, print_function

from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense, Activation, Dot, Concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.generic_utils import Progbar
from keras.utils import to_categorical
import keras.backend as K
import numpy as np

from .utils import *
from .oov import OOV

__author__ = 'Cong Bao'

class ChatBot(object):

    def __init__(self, **kwargs):
        self.text_dir = kwargs.get('text_dir')
        self.embd_dir = kwargs.get('embd_dir')
        self.ckpt_dir = kwargs.get('ckpt_dir')

        self.dim = kwargs.get('dim')
        self.lr = kwargs.get('lr')
        self.bs = kwargs.get('bs')
        self.epoch = kwargs.get('epoch')

    def load_data(self):
        raw_text = text_preprocess(load_text(self.text_dir))
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(['bos ' + line + ' eos' for line in raw_text])
        self.voc_size = len(self.tokenizer.word_index) + 1
        print('Number of tokens: ', self.voc_size)
        self.en_ipt = self.tokenizer.texts_to_sequences(raw_text[:-1])
        self.de_ipt = self.tokenizer.texts_to_sequences(['bos ' + line for line in raw_text[1:]])
        self.max_en_seq = max([len(s) for s in self.en_ipt])
        self.max_de_seq = max([len(s) for s in self.de_ipt])
        print('Max input sequence length: ', self.max_en_seq)
        print('Max output sequence length: ', self.max_de_seq)
        self.en_ipt = np.asarray(pad_sequences(self.en_ipt, padding='post'))
        self.de_ipt = np.asarray(pad_sequences(self.de_ipt, padding='post'))
        de_opt_seq = self.tokenizer.texts_to_sequences([line + ' eos' for line in raw_text[1:]])
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
        self.embed_mat = np.zeros((self.voc_size, self.dim))
        self.word2idx = {}
        self.idx2word = [None] * self.voc_size
        for word, idx in self.tokenizer.word_index.items():
            self.embed_mat[idx] = self.embed_dict[word]
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def build_model(self):
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

    def train_model(self):
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        self.model.fit([self.en_ipt, self.de_ipt], self.de_opt, batch_size=self.bs, epochs=self.epoch, validation_split=.1)

    def dialogue(self, input_text, mode='beam', k=5):
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
        return answer.strip().capitalize()
