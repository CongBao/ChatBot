""" The model of chat bot """

from __future__ import division, print_function

from keras.models import Model
from keras.layers import Input, LSTM, CuDNNLSTM, Dense
import numpy as np

__author__ = 'Cong Bao'

class ChatBot(object):

    def __init__(self, **kwargs):
        self.text_dir = kwargs.get('text_dir')
        self.ckpt_dir = kwargs.get('ckpt_dir')

        self.dim = kwargs.get('dim')
        self.lr = kwargs.get('lr')
        self.bs = kwargs.get('bs')
        self.epoch = kwargs.get('epoch')

        self.cpu = kwargs.get('cpu')

        self.model = None

    def load_data(self):
        pass

    def build_model(self):
        lstm = LSTM if self.cpu else CuDNNLSTM

        encoder_inputs = Input(shape=(None, self.dim))
        encoder = lstm(256, return_state=True)
        _, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, self.dim))
        decoder = lstm(256, return_state=True, return_sequences=True)
        decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)
        outputs = Dense(self.dim, activation='softmax')(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], outputs)

    def train_model(self):
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')