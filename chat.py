# -*- coding: utf-8 -*-

# Load pre-trained model and chat with ChatBot

from __future__ import print_function

import argparse
import os

from chatbot.utils import Corrector

__author__ = 'Cong Bao'

CHECKPOINT = os.getcwd() + '/checkpoint/'
DIMENSION = 300
MODE = 'beam'
K = 5

def main():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--text', dest='text', type=str, required=True, help='Path of dialog text')
    add_arg('--ckpt', dest='ckpt', type=str, default=CHECKPOINT, help='Path to store checkpoints')
    add_arg('-d', dest='dim', type=int, default=DIMENSION, help='Dimensionality of word embedding')
    add_arg('-m', dest='mode', type=str, default=MODE, help='The mode used to decode, either beam or greedy')
    add_arg('-k', dest='k', type=int, default=K, help='Beam search size if in beam mode')
    add_arg('--cpu-only', dest='cpu', action='store_true', help='whether use cpu or not')
    args = parser.parse_args()
    corr = Corrector().correct
    params = {
        'text': args.text,
        'ckpt': corr(args.ckpt),
        'dim': args.dim
    }
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from chatbot.model import ChatBot
    cb = ChatBot(**params)
    print('Dialog text path: ', params['text'])
    print('Checkpoint directory: ', params['ckpt'])
    print('Word embedding dimensionality: ', params['dim'])
    print('Running on %s' % ('CPU' if args.cpu else 'GPU'))
    cb.load_saved_data()
    cb.build_model(load_weights=True)
    print('[Start conversation. Type \'exit\' to stop.]')
    seq = ''
    while True:
        seq = input('[User]: ')
        seq = seq.strip().lower()
        if seq == 'exit':
            break
        print('[Comp]:', cb.dialogue(seq, mode=args.mode, k=args.k))
    print('[Comp]: Bye!')

if __name__ == '__main__':
    main()