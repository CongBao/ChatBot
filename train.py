# -*- coding: utf-8 -*-

# Launch the training process of model

from __future__ import print_function

import argparse
import os

from chatbot.utils import Corrector

__author__ = 'Cong Bao'

CHECKPOINT = os.getcwd() + '/checkpoint/'
DIMENSION = 300
LEARNING_RATE = 0.01
BATCH_SIZE = 32
EPOCH = 100
TF_RATIO = 0.7


def main():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--text', dest='text', type=str, default=None, help='Path of dialog text')
    add_arg('--embed', dest='embed', type=str, default=None, help='Path of pre-trained word embedding')
    add_arg('--ckpt', dest='ckpt', type=str, default=CHECKPOINT, help='Path to store checkpoints')
    add_arg('--tfr', dest='tfr', type=float, default=TF_RATIO, help='Ratio of teacher forcing learning')
    add_arg('-d', dest='dim', type=int, default=DIMENSION, help='Dimensionality of word embedding')
    add_arg('-r', dest='rate', type=float, default=LEARNING_RATE, help='Learning rate')
    add_arg('-b', dest='batch', type=int, default=BATCH_SIZE, help='Batch size')
    add_arg('-e', dest='epoch', type=int, default=EPOCH, help='Epoch number')
    add_arg('--cpu-only', dest='cpu', action='store_true', help='whether use cpu or not')
    args = parser.parse_args()
    corr = Corrector().correct
    params = {
        'text': args.text,
        'embd': args.embed,
        'ckpt': corr(args.ckpt),
        'dim': args.dim,
        'lr': args.rate,
        'bs': args.batch,
        'epoch': args.epoch,
        'tfr': args.tfr
    }
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if not os.path.exists(params['ckpt']):
        os.makedirs(params['ckpt'])
    from chatbot.model import ChatBot
    cb = ChatBot(**params)
    print('Dialog text path: ', params['text'])
    print('Word embedding path: ', params['embd'])
    print('Checkpoint directory: ', params['ckpt'])
    print('Word embedding dimensionality: ', params['dim'])
    print('Learning rate: ', params['lr'])
    print('Batch size', params['bs'])
    print('Epoch: ', params['epoch'])
    print('Teacher forcing ratio: ', params['tfr'])
    print('Running on %s' % ('CPU' if args.cpu else 'GPU'))
    cb.load_data()
    cb.build_model()
    try:
        cb.train_model()
        print('Training complete')
    except (KeyboardInterrupt, SystemExit):
        print('Abort!')

if __name__ == '__main__':
    main()