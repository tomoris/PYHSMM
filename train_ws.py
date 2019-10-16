#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ctypes
import argparse
import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)


class GoString(ctypes.Structure):
    _fields_ = [("p", ctypes.c_char_p), ("n", ctypes.c_longlong)]


lib = ctypes.cdll.LoadLibrary("./trainWordSegmentation.so")

lib.trainWordSegmentation.argtypes = [
    GoString,
    GoString,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_longlong,
    ctypes.c_longlong,
    ctypes.c_longlong,
    ctypes.c_double,
    ctypes.c_longlong,
    ctypes.c_longlong,
    ctypes.c_longlong]


def train_word_segmentation(modelForWS, trainFilePathForWS,
                            initialTheta, initialD=0.1,
                            gammaA=1.0, gammaB=1.0, betaA=1.0,
                            betaB=1.0, alpha=1.0, beta=1.0, maxNgram=2,
                            maxWordLength=10, posSize=10,
                            base=(1.0 / 2097152.0),
                            epoch=10,
                            threads=8,
                            batch=128):

    modelForWS_bytes = modelForWS.encode(encoding='ascii')
    trainFilePathForWS_bytes = trainFilePathForWS.encode(encoding='ascii')
    lib.trainWordSegmentation(GoString(modelForWS_bytes,
                                       len(modelForWS_bytes)),
                              GoString(trainFilePathForWS_bytes,
                                       len(trainFilePathForWS_bytes)),
                              initialTheta, initialD,
                              gammaA, gammaB, betaA,
                              betaB, alpha, beta, maxNgram,
                              maxWordLength, posSize,
                              base, epoch, threads, batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='unsupervised word segmentation')
    parser.add_argument(
        '--model',
        help='unsupervised word segmentation model',
        type=str,
        choices=['npylm', 'pyhsmm'])
    parser.add_argument(
        '--train_file',
        help='training file path',
        type=str,
        default=None)
    parser.add_argument(
        '--maxNgram',
        help='hyper-parameter in HPYLM - PYHSMM',
        type=int,
        default=2)
    parser.add_argument(
        '--theta',
        help='initial hyper-parameter in HPYLM - PYHSMM',
        type=float,
        default=2.0)
    parser.add_argument(
        '--d',
        help='initial hyper-parameter in HPYLM - PYHSMM',
        type=float,
        default=0.1)
    parser.add_argument(
        '--gammaA',
        help='hyper-parameter in HPYLM - PYHSMM',
        type=float,
        default=1.0)
    parser.add_argument(
        '--gammaB',
        help='hyper-parameter in HPYLM - PYHSMM',
        type=float,
        default=1.0)
    parser.add_argument(
        '--betaA',
        help='hyper-parameter in HPYLM - PYHSMM',
        type=float,
        default=1.0)
    parser.add_argument(
        '--betaB',
        help='hyper-parameter in HPYLM - PYHSMM',
        type=float,
        default=1.0)
    parser.add_argument(
        '--vocabSize',
        help='hyper-parameter in HPYLM - PYHSMM \
            (default parameter is size of charcter vocab. (utf-8))',
        type=float,
        default=2097152.0)
    parser.add_argument(
        '--alpha',
        help='hyper-parameter in VPYLM - PYHSMM \
        (used sampling depth in VPYLM)',
        type=float,
        default=1.0)
    parser.add_argument(
        '--beta',
        help='hyper-parameter in VPYLM - PYHSMM \
        (used sampling depth in VPYLM)',
        type=float,
        default=1.0)
    parser.add_argument(
        '--maxWordLength',
        help='hyper-parameter in NPYLM - PYHSMM',
        type=int,
        default=10)
    parser.add_argument(
        '--posSize',
        help='hyper-parameter in PYHSMM',
        type=int,
        default=10)
    parser.add_argument(
        '--epoch',
        help='hyper-parameter in HPYLM - PYHSMM',
        type=int,
        default=10)
    parser.add_argument(
        '--batch',
        help='hyper-parameter in NPYLM - PYHSMM',
        type=int,
        default=128)
    parser.add_argument(
        '--threads',
        help='hyper-parameter in NPYLM - PYHSMM',
        type=int,
        default=8)
    args = parser.parse_args()

    train_word_segmentation(args.model, args.train_file,
                            args.theta, args.d,
                            args.gammaA, args.gammaB, args.betaA,
                            args.betaB, args.alpha, args.beta,
                            args.maxNgram, args.maxWordLength,
                            args.posSize,
                            (1.0 / args.vocabSize), args.epoch,
                            args.threads, args.batch)
