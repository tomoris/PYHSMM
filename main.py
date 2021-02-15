#!/usr/bin/env python3


# -*- coding: utf-8 -*-


import argparse
import signal

# from pylib import go
from pylib import bayselm

signal.signal(signal.SIGINT, signal.SIG_DFL)


def train_word_segmentation(modelName, trainFilePathForWS, initialTheta=2.0, initialD=0.1, gammaA=1.0, gammaB=1.0,
                            betaA=1.0, betaB=1.0, alpha=1.0, beta=1.0, maxNgram=2, maxWordLength=10,
                            posSize=10, base=(1.0 / 2097152.0), epoch=10, threads=8, batch=128,
                            saveFile='', saveFormat='notindent'):
    if modelName == 'npylm':
        model = bayselm.NewNPYLM(initialTheta, initialD, gammaA, gammaB, betaA, betaB, alpha, beta,
                                 maxNgram, maxWordLength)
    elif modelName == 'pyhsmm':
        model = bayselm.NewPYHSMM(initialTheta, initialD, gammaA, gammaB, betaA, betaB, alpha, beta,
                                  maxNgram, maxWordLength, posSize)
    data_container = bayselm.NewDataContainer(trainFilePathForWS)
    # data_container = bayselm.NewDataContainerFromSents(go.Slice_string(['aiueo, abcdefg']))
    model.Initialize(data_container)
    for e in range(epoch):
        model.TrainWordSegmentation(data_container, threads, batch)
        # model.TrainWordSegmentationAndPOSTagging(data_container, threads, batch)
    test_data_container = model.TestWordSegmentationForPython(data_container.Sents, threads)
    for i in range(test_data_container.Size):
        word_seq = test_data_container.GetWordSeq(i)
        print([_ for _ in word_seq])
    if saveFile != '':
        bayselm.Save(model, saveFile, saveFormat)
        # load_model = bayselm.PYHSMM(bayselm.Load('pyhsmm', saveFile))
        # test_data_container_load = load_model.TestWordSegmentationForPython(data_container.Sents, threads)
        # for i in range(test_data_container_load.Size):
        #     print(test_data_container_load.GetWordSeq(i))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='unsupervised word segmentation')
    parser.add_argument(
        '--mode',
        help='language model or unsupervised word segmentation model',
        type=str,
        choices=['lm', 'ws'],
        required=True)
    parser.add_argument(
        '--model',
        help='select Baysian language model',
        type=str,
        choices=['npylm', 'pyhsmm'],
        required=True)
    parser.add_argument(
        '--trainFile',
        help='training file path',
        type=str,
        required=True)
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
    parser.add_argument(
        '--saveFile',
        help='file path to save model',
        type=str,
        default='')
    parser.add_argument(
        '--saveFormat',
        help='model save format',
        type=str,
        choices=['notindent', 'indent'],
        default='notindent')
    args = parser.parse_args()

    if args.mode == 'lm':
        raise NotImplementedError
    elif args.mode == 'ws':
        train_word_segmentation(args.model, args.trainFile, args.theta, args.d, args.gammaA, args.gammaB, args.betaA,
                                args.betaB, args.alpha, args.beta, args.maxNgram, args.maxWordLength, args.posSize,
                                (1.0 / args.vocabSize), args.epoch, args.threads, args.batch,
                                args.saveFile, args.saveFormat)
    else:
        assert(False)
