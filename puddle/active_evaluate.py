#!/usr/bin/env python

import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.cross_validation import train_test_split
import logging
logger = logging.getLogger(__name__)


def binary_class_metrics(Y_train,Y_test,P,n_total_train,thresh=0.5):
    """
    Return a dictionary of classification metrics

    :param Y_train: (int) Training Labels
    :param Y_test: (int) Test Labels
    :param P: (array(float), shape=(n_total_train, 2)) Probabilities
    :param n_total_train: (int) Total Number of data points trained/labeled
    :param thresh: (float) Binary Decision Threshold
    :return: (dict)
    """
    Y_hat = P[:,1] > thresh
    return {
            'accuracy':accuracy_score(Y_test,Y_hat),
            'f1':f1_score(Y_test,Y_hat),
            'logloss':log_loss(Y_test,P),
            'train_size':len(Y_train),
            'n_total_train':n_total_train,
            'train_fraction':len(Y_train)/float(n_total_train),
            'train_positives':Y_train.sum(),
            'train_base_rate':Y_train.mean(),
            'test_size':len(Y_test),
            'test_positives':Y_test.sum(),
            'test_base_rate':Y_test.mean()
            }

def active_evaluate(X, Y, model, selector, epochs, labels_per_epoch,test_size=0.1, bootstrap_size=10):
    """

    :param X:
    :param Y:
    :param model:
    :param selector:
    :param epochs:
    :param labels_per_epoch:
    :param test_size:
    :return:
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    selected_ixs = random.sample(set(np.arange(X_train.shape[0])), bootstrap_size) # ordered list of indices we've asked for human labels so far
    model.fit(X_train[selected_ixs], Y_train[selected_ixs])
    metrics_by_epoch = []
    for epoch in range(epochs):
        logger.info('%d/%d epochs...' % (epoch, epochs))
        if epochs <= 20 or epoch % (epochs//20) == 0:
            print('%d/%d epochs...'%(epoch, epochs))
        next_ixs = selector.next_indices(X_train, selected_ixs, Y_train[selected_ixs], model,
                                         n_ixs=labels_per_epoch)
        assert len(set(next_ixs)&set(selected_ixs)) == 0 #must be new indices
        selected_ixs += next_ixs
        # TODO allow for importance weights in model
        model.fit(X_train[selected_ixs], Y_train[selected_ixs])
        P = model.predict_proba(X_test)
        metrics = binary_class_metrics(Y_train[selected_ixs],Y_test,P,len(Y_train))
        metrics_by_epoch.append(metrics)
    return metrics_by_epoch

