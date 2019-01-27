#!/usr/bin/env python 

import sys,os 
import numpy as np
import json
import random
import time
from collections import Counter
from sklearn.metrics import average_precision_score, accuracy_score, f1_score, log_loss
from sklearn.cross_validation import train_test_split

def binary_class_metrics(Y_train,Y_test,P,thresh=0.5):
    Y_hat = P[:,1] > thresh
    return {
            'accuracy':accuracy_score(Y_test,Y_hat),
            'f1':f1_score(Y_test,Y_hat),
            'logloss':log_loss(Y_test,P),
            'train_size':len(Y_train),
            'train_positives':Y_train.sum(),
            'train_base_rate':Y_train.mean(),
            'test_size':len(Y_test),
            'test_positives':Y_test.sum(),
            'test_base_rate':Y_test.mean()
            }

def active_evaluate(X,Y,model,selector,epochs,labels_per_epoch,test_size=0.1):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=test_size)
    selected_ixs = [] #ordered list of indices we've asked for human labels from so far
    metrics_by_epoch = []
    for epoch in range(epochs):
        next_ixs = selector.next_indices(X_train,selected_ixs,Y_train[selected_ixs],model,
            n_ixs=labels_per_epoch)
        assert len(set(next_ixs)&set(selected_ixs)) == 0 #must be new indices
        selected_ixs += next_ixs
        # TODO allow for importance weights in model
        model.fit(X_train[selected_ixs],Y_train[selected_ixs])
        P = model.predict_proba(X_test)
        metrics = binary_class_metrics(Y_train[selected_ixs],Y_test,P)
        metrics_by_epoch.append(metrics)
    return metrics_by_epoch

