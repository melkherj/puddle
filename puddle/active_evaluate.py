#!/usr/bin/env python
import pandas as pd
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

def active_evaluate(X, Y, model, selector, epochs, labels_per_epoch, bootstrap_size=10, batch_run=None, batch_info=None):
    """

    :param X: (np.matrix)
            Training Data Matrix Features
    :param Y: (np.array)
            Training Labels
    :param model: (sklearn model)
            Classifier Model
    :param selector:
            Selector class - Must expose a method called next_indices(X, selected_ixs, Y, model,n_ixs=...)
            the returns the next indices to sample given training features X, human
            labeled points Y, and their associated indices 'selected_ixs'
    :param epochs:
    :param labels_per_epoch:
    :param bootstrap_size:
    :return:
    """

    # ordered list of indices we've asked for human labels so far

    # TODO take the bootstrap outside of active_evaluate function
    metrics_by_epoch = []
    positive_ixs = random.sample(set(np.where(Y)[0]), int(bootstrap_size/2)+1)
    negative_ixs = random.sample(set(np.where(~Y)[0]), int(bootstrap_size/2)+1)

    try:
        selected_ixs = positive_ixs + negative_ixs
        # selected_ixs = random.sample(set(np.arange(X.shape[0])), bootstrap_size) +
        model.fit(X[selected_ixs], Y[selected_ixs])  # pre-fit
    except:
        return pd.DataFrame()
    for epoch in range(epochs):
        if epochs <= 20 or epoch % (epochs//20) == 0:
            print('Batch {}, bootstrap {} -- {}/{} epochs...'.format(batch_info, batch_run, epoch, epochs))
        next_ixs = selector.next_indices(X, selected_ixs, Y[selected_ixs], model,
                                         n_ixs=labels_per_epoch)
        assert len(set(next_ixs) & set(selected_ixs)) == 0 # must be new indices
        selected_ixs += next_ixs
        # TODO allow for importance weights in model
        model.fit(X[selected_ixs], Y[selected_ixs])
        P = model.predict_proba(X)
        metrics = binary_class_metrics(Y[selected_ixs],Y,P,len(Y))
        metrics['seed_initialization_run'] = batch_run
        metrics_by_epoch.append(metrics)

    df = pd.DataFrame(metrics_by_epoch)
    df["f1_mean"] = df.groupby(['train_size'])['f1'].transform(np.mean)
    df["f1_std"] = df.groupby(['train_size'])['f1'].transform(np.std)
    return df


def run_experiment(experiment, epochs=None, labels_per_epoch=None):
    """

    :param experiment:
    :return:
    """
    dataset, model1, selector1, batch_run = experiment
    dataset_name, (X,Y) = dataset
    (model_name, model) = model1
    (selector_name, selector) = selector1
    print('evaluating dataset/model/selector: ',dataset_name,model_name,selector_name,'...')

    # TODO don't hardcode epochs and number of examples per epoch
    # epochs = len(Y)//labels_per_epoch
    try:
        metrics_by_epoch = active_evaluate(X, Y, model, selector, epochs, labels_per_epoch, batch_run=batch_run, batch_info=[dataset_name, model_name, selector_name])
    except Exception as e:
        print("Trouble with dataset {}, model {}, selector {}".format(dataset_name, model_name, selector_name))
        raise e
    metrics_by_epoch['dataset'] = dataset_name
    metrics_by_epoch['model'] = model_name
    metrics_by_epoch['selector'] = selector_name
    metrics_by_epoch["batch_run"] = batch_run
    return metrics_by_epoch