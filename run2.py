#!/usr/bin/env python

from active_evaluate import load_conf
import sys,os
from mnist_modeler import MNISTModeler
import numpy as np
import json
from selectors_registry import selectors_registry
import random
from subprocess import check_output
from matplotlib import pyplot as plt
import pandas as pd
import logging
import math

cheat = False

def sigmoid(z):
    return 1./(1.+math.exp(-z))

def logodds(p):
    return math.log(p/float(1-p))

def adjust(thresh,n_pos,n_neg,gamma=0.2,lower=0.001,upper=0.999):
    ''' thresh is current score threshold
        ratio is ratio of negatives seen so far to positives '''
    ratio = float(n_pos)/n_neg
    thresh = sigmoid(logodds(thresh)-gamma*math.log(ratio))
    return max(min(thresh,upper),lower)

def add_noise(thresh,variance=0.05):
    return sigmoid(logodds(thresh)+variance*float(np.random.randn()))

# set up logger
def setup_logging(loggername='active_semisup',logfile='active_semisup.log'):
    logger = logging.getLogger(loggername)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logging()

k_I = 80 #minibatch size (for now same for supervised/semi-supervised)
k_semi = 20 #minibatch size (for now same for supervised/semi-supervised)
epochs = 100
active_samples = 5
semisup_thresh = 0.99

class M(object):
    def __init__(self,mnm):
        logger.info('Setting up M')
        self.Is = [] #distinct id's used so far
        self.I_sequence = [] #sequenceof mini-batches trained on.  so we can exactly reproduce results
        self.logger = logging.getLogger('active_semisup.M')
        self.accuracies = []
        self.mnm = mnm
        self.Y = np.argmax(mnm.train_Y,axis=1)
        self.n,self.d = mnm.train_Y.shape
        self.score_train()
        logger.info('Done setting up M')
    
    def update(self,I,score_train=True,calculate_accuracy=True,semisup=[]):
        I = map(int,list(I))
        self.I_sequence.append(I)
        self.Is = sorted(list(set(I)|set(self.Is))) #all indices requested to label so far
        I = I+semisup
        x = self.mnm.train_X[I]
        y = self.mnm.train_Y[I]
        if len(semisup)>0:
            y[-len(semisup):] = self.Yp_sparse[semisup,:] #use fake-labels for semi-supervised
        # replace labels with pseudo-labels for semi-supervised
        self.mnm.update_model(x,y,score_train=score_train)
        if calculate_accuracy:
            self.test_accuracy()
            
    def test_accuracy(self):
        accuracy = self.mnm.test_accuracy()
        self.accuracies.append((len(self.Is),accuracy))
        logger.info('with %d labels total, %d added, accuracy %.3f'%(len(self.Is),len(I),accuracy))
        return accuracy
        
    def score_train(self):
        mnm.score_train()
        self.Yp = np.argmax(mnm.P,axis=1) #pick most likely label for each category
        self.Yp_sparse = np.zeros((self.n,self.d))
        for i,j in zip(range(self.n),self.Yp): #create a sparse representation of Yp
            self.Yp_sparse[i,j] = 1.0

    def topcat(self,category,k=10):
        ''' top <k> indices from the given category'''
        # Select top ranked indices j->j+10 for given category
        cat_sort = map(int,list(np.argsort(self.mnm.P[:,category])))
        return cat_sort[-k:]

    def viz_train(self,i):
        x = self.mnm.train_X[i,:]
        self.logger.info('index: %d,label: %d, predicted: %d'%(i,self.Y[i],self.Yp[i]))
        plt.imshow(x.reshape(28,28), interpolation='nearest',cmap='Greys')
        plt.show()

conf = load_conf()

################################################################################################
# get variables needed for active learning system
mnm = MNISTModeler(train_n=conf['train_size'],test_n=conf['test_size'],
        seed=conf['random_seed'])

m = M(mnm)

# # Select Indices #
I = range(30)

# # Update Model #
semisup = []
random.seed(0)

active_thresh = 0.8
n_pos = 1
n_neg = 1
active_thresh_var0 = 0.5

for super_epoch in range(20):
    for i in range(epochs):
        # assume uniform-predicted-category!  pick uniform number from each, so predictions are balanced
        # with replacement!
        category = random.choice(range(m.d))
    
        I_cat = np.array(I)
        semisup_cat = np.array(semisup)
        I_cat = I_cat[m.Y[I_cat]==category]
        if len(semisup_cat)>0:
            semisup_cat = semisup_cat[m.Yp[semisup_cat]==category]
        if len(I_cat)>0:
            I_cat = np.random.choice(I_cat,k_I)
        if len(semisup_cat)>0:
            semisup_cat = np.random.choice(semisup_cat,k_semi)
        
        # back to lists
        I_cat = map(int,I_cat)
        semisup_cat = map(int,semisup_cat)
        m.update(I_cat,score_train=False,calculate_accuracy=False,semisup=semisup_cat)

        if i%10==0:
            logger.info('epoch %d'%i)
            sys.stdout.flush()
    
    ### check performance, re-score
    logger.info('Number of labels used: %d'%len(m.Is))
    m.test_accuracy()
    m.score_train()
    sys.stdout.flush()
    Pmax = mnm.P.max(axis=1) #useful for active/semi-supervised example selection

    # Create semi-supervised indices/label set
    semisup = map(int,list(np.nonzero(Pmax>semisup_thresh)[0]))

    # Add to labeled set (this is currently the cheating part)
    # randomly sample labels from incorrect semisupervised indices
    semisup_disagree = np.nonzero(m.Yp[semisup]!=m.Y[semisup])[0]
    semisup_disagree = [semisup[i] for i in semisup_disagree] 
    if cheat:
        active = list(np.random.choice(semisup_disagree,active_samples))
    else:
        # get active labels, adjust threshold
        logger.info('active_thresh before: %.4f'%active_thresh)
        active_thresh = adjust(active_thresh,n_pos,n_neg,gamma=0.2)
        logger.info('active_thresh after: %.4f'%active_thresh)
        noised_threshs = []
        active = []
        for _ in range(active_samples):
            thresh_noised = add_noise(active_thresh,variance=active_thresh_var0/(super_epoch+1))
            noised_threshs.append(thresh_noised)
            active.append(int(np.argmin(np.abs(Pmax-thresh_noised))))
        noised_threshs = ','.join([str(round(t,4)) for t in noised_threshs])
        correct = m.Yp[active]==m.Y[active]
        n_pos += sum(correct)
        n_neg += sum(~correct)
        logger.info('noised threshs: %s'%noised_threshs)
        logger.info('n_pos,n_neg=%d,%d'%(n_pos,n_neg))
    mnm.checkpoint('super_epoch_%d'%super_epoch)
    
    logger.info('before active/I len: %d/%d'%(len(active),len(I)))
    I += active
    logger.info('after active/I len: %d/%d'%(len(active),len(I)))
    logger.info('active score avg/std: %.3f,%.3f'%(
            mnm.P[active,:].max(axis=1).mean(),
            mnm.P[active,:].max(axis=1).std()))
    logger.info('overall semisup score avg/std: %.3f,%.3f'%(
            mnm.P[semisup,:].max(axis=1).mean(),
            mnm.P[semisup,:].max(axis=1).std()))
    logger.info('n semisup, n semisup disagree: %d,%d'%(len(semisup),len(semisup_disagree)))

