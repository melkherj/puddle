#!/usr/bin/env python 

import sys,os 
from mnist_modeler import MNISTModeler
import numpy as np
import json
from selectors_registry import selectors_registry
import random
from subprocess import check_output
import logging
from utils import setup_logging,adjust_freq_sample,add_noise
import time
from collections import Counter

def append_results(results,filename='results/mnist'):
    # append to active_results
    with open(filename,'a') as f:
        f.write(json.dumps(results)+'\n')
        
def load_conf(filename='active_conf.json'):
    # which selector, for how many iterations, what dataset sizes, etc?
    with open(filename,'r') as f:
        conf = json.load(f)
    return conf

def active_evaluate(conf):

    setup_logging()
    logger = logging.getLogger('active_semisup.active_evaluate')
    
    ################################################################################################
    # git version
    #repo must be clean, since we're using git hash
    git_hash = check_output(['git','rev-parse','--short','HEAD']).strip()
    if check_output(['git','diff'])!='':
        git_hash += '-dirty'
    
    ################################################################################################
    # get the selector class we'll be using for active selection
    SelectorClass = selectors_registry[conf['selector']['name']]
    
    ################################################################################################
    # get variables needed for active learning system
    mnm = MNISTModeler(seed=conf['random_seed'])
    selector = SelectorClass(conf=conf['selector'],n_ixs=conf['batch_size'],seed=conf['random_seed'])
    state = []
    Y = np.ones(len(mnm.Y_train))*-1
    I = random.sample(range(mnm.n),conf['initial_random_training']) #start off w/30
    exp_accuracy = 0.0 #exponentially decaying weighted accuracy, with factor conf['accuracy_gamma']
    epoch_stats = []
    
    ################################################################################################
    ### simulate active and semi-supervised labeling, track accuracy
    for epoch in range(conf['epochs']):
        epoch_start_time = time.time()
        # get next indices
        n_ixs = int(random.random()<conf['active_label_prob']) #1 active label with probability 0.1
        I_next,semisup = selector.next_indices(Y,state,mnm,n_ixs=n_ixs,semisupervised=True)
        I = sorted(list(set(I)|set(I_next))) #distinct add I_next
        Pmax = mnm.P.max(axis=1)

        n_labeled_correct = sum(mnm.Yp[I_next]==mnm.Y[I_next]) if len(I_next)>0 else None
        logger.info('label dist:'+str(sorted(
            Counter(mnm.Y[I]).items()
            )))
        labeled_score = float(Pmax[I_next].mean()) if len(I_next)>0 else None
        score_ixs = random.sample(range(mnm.n),conf['n_rescore'])
        scorediff = mnm.score_train(indices=np.array(score_ixs))
        # label/set state
        for ix,label_row in zip(I_next,mnm.Y_train[I_next]):
            y_label = int(np.nonzero(label_row)[0][0])
            state.append((ix,y_label))
            Y[ix] = y_label
        # update model
    
        # now bootstrap sample from semi/active, uniform probability across categories
        #numbers of active/semisupervised to bootstrap sample for this iteration
        I_labels = mnm.Y[I] 
        I_samples = adjust_freq_sample(I,I_labels,[1]*mnm.d,conf['minibatch_labeled'])
        if conf['selector']['name']=='max':
            semisup_labels = mnm.Yp[semisup]
            semisup_samples = adjust_freq_sample(semisup,semisup_labels,[1]*mnm.d,conf['minibatch_semisup'])
        elif conf['selector']['name']=='random':
            semisup_samples = []
    
        #        len(I_samples),len(semisup_samples)))
        mnm.update_model(I_samples,semisup=semisup_samples)
        # evaluate accuracy
        if epoch % conf['accuracy_epochs']==0:
            accuracy = float(mnm.test_accuracy(
                indices=random.sample(range(mnm.n_test),conf['n_accuracy'])))
        exp_accuracy = conf['accuracy_gamma']*accuracy+ (1-conf['accuracy_gamma'])*exp_accuracy
        s = 'batch %d: accuracy/exp-weighted/scorediff/n-labeled/n-semisupervised\n'%(epoch)
        s += '    %.3f/%.3f/%.3f/%d/%d\n'%(accuracy,exp_accuracy,scorediff,len(I),len(semisup))
        stats = {'accuracy':accuracy,
                 'exp_accuracy':exp_accuracy,
                 'scorediff':scorediff,
                 'n_labeled':len(I),
                 'n_semisup':len(semisup),
                 'epoch':epoch,
                 'seconds':time.time()-epoch_start_time,
                 'n_next_labeled':len(I_next),
                 'n_labeled_correct':n_labeled_correct,
                 'labeled_score':labeled_score}
        epoch_stats.append(stats)
        logger.info(s)
       
    
    ################################################################################################
    ### get results in a format emenable to saving
    results = conf.copy()
    results['epoch_stats'] = epoch_stats
    results['githash'] = git_hash

    return results
    
