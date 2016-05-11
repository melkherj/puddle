#!/usr/bin/env python

from active_evaluate import load_conf
import sys,os
from mnist_modeler import MNISTModeler
import numpy as np
import json
from selectors_registry import selectors_registry
import random
from subprocess import check_output

conf = load_conf()

################################################################################################
# get variables needed for active learning system
mnm = MNISTModeler(train_n=conf['train_size'],test_n=conf['test_size'],
        seed=conf['random_seed'])

k = 50
for i in range(1000):
    x = mnm.train_X[i*k:(i+1)*k]
    y = mnm.train_Y[i*k:(i+1)*k]
    mnm.update_model(x,y)
    print 'with %d labels accuracy %.3f'%((i+1)*k,mnm.test_accuracy())
