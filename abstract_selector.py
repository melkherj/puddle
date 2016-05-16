import numpy as np
import random

#@author melkherj

class Selector(object):
    ''' An entity that just has available labels and scores.  From this, it determines
        - whether there are enough labels to train a model
        - if so, & given scores, what examples to view 
        percentiles indicates whether we first transform scores to percentiles '''

    def __init__(self,n_ixs=1,seed=None,conf={}):
        self.n_ixs=n_ixs
        self.seed = seed
        self.conf = conf
        random.seed(self.seed)
        np.random.seed(self.seed)

    def next_indices(self,Y,state,modeler):
        ''' Return a list of indices giving examples to be displayed '''
        raise NotImplementedError()

    def randomly_select(self,Y):
        '''
            Given labels (-1/0/1 for no/haven't labeled/yes respectively), scores P,
            and current state of labeling, return the next index (into the P/Y arrays) to label
            None if there's nothing left to label '''
        n = len(Y)
        ixs = set(range(n))
        ixs = ixs-set(np.nonzero(Y>=0)[0]) #remove cases already labeled
        ixs = sorted(list(ixs))
        if len(ixs)<self.n_ixs:
            return ixs 
        else:
            return random.sample(ixs,self.n_ixs)
