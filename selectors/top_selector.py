import random,math
import numpy as np
from abstract_selector import Selector
from operator import itemgetter
from scipy.stats import rankdata
from utils import add_noise,adjust,logodds,sigmoid
from collections import Counter
import logging

#@author melkherj

class TopSelector(Selector):
    ''' Select top n_ixs by max score '''

    def __init__(self,**kwargs):
        Selector.__init__(self,**kwargs)
        # start off with something for stability
        self.logger = logging.getLogger('active_semisup.max_selector')

    def next_indices(self,Y,state_T,modeler,n_ixs=None):
        if n_ixs is None:
            n_ixs = self.n_ixs
        # not already selected
        n = len(Y)

        #fraction that should be semisup at this number of labels
        #semisup = np.argsort(modeler.Pmax)[-n_ixs:]
        semisup = np.nonzero(modeler.Pmax>0.97)[0] #argsort(modeler.Pmax)[-n_ixs:]
        semisup = map(int,semisup)
        return semisup
