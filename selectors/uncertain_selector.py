import random,math
import numpy as np
from abstract_selector import Selector
from operator import itemgetter
from scipy.stats import rankdata
from utils import add_noise,adjust,logodds,sigmoid
from collections import Counter
import logging

#@author melkherj

class UncertainSelector(Selector):
    def __init__(self,**kwargs):
        Selector.__init__(self,**kwargs)
        self.thresh = self.conf['active_thresh']
        # start off with something for stability
        self.logger = logging.getLogger('active_semisup.max_selector')

    def next_indices(self,Y,state_T,modeler,n_ixs=None):
        if n_ixs is None:
            n_ixs = self.n_ixs
        # not already selected
        P_select = np.abs(modeler.Pmax-self.thresh)
        P_select[Y>=0] = np.inf #don't return cases already labeled
        active = map(int,np.argsort(P_select)[:n_ixs]) #min n_ixs by P_select
        return active
