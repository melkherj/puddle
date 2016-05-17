import random,math
import numpy as np
from abstract_selector import Selector
from operator import itemgetter

#@author melkherj

class RandomSelector(Selector):
    def __init__(self,**kwargs):
        Selector.__init__(self,**kwargs)

    #TODO  make "thresh" an ordered history of all labels/thresholds
    def next_indices(self,Y,state_T,modeler,n_ixs=None):
        ''' 
            Given labels (-1/0/1 for no/haven't labeled/yes respectively), scores P, 
            and current state of labeling, return the next index (into the P/Y arrays) to label
            None if there's nothing left to label '''
        if n_ixs is None:
            n_ixs = self.default_n_ixs
        return self.randomly_select(Y,n_ixs=n_ixs)
