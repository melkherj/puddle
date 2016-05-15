import random,math
import numpy as np
from abstract_selector import Selector
from operator import itemgetter
from scipy.stats import rankdata
from utils import add_noise

#@author melkherj

class MaxSelector(Selector):
    def __init__(self,**kwargs):
        Selector.__init__(self,**kwargs)

    def next_indices(self,Y,state_T,modeler,semisupervised=False,n_ixs=None):
        if n_ixs is None:
            n_ixs = self.n_ixs
        # not already selected
        n = len(Y)
        Pmax = modeler.P.max(axis=1)
        active_thresh = 0.9
        variance = 0.1
        semisup_thresh = 0.99
        active = [int(np.argmin(np.abs(Pmax-
                        add_noise(active_thresh,variance=variance))))
                    for t in range(n_ixs)]

        if semisupervised:
            semisup = map(int,list(np.nonzero(Pmax>semisup_thresh)[0]))
            return (active,semisup)
        else:
            return active
