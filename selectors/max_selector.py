import random,math
import numpy as np
from abstract_selector import Selector
from operator import itemgetter
from scipy.stats import rankdata
from utils import add_noise,adjust

#@author melkherj

class MaxSelector(Selector):
    def __init__(self,**kwargs):
        Selector.__init__(self,**kwargs)
        self.thresh = self.conf['active_thresh']
        # start off with something for stability
        self.n_pos = 1.0
        self.n_neg = 1.0

    def next_indices(self,Y,state_T,modeler,semisupervised=False,n_ixs=None):
        if n_ixs is None:
            n_ixs = self.n_ixs
        # not already selected
        n = len(Y)
        Pmax = modeler.P.max(axis=1)
        active = [int(np.argmin(np.abs(Pmax-
                        add_noise(self.thresh,
                        variance=self.conf['active_variance']))))
                    for t in range(n_ixs)]
        self.n_pos += sum(modeler.Yp[active]==modeler.Y[active])
        self.n_neg += sum(modeler.Yp[active]!=modeler.Y[active])
        self.thresh = adjust(self.thresh,n_pos=self.n_pos,n_neg=self.n_neg,
                p=self.conf['active_thresh'])
        if semisupervised:
            semisup = map(int,list(np.nonzero(Pmax>self.conf['semisup_thresh'])[0]))
            return (active,semisup)
        else:
            return active
