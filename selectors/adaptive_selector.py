import random,math
import numpy as np
from abstract_selector import Selector
from operator import itemgetter
from scipy.stats import rankdata
from utils import add_noise,adjust,logodds,sigmoid
from collections import Counter
import logging

#@author melkherj

class AdaptiveSelector(Selector):
    def __init__(self,**kwargs):
        Selector.__init__(self,**kwargs)
        self.thresh = self.conf['active_thresh']
        # start off with something for stability
        self.n_pos = 1.0
        self.exp_n_pos = 1.0
        self.n_neg = 1.0
        self.exp_n_neg = 1.0
        self.logger = logging.getLogger('active_semisup.max_selector')

    def next_indices(self,Y,state_T,modeler,n_ixs=None):
        if n_ixs is None:
            n_ixs = self.n_ixs
        # not already selected
        k = self.n_pos+self.n_neg
        active = [int(np.argmin(np.abs(modeler.Pmax-
                        add_noise(self.thresh,
                        variance=1.5*math.exp(-0.027*k)))))
                    for t in range(n_ixs)]
        if n_ixs>0: 
            # if there's at least one label in this batch, adjust the threshold
            n_pos_next = sum(modeler.Yp[active]==modeler.Y[active])
            n_neg_next = sum(modeler.Yp[active]!=modeler.Y[active])
            self.n_pos += n_pos_next
            self.n_neg += n_neg_next
            decay = self.conf['active_decay']
            self.exp_n_pos = self.exp_n_pos*decay + (1-decay)*n_pos_next
            self.exp_n_neg = self.exp_n_neg*decay + (1-decay)*n_neg_next
            self.logger.info('thresh1: %d,%d-%d,%d-%.3f'%(self.n_pos,self.n_neg,
                self.exp_n_pos,self.exp_n_neg,self.thresh))
            self.thresh = adjust(self.thresh,n_pos=self.exp_n_pos,n_neg=self.exp_n_neg,
                    p=self.conf['active_thresh'])
            self.logger.info('thresh2: %d,%d-%d,%d-%.3f'%(self.n_pos,self.n_neg,
                self.exp_n_pos,self.exp_n_neg,self.thresh))
        return active
