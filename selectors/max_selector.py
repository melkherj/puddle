import random,math
import numpy as np
from abstract_selector import Selector
from operator import itemgetter
from scipy.stats import rankdata
from utils import add_noise,adjust,logodds,sigmoid
from collections import Counter
import logging

#@author melkherj

class MaxSelector(Selector):
    def __init__(self,**kwargs):
        Selector.__init__(self,**kwargs)
        self.thresh = self.conf['active_thresh']
        # start off with something for stability
        self.n_pos = 1.0
        self.exp_n_pos = 1.0
        self.n_neg = 1.0
        self.exp_n_neg = 1.0
        self.logger = logging.getLogger('active_semisup.max_selector')

    def next_indices(self,Y,state_T,modeler,semisupervised=False,n_ixs=None):
        if n_ixs is None:
            n_ixs = self.n_ixs
        # not already selected
        n = len(Y)
        Pmax = modeler.P.max(axis=1)
        k = self.n_pos+self.n_neg
        alpha = 0.022
        beta = 4.80
        p_active = 0.99-np.exp(-alpha*(k+5+beta)) #1. - 1./((k+10.)**0.5)
        p_semisup = 0.99-np.exp(-alpha*(k+50+beta)) #sigmoid(logodds(p_active)+0.5)
        active_thresh = Pmax[int(p_active*n)]
        active = [int(np.argmin(np.abs(Pmax-
                        add_noise(self.thresh,
                        variance=1.5*math.exp(-0.027*k)))))
                    for t in range(n_ixs)]

        if False: #len(state_T)>=30 and len(active)>0: #start doing active at around 30?
            n_pos_next = sum(modeler.Yp[active]==modeler.Y[active])
            n_neg_next += sum(modeler.Yp[active]!=modeler.Y[active])
            self.n_pos += n_pos_next
            self.n_neg += n_neg_next
            decay = conf['active_decay']
            self.exp_n_pos = self.exp_n_pos*decay + (1-decay)*n_pos_next
            self.exp_n_neg = self.exp_n_neg*decay + (1-decay)*n_neg_next
            self.logger.info('thresh1: %d,%d-%d,%d-%.3f'%(self.n_pos,self.n_neg,
                self.exp_n_pos,self.exp_n_neg,self.thresh))
            self.thresh = adjust(self.thresh,n_pos=self.exp_n_pos,n_neg=self.exp_n_neg,
                p=self.conf['active_thresh'])
            self.logger.info('thresh2: %d,%d-%d,%d-%.3f'%(self.n_pos,self.n_neg,
                self.exp_n_pos,self.exp_n_neg,self.thresh))
        if semisupervised:
            #fraction that should be semisup at this number of labels
            semisup = np.argsort(Pmax)[-int(p_semisup*n):]
            semisup = map(int,semisup)
            return (active,semisup)
        else:
            return active
