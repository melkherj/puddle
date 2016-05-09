import random,math
import numpy as np
from abstract_selector import Selector
from operator import itemgetter
from scipy.stats import rankdata

#@author melkherj

class EntropySelector(Selector):
    def __init__(self,**kwargs):
        Selector.__init__(self,**kwargs)

    def next_indices(self,Y,state_T,modeler):
        # not already selected
        n = len(Y)
        P = modeler.P
        entropy = -(P*np.log(P)).sum(axis=1)
        entropy[Y>=0] = np.NINF #don't pick already-labeled cases
        
        # turn entropy into a probability of selecting (P2)
        P2 = 1./(n-rankdata(entropy)+n/4)
        P2 = P2/P2.sum()
        # alpha-smooth
        k = len(state_T) #number labeled so far
        eps = max(1.0-0.03*k,0.05) #how much to emphasize uniform (exploration vs. exploitation)
        P2 = eps*1./n + (1-eps)*P2 #eps = amount of exploration

        # biased-sample
        ixs = np.random.choice(range(n),size=self.n_ixs,p=P2,replace=False)
        
        return ixs
