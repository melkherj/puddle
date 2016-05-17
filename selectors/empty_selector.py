import random,math
import numpy as np
from abstract_selector import Selector
from operator import itemgetter

#@author melkherj

class EmptySelector(Selector):
    def __init__(self,**kwargs):
        Selector.__init__(self,**kwargs)

    def next_indices(self,Y,state_T,modeler,n_ixs=None):
        return []
