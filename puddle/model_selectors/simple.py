import random
import numpy as np

class RandomSelector:
    def next_indices(self,X,ixs,Y_sub,model,n_ixs=1):
        n = X.shape[0]
        remaining_ixs = list(set(range(n)) - set(ixs))
        n_to_sample = min(n_ixs,len(remaining_ixs))
        return random.sample(remaining_ixs,n_to_sample)

class UncertaintySelector:
    def next_indices(self,X,ixs,Y_sub,model,n_ixs=1):
        P = model.predict_proba(X)[:,1]
        margin = np.abs(P-0.5)
        big_margin = 1000.
        margin[ixs] = big_margin #don't pick any indices we've already picked
        chosen_ixs = np.argsort(margin)[:n_ixs]
        return list(set(chosen_ixs)-set(ixs))

simple_selectors = {
    'random':RandomSelector(),
    'uncertainty':UncertaintySelector()
    }
