import random

class RandomSelector:
    def next_indices(self,X,ixs,Y_sub,model,n_ixs=1):
        n = len(X)
        remaining_ixs = list(set(range(n)) - set(ixs))
        n_to_sample = min(n_ixs,len(remaining_ixs))
        return random.sample(remaining_ixs,n_to_sample)

simple_selectors = {
    'random':RandomSelector()
    }
