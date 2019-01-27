import random
import numpy as np
from scipy.special import logit
from scipy.sparse import issparse
import logging
logger = logging.getLogger(__name__)

class RandomSelector:
    """
    Random Active Learning Selector
    """
    def next_indices(self, X, ixs, Y_sub, model, n_ixs=1):
        """

        :param X: (array) Pool of feature vectors
        :param ixs: (list(n)) List of formerly Labeled training indices
        :param Y_sub: (array(n,2)) Array of former labels
        :param model: Model class that learning to label
        :param n_ixs: The number of 'steps ahead' to look
        :return:
        """
        logger.info("Random Selector call: ixs={}, mode={}".format(len(ixs), model))
        n = X.shape[0]
        remaining_ixs = list(set(range(n)) - set(ixs))
        n_to_sample = min(n_ixs,len(remaining_ixs))
        return random.sample(remaining_ixs,n_to_sample)


class UncertaintySelector:
    def next_indices(self,X,ixs,Y_sub,model,n_ixs=1):
        logger.info("Uncertainy Selector call: ixs={}, mode={}".format(len(ixs), model))
        P = model.predict_proba(X)[:,1]
        margin = np.abs(P-0.5)
        big_margin = 1000.
        margin[ixs] = big_margin # don't pick any indices we've already picked
        chosen_ixs = np.argsort(margin)[:n_ixs]
        return list(set(chosen_ixs)-set(ixs))


class FisherSelector:
    def next_indices(self, X, ixs, Y_sub, model, n_ixs=1):
        """

        :param X: (array) Pool of feature vectors
        :param ixs: (list(n)) List of formerly Labeled training indices
        :param Y_sub: (array(n,2)) Array of former labels
        :param model: Model class that learning to label
        :param n_ixs: The number of 'steps ahead' to look
        :return:
        """
        # If feature space too large, default to uncertainty
        if X.shape[1] > 500:
                P = model.predict_proba(X)[:, 1]
                margin = np.abs(P - 0.5)
                big_margin = 1000.
                margin[ixs] = big_margin  # don't pick any indices we've already picked
                chosen_ixs = np.argsort(margin)[:n_ixs]
                return list(set(chosen_ixs) - set(ixs))
        logger.info("Fisher Selector call: ixs={}, X.shape={}, mode={}".format(len(ixs), X.shape, model))

        logger.info('Calculating prior information matrix...')
        prior_fisher = self.get_fisher_information_matrix(model, X[ixs])
        logger.info('Calculating prior information...')
        prior_info = np.linalg.det(prior_fisher)
        logger.info('Calculating fisher information matrices for every X...')
        matrices = [self.get_fisher_information_matrix(model, x) for x in X]
        logger.info('Calculating incremental information gain for every X...')
        info_gains = [np.log(np.linalg.det(prior_fisher + M)) - prior_info for M in matrices]
        logger.info('Sorting...')
        chosen_ixs = np.argsort(info_gains)[:n_ixs]
        return list(set(chosen_ixs)-set(ixs))


    @classmethod
    def get_fisher_information_matrix(self, model, X):
        """ Calculate Fisher Information Matrix for sample data X
            Assumes logistic regression model specification
        """
        if issparse(X):
            X = X.todense()
        if len(X.shape) < 2:
            X = np.matrix(X)  # N by D matrix

        D_ii = 2.0 * (1.0 + np.cosh(logit(model.predict_proba(X)[:, 1])))  # N array
        D_ii2 = (X / D_ii[:, np.newaxis]).T
        F_ij = np.dot(D_ii2, X)
        # F_ij += self.C * np.diag(np.ones(F_ij.shape[0]))  # Prior Regularization
        return F_ij


simple_selectors = {
    'random':RandomSelector(),
    'uncertainty':UncertaintySelector(),
    'fisher': FisherSelector()
    }
