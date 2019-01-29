import random
import numpy as np
from scipy.special import logit
from scipy.sparse import issparse
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD, PCA
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
        n = X.shape[0]
        remaining_ixs = list(set(range(n)) - set(ixs))
        n_to_sample = min(n_ixs,len(remaining_ixs))
        return random.sample(remaining_ixs,n_to_sample)


class UncertaintySelector:
    def next_indices(self,X,ixs,Y_sub,model,n_ixs=1):
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
        # If feature space too large, decompose feature space to 50 features
        if X.shape[1] > 100:
            if issparse(X):
                X = TruncatedSVD(n_components=50).fit_transform(X)
            else:
                X = PCA(n_components=50).fit_transform(X)

            logger.info('Feature Space Large, overwriting model with internal Logistic Regression...')
            model = LogisticRegression()
            model.fit(X[ixs], Y_sub)

        prior_fisher = self.get_fisher_information_matrix(model, X[ixs])
        prior_info = np.linalg.det(prior_fisher)
        matrices = [self.get_fisher_information_matrix(model, x) for x in X]
        info_gains = [np.log(np.linalg.det(prior_fisher + M)) - prior_info for M in matrices]
        chosen_ixs = np.argsort(info_gains)

        return list(set(chosen_ixs)-set(ixs))[:n_ixs]


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
        reg = model.C if hasattr(model, 'C') else 1
        F_ij += reg * np.diag(np.ones(F_ij.shape[0]))  # Prior Regularization
        return F_ij


simple_selectors = {
    'random':RandomSelector(),
    'uncertainty':UncertaintySelector(),
    'fisher': FisherSelector()
    }
