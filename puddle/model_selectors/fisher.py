# from sklearn.linear_model import LogisticRegression
# import numpy as np
# import scipy.stats as stat
# from scipy.sparse import issparse
# import logging
# logger = logging.getLogger(__name__)
#
#
# class ActiveLearningLogisticRegression(LogisticRegression):
#     """ Wrapper class for scikit-learn's Logistic Regression classifier.
#     New Attributes:
#     ---------------
#     coef_sigma_: array, shape (1, n_features)
#         Expected Standard Deviation / confidence intervals of coefficients in
#         the decision function
#     coef_z_scores_: array, shape (1, n_features)
#         Z-scores (z = beta/sigma_sigma) for the coefficients in decision function
#     coef_p_values_: array, shape (1, n_features)
#         One-sided p-value estimates for coefficients
#     F_ij_:  array, shape(n_feature, n_features)
#         Fisher information matrix, represented as the hessian d/dx_i d/dx_j
#         of the log likelihood
#     Methods:
#     -------------
#     This class exposes an expected information gain for some new data
#     matrix of examples as well, through the following method:
#     self.eig(X)
#     The returned value is the difference in Fisher information between
#     prior (fitted values) and new values
#     H(X_new + X_train) >= H(X_old)
#     matrix (poster = prior + new_data X)
#     """
#
#     def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
#                  fit_intercept=True, intercept_scaling=1, class_weight=None,
#                  random_state=None, solver='lbfgs', max_iter=100,
#                  multi_class='ovr', verbose=0, warm_start=False, n_jobs=None):
#         """ Logistic Regression Wrapper - See Class Docstring """
#         super(ActiveLearningLogisticRegression, self).__init__(penalty=penalty, dual=dual, tol=tol, C=C,
#                                                                fit_intercept=fit_intercept,
#                                                                intercept_scaling=intercept_scaling,
#                                                                class_weight=class_weight,
#                                                                random_state=random_state, solver=solver,
#                                                                max_iter=max_iter,
#                                                                multi_class=multi_class, verbose=verbose,
#                                                                warm_start=warm_start, n_jobs=n_jobs)
#
#     def fit(self, X, y, fisher=True, pvalues=True):
#         """ Sklearn class fit wrapper - appends fisher information and p-value calculation """
#         super(ActiveLearningLogisticRegression, self).fit(X, y)
#         # Get p-values for the fitted model #
#         self.F_ij_ = self._get_fisher_information_matrix(X) if fisher else None
#         self.fisher_info = np.log(np.linalg.det(self.F_ij_)) if fisher else None
#         self.coef_p_values_, self.coef_sigma_, self.coef_z_scores_ = self._get_p_values(self.F_ij_) if pvalues else None
#
#     def _get_p_values(self, F_ij):
#         """ Use Cramer Rao Bound to calculate p-values on regression coefficients """
#         Cramer_Rao = np.linalg.inv(F_ij)  ## Inverse Information Matrix
#         sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
#         z_scores = self.coef_[0] / sigma_estimates  # z-score for each model coefficient
#         p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]  ### two tailed test for p-values
#         return p_values, sigma_estimates, z_scores
#
#     def _get_fisher_information_matrix(self, X):
#         """ Calculate Fisher Information Matrix for sample data X """
#         if issparse(X):
#             X = X.todense()
#         if len(X.shape) < 2:
#             X = np.matrix(X)  # N by D matrix
#
#         D_ii = 2.0 * (1.0 + np.cosh(self.decision_function(X)))  # N array
#         D_ii2 = (X / D_ii[:, np.newaxis]).T
#         F_ij = np.dot(D_ii2, X)
#         F_ij += self.C * np.diag(np.ones(F_ij.shape[0]))  # Prior Regularization
#         if F_ij.shape[0] != F_ij.shape[1]:
#             raise ValueError(
#                 'F_ij is not square matrix. F_ij shape {}, X shape {}, D_ii shape {}, D_ii2 shape {}'.format(F_ij.shape,
#                                                                                                              X.shape,
#                                                                                                              D_ii.shape,
#                                                                                                              D_ii2))
#         return F_ij
#
#         # error: F_ij is not square matrix. F_ij shape (24, 30), X shape (30, 24), D_ii shape (30,)
#
#     def eig(self, X, axis=None):
#         """Expected information gain for sample X
#
#         Parameters:
#         -----------
#         X : numpy array, numpy matrix or pandas dataframe
#             Feature dimensions must match the fitted training set for this model.
#
#         When axis=1, will returned PER new sample x, not for the entire
#         data matrix.
#         """
#         if not hasattr(self, 'coef_'):
#             raise ValueError("Please call fit before estimating information gain on new samples")
#
#         if not axis:
#             new_F_ij = self._get_fisher_information_matrix(X)
#             new_info = np.log(np.linalg.det(self.F_ij_ + new_F_ij))
#             return new_info - self.fisher_info
#         elif axis == 1:
#             matrices = [self._get_fisher_information_matrix(x) for x in X]
#             return [np.log(np.linalg.det(self.F_ij_ + M)) - self.fisher_info for M in matrices]
#         else:
#             raise ValueError("axis must be None or 1")
#
#
#     def next_indices(self, X, ixs, Y_labeled, model, n_ixs=1):
#         """
#
#         :param X: (array) Pool of feature vectors
#         :param ixs: (list(n)) List of formerly Labeled training indices
#         :param Y_sub: (array(n,2)) Array of former labels
#         :param model: Model class that learning to label
#         :param n_ixs: The number of 'steps ahead' to look
#         :return:
#         """
#         logger.info("Fitting model with {} examples".format(len(ixs)))
#         self.fit(X[ixs], Y_labeled)
#         logger.info("Calculating information vector...".format(len(ixs)))
#         info_vector = self.eig(X, axis=1)
#         logger.info("Information vector type {}, shape {}".format(type(info_vector)), info_vector.shape)
#
#         chosen_ixs = np.argsort(info_vector)[:n_ixs]
#         return list(set(chosen_ixs)-set(ixs))
#
#
