import numpy as np
from sklearn.datasets import load_boston, load_iris, fetch_kddcup99, fetch_20newsgroups_vectorized, load_breast_cancer
from sklearn.datasets.california_housing import fetch_california_housing
import os, ssl
import random
from sklearn.datasets import make_classification

# There's a bug on macs, this works around it
# https://github.com/scikit-learn/scikit-learn/issues/10201
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

def sklearn_binarize_by_percentile(dat,percentile=50,downsample_size=None):
    thresh = np.percentile(dat['target'],percentile)
    X = dat['data']
    Y = (dat['target']>thresh).astype(np.int)
    if downsample_size:
        ixs = random.sample(range(len(Y)), min(downsample_size, X.shape[0]))
        X = X[ixs]
        Y = Y[ixs]
    return X,Y

def breast_cancer(downsample_size=None):
    res =load_breast_cancer()
    X,Y = res["data"], res["target"]
    if downsample_size:
        ixs = random.sample(range(len(Y)), min(downsample_size, X.shape[0]))
        X = X[ixs]
        Y = Y[ixs]
    return X,Y


def kdd99_portsweep_tiny(size=10000):
    #TODO make better based on actual studies, not tiny and arbitrary labels
    dat = fetch_kddcup99()
    X = dat['data']
    Y = (dat['target']=='portsweep').astype(np.int)
    portsweep_ixs = np.nonzero(Y==1)[0]
    ixs = list(set(random.sample(range(len(X)),size))|set(portsweep_ixs))
    return X[ixs],Y[ixs]

def sklearn_classification(downsample_size=10**4, weights=None):
    X, Y = make_classification(n_samples=downsample_size, n_features=20,
                               n_informative=4, n_redundant=3,
                               n_clusters_per_class=2, weights=weights)
    return X, Y

def classification_datasets(downsample_size=None):
    """

    :param downsample_size:
        The maximum number of points to
    :return: (dict)
        Dictionary of datasetnames (keys) and X,Y training set (values)
    """
    datasets = {
#        'iris': sklearn_binarize_by_percentile(load_iris()),
#        'kdd99_tiny':kdd99_portsweep_tiny(),
        'skewed_sklearn_10_to_1': sklearn_classification(downsample_size=downsample_size, weights=[0.9, 0.1]),
        'skewed_sklearn_20_to_1': sklearn_classification(downsample_size=downsample_size, weights=[0.95, 0.05]),
        # 'breast_cancer':breast_cancer(downsample_size=downsample_size),
        # 'boston': sklearn_binarize_by_percentile(load_boston(), downsample_size=downsample_size),
         'california_housing': sklearn_binarize_by_percentile(fetch_california_housing(),
             downsample_size=downsample_size),
         '20newsgroups_vectorized': sklearn_binarize_by_percentile(fetch_20newsgroups_vectorized(),
             downsample_size=downsample_size)
        }
    return datasets
