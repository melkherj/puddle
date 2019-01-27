import numpy as np
from sklearn.datasets import load_boston, load_iris, fetch_kddcup99, fetch_20newsgroups_vectorized
from sklearn.datasets.california_housing import fetch_california_housing
import os, ssl
import random

# There's a bug on macs, this works around it
# https://github.com/scikit-learn/scikit-learn/issues/10201
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

def sklearn_binarize_by_percentile(dat,percentile=50,downsample_size=None):
    target = dat['target']
    thresh = np.percentile(target,percentile)
    X = dat['data']
    Y = (dat['target']>thresh).astype(np.int)
    if downsample_size:
        ixs = random.sample(range(len(Y)),downsample_size)
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


def classification_datasets():
    datasets = {
#        'iris': sklearn_binarize_by_percentile(load_iris()),
#        'kdd99_tiny':kdd99_portsweep_tiny(),
        'boston': sklearn_binarize_by_percentile(load_boston()),
        'california_housing': sklearn_binarize_by_percentile(fetch_california_housing(),
            downsample_size=1000),
        '20newsgroups_vectorized': sklearn_binarize_by_percentile(fetch_20newsgroups_vectorized(),
            downsample_size=4000)
        }
    return datasets
