import numpy as np
from sklearn.datasets import load_boston, load_iris

def sklearn_binarize_by_percentile(dat,percentile=50):
    target = dat['target']
    thresh = np.percentile(target,percentile)
    X = dat['data']
    Y = (dat['target']>thresh).astype(np.int)
    return X,Y

def classification_datasets():
    datasets = {
        'boston': sklearn_binarize_by_percentile(load_boston()),
        'iris': sklearn_binarize_by_percentile(load_iris())
        }
    return datasets
