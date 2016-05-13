import math
import numpy as np
import logging

def sigmoid(z):
    return 1./(1.+math.exp(-z))

def logodds(p):
    return math.log(p/float(1-p))

def adjust(thresh,n_pos,n_neg,gamma=0.2,lower=0.001,upper=0.999):
    ''' thresh is current score threshold
        ratio is ratio of negatives seen so far to positives '''
    ratio = float(n_pos)/n_neg
    thresh = sigmoid(logodds(thresh)-gamma*math.log(ratio))
    return max(min(thresh,upper),lower)

def add_noise(thresh,variance=0.05):
    return sigmoid(logodds(thresh)+variance*float(np.random.randn()))

# set up logger
def setup_logging(loggername='active_semisup',logfile='active_semisup.log'):
    logger = logging.getLogger(loggername)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
