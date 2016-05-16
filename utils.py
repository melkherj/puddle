import math
import numpy as np
import logging
from collections import Counter

def adjust_freq_sample(ixs,labels,cat_ps,n_samples=1):
    ''' given a list of indices, labels, and 
        the relative probability (not required to be normalized) of sampling
        an index with a particular label category, return n such bootstrap samples
        '''
    if len(ixs)==0:
        return []
    label_cs = Counter(labels)
    ps = np.array([float(cat_ps[lab])/label_cs[lab] for lab in labels])
    ps = ps/ps.sum()
    return [int(np.random.choice(ixs,p=ps)) for _ in range(n_samples)]

def sigmoid(z):
    return 1./(1.+math.exp(-z))

def logodds(p):
    return math.log(p/float(1-p))

def adjust(thresh,n_pos,n_neg,p=0.5,gamma=0.2,lower=0.001,upper=0.999):
    ''' thresh is current score threshold
        ratio is ratio of negatives seen so far to positives '''
    ratio = float(n_pos)/n_neg*(1-p)/p
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

def file_md5(fname):
    ''' Given the path to a file, return the md5 of that file '''
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def graph_hash(graphdef):
    ''' Get the hash of a tensorflow graphdef object.  spit out to file & hash file '''
    filename = "/tmp/temp_graph.pb"
    tf.train.write_graph(graphdef, ".logs", filename, True)
    return file_md5(filename)

def session_hash(sess):
    ''' get the hash of a tensorflow session, which contains model weights etc '''
    filename = '/tmp/temp_sess_model.ckpt'
    saver = tf.train.Saver()
    saver.save(sess,filename)
    return file_md5(filename)
