import logging
import numpy as np
import matplotlib.pyplot as plt

class Modeler(object):
    def __init__(self,mnm):
        self.logger = logging.getLogger('active_semisup.M')
        self.logger.info('Setting up M')
        self.Is = [] #distinct id's used so far
        self.I_sequence = [] #sequenceof mini-batches trained on.  so we can exactly reproduce results
        self.accuracies = []
        self.mnm = mnm
        self.Y = np.argmax(mnm.train_Y,axis=1)
        self.n,self.d = mnm.train_Y.shape
        self.score_train()
        self.logger.info('Done setting up M')
    
    def update(self,I,score_train=True,calculate_accuracy=True,semisup=[]):
        I = map(int,list(I))
        self.I_sequence.append(I)
        self.Is = sorted(list(set(I)|set(self.Is))) #all indices requested to label so far
        I = I+semisup
        x = self.mnm.train_X[I]
        y = self.mnm.train_Y[I]
        if len(semisup)>0:
            y[-len(semisup):] = self.Yp_sparse[semisup,:] #use fake-labels for semi-supervised
        # replace labels with pseudo-labels for semi-supervised
        self.mnm.update_model(x,y,score_train=score_train)
        if calculate_accuracy:
            self.test_accuracy()
            
    def test_accuracy(self):
        accuracy = self.mnm.test_accuracy()
        self.accuracies.append((len(self.Is),accuracy))
        logger.info('with %d labels total, %d added, accuracy %.3f'%(len(self.Is),len(I),accuracy))
        return accuracy
        
    def score_train(self):
        mnm.score_train()
        self.Yp = np.argmax(mnm.P,axis=1) #pick most likely label for each category
        self.Yp_sparse = np.zeros((self.n,self.d))
        for i,j in zip(range(self.n),self.Yp): #create a sparse representation of Yp
            self.Yp_sparse[i,j] = 1.0

    def topcat(self,category,k=10):
        ''' top <k> indices from the given category'''
        # Select top ranked indices j->j+10 for given category
        cat_sort = map(int,list(np.argsort(self.mnm.P[:,category])))
        return cat_sort[-k:]

    def viz_train(self,i):
        x = self.mnm.train_X[i,:]
        self.logger.info('index: %d,label: %d, predicted: %d'%(i,self.Y[i],self.Yp[i]))
        plt.imshow(x.reshape(28,28), interpolation='nearest',cmap='Greys')
        plt.show()
