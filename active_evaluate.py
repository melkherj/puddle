import sys,os 
from mnist_modeler import MNISTModeler
import numpy as np
import json
from selectors_registry import selectors_registry
import random
from subprocess import check_output

def append_results(results,filename='results/mnist'):
    # append to active_results
    with open(filename,'a') as f:
        f.write(json.dumps(results)+'\n')
        
def load_conf(filename='active_conf.json'):
    # which selector, for how many iterations, what dataset sizes, etc?
    with open(filename,'r') as f:
        conf = json.load(f)
    return conf

def active_evaluate(conf):
    ################################################################################################
    # git version
    #repo must be clean, since we're using git hash
    if check_output(['git','diff'])=='': 
        git_hash = check_output(['git','rev-parse','--short','HEAD']).strip()
    else:
        git_hash = None

    ################################################################################################
    # get the selector class we'll be using for active selection
    SelectorClass = selectors_registry[conf['selector_name']]

    ################################################################################################
    # get variables needed for active learning system
    mnm = MNISTModeler(train_n=conf['train_size'],test_n=conf['test_size'],
            seed=conf['random_seed'])
    selector = SelectorClass(n_ixs=conf['batch_size'],seed=conf['random_seed'])
    state = []
    Y = np.ones(len(mnm.train_Y))*-1
    accuracies = []
    I = []
    
    ################################################################################################
    ### simulate active labeling, track accuracy
    for i in range(conf['epochs']):
        print 'learning/evaluating on batch %d'%i
        # get next indices
        I_next = selector.next_indices(Y,state,mnm)
        # label/set state
        for ix,label_row in zip(I_next,mnm.train_Y[I_next]):
            y_label = int(np.nonzero(label_row)[0][0])
            state.append((ix,y_label))
            Y[ix] = y_label
        # update model
        x = mnm.train_X[I_next]
        y = mnm.train_Y[I_next]
        mnm.update_model(x,y)
        # evaluate accuracy
        accuracy = float(mnm.test_accuracy())
        accuracies.append( accuracy )
        print 'accuracy: %.3f'%accuracy
        sys.stdout.flush()
       
    
    ################################################################################################
    ### get results in a format emenable to saving
    results = conf.copy()
    results['accuracies'] = accuracies
    results['final_accuracy'] = accuracies[-1]
    results['githash'] = git_hash
    
    return results
