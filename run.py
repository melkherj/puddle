#!/usr/bin/env python
from active_evaluate import active_evaluate,append_results,load_conf
import random

# run some selector, see what performance is like, append to the results file
conf = load_conf()
for i in range(1000):
    conf['random_seed'] = i
    conf['selector']['active_thresh'] = 0.4*random.random()+0.3 #0.3->0.7
    results = active_evaluate(conf)
    append_results(results)

