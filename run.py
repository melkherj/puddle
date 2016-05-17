#!/usr/bin/env python
from active_evaluate import active_evaluate,append_results,load_conf
import random

# run some selector, see what performance is like, append to the results file
conf = load_conf()
conf['random_seed'] = 0
#for semisup_selector in ['random','top']
#for active_selector in ['adaptive','random']:
#    conf['active_selector']['name'] = active_selector
    
for i in range(1000):
    results = active_evaluate(conf)
    append_results(results)

