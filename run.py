#!/usr/bin/env python
from active_evaluate import active_evaluate,append_results,load_conf
import random

# run some selector, see what performance is like, append to the results file
conf = load_conf()


confs = [
        ('adaptive','random'),
        ('random','top'),
#        ('uncertain','top'),
#        ('random','random'),
#        ('random','empty'),
        ]
for active,semisup in confs:
    conf['active_selector']['name'] = active
    conf['semisup_selector']['name'] = semisup
    results = active_evaluate(conf)
    append_results(results)
    
#for categories in [[3,5],range(10)]:
#    for active_thresh in [0.3,0.5,0.7]:
#        for 

#for _ in range(1000):
#    conf['active_selector']['
#for semisup_selector in ['empty']: #['top','empty','random']:
#    conf['semisup_selector']['name'] = semisup_selector
#    for active_selector in ['random']: #['adaptive','uncertain','random']:
#        conf['active_selector']['name'] = active_selector

