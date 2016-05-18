#!/usr/bin/env python
from active_evaluate import active_evaluate,append_results,load_conf
import random

# run some selector, see what performance is like, append to the results file
conf = load_conf()
conf['random_seed'] = 0
#for _ in range(1000):
#    conf['active_selector']['
#for semisup_selector in ['empty']: #['top','empty','random']:
#    conf['semisup_selector']['name'] = semisup_selector
#    for active_selector in ['random']: #['adaptive','uncertain','random']:
#        conf['active_selector']['name'] = active_selector
results = active_evaluate(conf)
append_results(results)

