#!/usr/bin/env python
from active_evaluate import active_evaluate,append_results,load_conf

# run some selector, see what performance is like, append to the results file
conf = load_conf()
results = active_evaluate(conf)
append_results(results)
