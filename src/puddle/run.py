import numpy as np
from .active_evaluate import active_evaluate
from .model_selectors.simple import simple_selectors
from .models.sklearn_simple import all_models
from .datasets.sklearn_datasets import classification_datasets
import json
import pandas as pd
import itertools

all_datasets = classification_datasets()

#(dataset,model,selector)
experiments = itertools.product(
    all_datasets.items(),
    all_models.items(),
    simple_selectors.items())

all_metrics = []
for (dataset_name,(X,Y)),(model_name,model),(selector_name,selector) in experiments:
    print('evaluating dataset/model/selector: ',dataset_name,model_name,selector_name,'...')
    metrics_by_epoch = active_evaluate(X,Y,model,selector,5,10)
    for metric in metrics_by_epoch:
        metric['dataset'] = dataset_name
        metric['model'] = model_name
        metric['selector'] = selector_name
        all_metrics.append(metric)
df = pd.DataFrame(metrics_by_epoch)

