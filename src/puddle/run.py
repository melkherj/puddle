import numpy as np
from .active_evaluate import active_evaluate
from .model_selectors.simple import simple_selectors
from .models.sklearn_simple import all_models
from .datasets.sklearn_datasets import classification_datasets
import json
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import os

labels_per_epoch = 10

all_datasets = classification_datasets()

#(dataset,model,selector)
experiments = list(itertools.product(
    all_datasets.items(),
    all_models.items(),
    simple_selectors.items()))

# Run Experiments
all_metrics = []
for (dataset_name,(X,Y)),(model_name,model),(selector_name,selector) in experiments:
    print('evaluating dataset/model/selector: ',dataset_name,model_name,selector_name,'...')
    #TODO don't hardcode epochs and number of examples per epoch
    epochs = len(X)//labels_per_epoch
    metrics_by_epoch = active_evaluate(X,Y,model,selector,epochs,labels_per_epoch)
    for metric in metrics_by_epoch:
        metric['dataset'] = dataset_name
        metric['model'] = model_name
        metric['selector'] = selector_name
        all_metrics.append(metric)
df = pd.DataFrame(all_metrics)

# Generate and Export Visualizations
df.to_csv('results/all_metrics.csv')
df = df.set_index(['dataset','model','selector'])

dataset_model_confs = list(set((conf[:2] for conf in df.index)))
# nuke all previous plots
for fname in os.listdir('results/plots'):
    os.remove(os.path.join('results/plots',fname))
html_overview = '<html><body><h1>Data Efficiency Results by Model/Dataset</h1>'
for ix in dataset_model_confs:
    dataset_name,model_name = ix
    df_selectors = df.loc[ix]
    selector_names = list(set(df_selectors.index))
    plt.figure(figsize=(7,6))
    plt.title('dataset=%s model=%s'%(dataset_name,model_name))
    plt.xlabel('fraction of labels')
    plt.ylabel('f1 score')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    ax_line_args = {'color':'black','linewidth':0.3,'linestyle':'dashed'}
    plt.axvline(x=0.0,**ax_line_args)
    plt.axvline(x=1.0,**ax_line_args)
    plt.axhline(y=0.0,**ax_line_args)
    plt.axhline(y=1.0,**ax_line_args)
    for selector_name in selector_names:
        df_experiment = df_selectors.loc[selector_name]
        plt.plot(df_experiment['train_fraction'],df_experiment['f1'],label=selector_name)
    ax = plt.gca()
    leg = plt.legend(loc = 'upper right')
    bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
    # Change to location of the legend.
    xOffset = 0.2
    bb.x0 += xOffset
    bb.x1 += xOffset
    leg.set_bbox_to_anchor(bb, transform = ax.transAxes)
    fig_path = 'plots/%s-%s.png'%(dataset_name,model_name)
    plt.tight_layout()
    plt.savefig(os.path.join('results',fig_path))
    plt.clf()
    html_overview += '<h3>%s, %s</h3><img src="%s">'%(dataset_name,model_name,fig_path)
html_overview += '</body></html>'
with open('results/overview.html','w') as f:
    f.write(html_overview)
