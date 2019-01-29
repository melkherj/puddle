from .active_evaluate import active_evaluate, run_experiment
from .model_selectors.simple import simple_selectors
from .models.sklearn_simple import all_models
from .datasets.sklearn_datasets import classification_datasets
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import os
import logging
import multiprocessing
from functools import partial
# LOGGING
logging.basicConfig(filename='evaluation.log',level=logging.DEBUG)
logger = logging.getLogger(__name__)

# TODO identify these args in command line at run time
sample_size=10**6
epochs=50
labels_per_epoch = 5
n_ensemble=10
n_cpus=4
all_datasets = classification_datasets(downsample_size=sample_size)

# List of all experiments
experiments = itertools.product(
                                all_datasets.items(),
                                all_models.items(),
                                simple_selectors.items(),
                                range(n_ensemble)
                                )

logger.info("RUNNING EXPERIMENTS:")
# Run Experiments
all_metrics = []
file_init = True

pool = multiprocessing.Pool(n_cpus)
run_my_experiment = partial(run_experiment, epochs=epochs, labels_per_epoch=labels_per_epoch)
all_metrics = pool.map(run_my_experiment, experiments)
pool.close()
pool.join()
df = pd.concat(all_metrics)
df.to_csv('results/all_metrics.csv')


# Generate and Export Visualizations
# Piecewise plots
df = df.set_index(['dataset','model','selector'])
dataset_model_confs = list(set((conf[:2] for conf in df.index)))

# nuke all previous plots
for fname in os.listdir('results/plots'):
    os.remove(os.path.join('results/plots', fname))

html_overview = '<html><body><h1>Data Efficiency Results by Model/Dataset</h1>'
for ix in dataset_model_confs:
    dataset_name, model_name = ix
    df_selectors = df.loc[ix]
    selector_names = list(set(df_selectors.index))
    plt.figure(figsize=(7, 6))
    plt.title('dataset=%s model=%s' % (dataset_name, model_name))
    plt.xlabel('number of training labels')
    plt.ylabel('f1 score')
    #     plt.xlim([-0.1, 1.1])
    #     plt.ylim([-0.1, 1.1])
    ax_line_args = {'color': 'black', 'linewidth': 0.3, 'linestyle': 'dashed'}
    plt.axvline(x=0.0, **ax_line_args)
    plt.axvline(x=1.0, **ax_line_args)
    plt.axhline(y=0.0, **ax_line_args)
    plt.axhline(y=1.0, **ax_line_args)

    for selector_name in selector_names:
        df_experiment = df_selectors.loc[selector_name]
        agg = df_experiment.groupby(['train_size']).agg({'f1': ['mean', 'std']})
        agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
        agg = agg.reset_index()
        agg['f1_upper'] = agg["f1_mean"] + agg['f1_std'].fillna(0)
        agg['f1_lower'] = agg["f1_mean"] - agg['f1_std'].fillna(0)
        plt.fill_between(agg['train_size'], agg['f1_lower'], y2=agg['f1_upper'], label=selector_name, alpha=0.2)

    ax = plt.gca()
    leg = plt.legend(loc='upper right')
    bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
    # Change to location of the legend.
    xOffset = 0.2
    bb.x0 += xOffset
    bb.x1 += xOffset
    leg.set_bbox_to_anchor(bb, transform=ax.transAxes)
    fig_path = 'plots/%s-%s.png' % (dataset_name, model_name)
    plt.tight_layout()
    plt.savefig(os.path.join('results',fig_path))
    plt.clf()
    html_overview += '<h3>%s, %s</h3><img src="%s">'%(dataset_name,model_name,fig_path)
html_overview += '</body></html>'
with open('results/overview.html','w') as f:
    f.write(html_overview)

# overall plot
import seaborn as sns
df = df.reset_index()
agg = df.groupby(['dataset', 'model', 'selector', 'train_size']).agg({'f1':['mean', 'std']})
agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
agg['f1_upper'] = agg["f1_mean"]+agg['f1_std'].fillna(0)
agg['f1_lower'] = agg["f1_mean"]-agg['f1_std'].fillna(0)
agg = agg.reset_index()

plt.figure(figsize=(8,8))
g = sns.FacetGrid(col='dataset', hue='selector', col_wrap=3, data=agg)
g.map(plt.fill_between, 'train_size', 'f1_lower', 'f1_upper', alpha=0.4)
plt.legend(loc='upper right')
plt.savefig('results/plots/overall_plot.png')

