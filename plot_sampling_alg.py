import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from pgmpy.factors.discrete import TabularCPD

from sampling import Sampler
from model import PlanningModel

# Load priors from pickle file
with open('./data_files/priors.pkl', 'rb') as f:
    priors = pickle.load(f)

# Sample for different goals
if False:
    data = {'goal': [], 't': [], 'i': [], 'n_samples': [], 'accuracy': [], 'surprise': [], 'cost': [], 'error': []}
    traces_a0_0 = []

    model = PlanningModel(alpha=3, rho=.95)

    for goal in range(8):
        print(goal)

        for i in [0, 5, 50]:
            for j in range(100):

                model.sample(goal, threshold=1e-2, prior=priors[i])
                cost, error, surprise =  model.get_it_measures()

                p_a = model.posterior.sum(axis=(1, 3, 5)).ravel()[goal]

                data['goal'].append(goal)
                data['t'].append(i)
                data['i'].append(j)
                data['n_samples'].append(model.n)
                data['accuracy'].append(p_a)
                data['surprise'].append(surprise)
                data['cost'].append(cost)
                data['error'].append(error)

    data = pd.DataFrame(data)
    data.to_csv('./data_files/sampling_illustration_results.csv')
else:
    data = pd.read_csv('./data_files/sampling_illustration_results.csv')

# Create figure with three subplots
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(15, 8))

# Plot # samples for each t
for t, ax in zip([0, 5, 50], [ax1, ax2, ax3]):
    # Get data for this t and calculate mean
    t_data = data[data['t'] == t].groupby('goal')['n_samples']
    means = t_data.mean()
    sems = t_data.std()
    
    # Create bar plot
    ax.bar(means.index, means.values, yerr=sems.values, capsize=5)
    ax.set_ylabel('Number of samples')
    ax.set_ylim((0, 3100))

# Plot accuracy for each t
for t, ax in zip([0, 5, 50], [ax4, ax5, ax6]):
    # Get data for this t and calculate mean
    t_data = data[data['t'] == t].groupby('goal')['accuracy']
    means = t_data.mean()
    sems = t_data.std()
    
    # Create bar plot
    ax.bar(means.index, means.values, yerr=sems.values, capsize=5)
    ax.set_ylabel('Accuracy')
    ax.set_ylim((0, 1))

# Plot surprise for each t
for t, ax in zip([0, 5, 50], [ax7, ax8, ax9]):
    # Get data for this t and calculate mean
    t_data = data[data['t'] == t].groupby('goal')['surprise']
    means = t_data.mean()
    sems = t_data.std()
    
    # Create bar plot
    ax.bar(means.index, means.values, yerr=sems.values, capsize=5)
    ax.set_xlabel('Goal')
    ax.set_ylabel('Surprise')
    ax.set_ylim((0, 3))

plt.tight_layout()
plt.show()

fig.savefig('./figs/sampling_alg.pdf')

a = 1