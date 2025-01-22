import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from pgmpy.factors.discrete import TabularCPD

from sampling import Sampler
from model import model, def_goal

# Load priors from pickle file
with open('priors.pkl', 'rb') as f:
    priors = pickle.load(f)

# Sample for different goals
data = {'goal': [], 't': [], 'i': [], 'n_samples': [], 'accuracy': []}
traces_a0_0 = []
for goal in range(8):
    print(goal)

    def_goal(goal)
    for i in [0, 5, 50]:
        for j in range(100):
            sampling_alg = Sampler(model, priors[i], threshold=1e-2)
            sampled_posterior, n = sampling_alg.query(None, evidence={'o3': 0})

            p_a = sampled_posterior.sum(axis=(1, 3, 5)).ravel()[goal]
            
            data['goal'].append(goal)
            data['t'].append(i)
            data['i'].append(j)
            data['n_samples'].append(n)
            data['accuracy'].append(p_a)

data = pd.DataFrame(data)

# Create figure with three subplots
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 8))

# Plot # samples for each t
for t, ax in zip([0, 5, 50], [ax1, ax2, ax3]):
    # Get data for this t and calculate mean
    t_data = data[data['t'] == t].groupby('goal')['n_samples']
    means = t_data.mean()
    sems = t_data.std()
    
    # Create bar plot
    ax.bar(means.index, means.values, yerr=sems.values, capsize=5)
    ax.set_xlabel('Goal')
    ax.set_ylabel('Number of samples')
    ax.set_title(f't = {t}')
    ax.set_ylim((0, 3100))

# Plot accuracy for each t
for t, ax in zip([0, 5, 50], [ax4, ax5, ax6]):
    # Get data for this t and calculate mean
    t_data = data[data['t'] == t].groupby('goal')['accuracy']
    means = t_data.mean()
    sems = t_data.std()
    
    # Create bar plot
    ax.bar(means.index, means.values, yerr=sems.values, capsize=5)
    ax.set_xlabel('Goal')
    ax.set_ylabel('Accuracy')
    ax.set_title(f't = {t}')
    ax.set_ylim((0, 1))

plt.tight_layout()
plt.show()

a = 1