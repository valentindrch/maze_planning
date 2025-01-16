import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from pgmpy.factors.discrete import TabularCPD

from sampling import Sampler
from model import model

# Load priors from pickle file
with open('priors.pkl', 'rb') as f:
    priors = pickle.load(f)

# Create function to change goal
def def_goal(index, value=.95):

    # Make cpd array
    arr = np.zeros((2, 8))
    arr[0, :] = 1 - value
    arr[0, index] = value
    arr[1, :] = value
    arr[1, index] = 1 - value

    # Add to model
    cpd_o3_given_s3 = TabularCPD(
        variable='o3',
        variable_card=2,
        evidence=['s3'],
        evidence_card=[8],
        values=arr,
    )
    model.add_cpds(cpd_o3_given_s3)

# Sample for different goals
data = {'goal': [], 't': [], 'i': [], 'n_samples': []}
for goal in range(8):
    print(goal)

    def_goal(goal)
    for i in [0, 5, 50]:
        for j in range(20):
            sampling_alg = Sampler(model, priors[i])
            sampled_posterior, n = sampling_alg.query(None, evidence={'o3': 0})
            
            data['goal'].append(goal)
            data['t'].append(i)
            data['i'].append(j)
            data['n_samples'].append(n)

data = pd.DataFrame(data)

# Create figure with three subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Plot for each t
for t, ax in zip([0, 5, 50], [ax1, ax2, ax3]):
    # Get data for this t and calculate mean
    t_data = data[data['t'] == t].groupby('goal')['n_samples'].mean()
    
    # Create bar plot
    ax.bar(t_data.index, t_data.values)
    ax.set_xlabel('Goal')
    ax.set_ylabel('Number of samples')
    ax.set_title(f't = {t}')

plt.tight_layout()
plt.show()

a = 1