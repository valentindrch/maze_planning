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
data = {'goal': [], 'distance': [], 't': [], 'i': [], 'n_samples': []}
goal_to_distance = {0: 3, 1: 3, 2: 3, 3: 3, 4: 0, 5: 1, 6: 2, 7: 2}
traces_a0_0 = []
for goal in range(8):
    print(goal)

    def_goal(goal)
    #for i in [0, 5, 50]:
    for i in [5]:
        for j in range(1000):
            sampling_alg = Sampler(model, priors[i], threshold=1e-2)
            sampled_posterior, n = sampling_alg.query(None, evidence={'o3': 0}, max_samples=80)
            
            data['goal'].append(goal)
            data['distance'].append(goal_to_distance[goal])
            data['t'].append(i)
            data['i'].append(j)
            data['n_samples'].append(n)
            traces_a0_0.append(sampling_alg.trace.sum(axis=(2, 3, 4, 5, 6))[:, 0])

data = pd.DataFrame(data)

# At time point 5 plot how sure the algorithm is after a certain time
# Group traces by goal
traces_by_distance = {i: [] for i in range(4)}
for idx, distance in enumerate(data['distance']):
    traces_by_distance[distance].append(traces_a0_0[idx])

# Find max length across all traces
max_len = max(len(trace) for traces in traces_by_distance.values() for trace in traces)

# Pad traces with NaN
padded_traces_by_distance = {i: [] for i in range(8)}
for distance in range(4):
    for trace in traces_by_distance[distance]:
        padded_trace = np.pad(trace, 
                             (0, max_len - len(trace)), 
                             mode='constant', 
                             constant_values=np.nan)
        padded_traces_by_distance[distance].append(padded_trace)

# Calculate mean and std for each distance
plt.figure(figsize=(10, 6))
for distance in range(4):
    distance_traces = np.array(padded_traces_by_distance[distance])
    mean_trace = np.nanmean(distance_traces, axis=0)
    std_trace = np.nanstd(distance_traces, axis=0) / np.sqrt(len(distance_traces))
    
    # Plot mean line with std shade
    x = np.arange(len(mean_trace))
    plt.plot(x, mean_trace, label=f'distance {distance}')
    plt.fill_between(x, mean_trace-std_trace, mean_trace+std_trace, alpha=0.2)

plt.xlabel('Sample')
plt.ylabel('Probability')
plt.title('Evolution of a0=0 probability by goal')
plt.legend()
plt.tight_layout()
plt.show()

a = 1