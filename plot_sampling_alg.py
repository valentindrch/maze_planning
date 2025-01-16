import matplotlib.pyplot as plt
import pickle
import numpy as np
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
def_goal(4)

n_samples = []
for i in [0, 5, 50]:
    print(i)

    n_samples_i = []
    for j in range(10):
        sampling_alg = Sampler(model, priors[i])
        sampled_posterior, n = sampling_alg.query(None, evidence={'o3': 0})
        n_samples_i.append(n)

    n_samples.append(n_samples_i)

a = 1