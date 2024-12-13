from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation
import pandas as pd
import numpy as np

# Define Dirichlet distribution
class Dirichlet():

    def __init__(self, shape=None, params=None):
        if params is None:
            self.params = np.ones(shape)  # for simplicity we start with all parameters set to 1 (uniform)
        else:
            self.params = params

    def infer(self, observations):
        if self.params.shape != observations.shape:  # in case of multiple observations
            self.params += observations.sum(axis=0)
        else:
            self.params += observations  # this is the very basic update function to get the posterior

    def get_MAP_cpd(self):
        return self.params / self.params.sum(axis=0)  # it's also very simple to get a point estimate for the prior over A

    def get_full_cpd(self):  # the MAP (point estimate) is a simplification. When considering the full distribution it gets more complicated...
        pass

# Create network structure
model = BayesianNetwork([('A', 'B')])
alpha = Dirichlet(shape=(2, 1))
cpd_A = TabularCPD('A', 2, alpha.get_MAP_cpd())
cpd_B = TabularCPD('B', 2, [[.7, .3],  # I.e., in 70% of cases A=0 leads to B=0 - this is basically the statistical information that we want to learn with the prior
                            [.3, .7]],
                    ['A'], [2])
model.add_cpds(cpd_A, cpd_B)

# Infer posterior p(A | B)
inference = BeliefPropagation(model)
posterior_A = inference.query(['A'], evidence={'B': 0})  # we always observe B=0 (like "reward")

# Infer hyperprior p(alpha)
alpha.infer(posterior_A.values.reshape(2, 1))
model.add_cpds(TabularCPD('A', 2, alpha.get_MAP_cpd()))  # new, learned prior over A based on previous inference 
print(f'Hyperparameters: {alpha.params}')
print(f'New prior over A: {alpha.get_MAP_cpd()}')
