from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation
import pandas as pd
import numpy as np

# Define Dirichlet distribution
class Dirichlet():

    def __init__(self, shape=None, params=None):
        if params is None:
            self.params = np.ones(shape)
        else:
            self.params = params

    def infer(self, observations):
        if self.params.shape != observations.shape:
            self.params += observations.sum(axis=0)
        else:
            self.params += observations

    def get_MAP_cpd(self):
        return self.params / self.params.sum(axis=0)

    def get_full_cpd(self):
        pass

# Create network structure
model = BayesianNetwork([('A', 'B')])
alpha = Dirichlet(shape=(2, 1))
cpd_A = TabularCPD('A', 2, alpha.get_MAP_cpd())
cpd_B = TabularCPD('B', 2, [[.7, .3],
                            [.3, .7]],
                    ['A'], [2])
model.add_cpds(cpd_A, cpd_B)

# Infer posterior p(A | B)
inference = BeliefPropagation(model)
posterior_A = inference.query(['A'], evidence={'B': 0})

# Infer hyperprior p(alpha)
alpha.infer(posterior_A.values.reshape(2, 1))
model.add_cpds(TabularCPD('A', 2, alpha.get_MAP_cpd()))  # new, learned prior over A based on previous inference 
print(f'Hyperparameters: {alpha.params}')
print(f'New prior over A: {alpha.get_MAP_cpd()}')
