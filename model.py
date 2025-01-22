from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import numpy as np

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

model = BayesianNetwork([
        ('a0', 's1'),
        ('s1', 'a1'),
        ('s1', 's2'),
        ('a1', 's2'),
        ('s2', 'a2'),
        ('s2', 's3'),
        ('a2', 's3'),
        ('s3', 'o3')
    ])

param = 3.0
# Initialize the Dirichlet priors
alpha_a0 = Dirichlet(shape=(2, 1), params=np.array([[param], [param]]))  # Dirichlet prior for a0
alpha_a1 = Dirichlet(shape=(2, 2), params=np.array([[param, param], [param, param]]))  # Dirichlet prior for a1
alpha_a2 = Dirichlet(shape=(2, 4), params=np.array([[param, param, param, param], [param, param, param, param]]))  # Dirichlet prior for a2
# P(a0)
cpd_a0 = TabularCPD(variable='a0', 
                    variable_card=2, 
                    values=alpha_a0.get_MAP_cpd())
# P(s1 | a0)
cpd_s1_given_a0 = TabularCPD(
    variable='s1',
    variable_card=2,
    evidence=['a0'],
    evidence_card=[2],
    values=[
        [0.99, 0.01],  
        [0.01, 0.99]   
    ]
)
# P(a1 | s1)
cpd_a1_given_s1 = TabularCPD(
    variable='a1',
    variable_card=2,
    evidence=['s1'],
    evidence_card=[2],
    values=alpha_a1.get_MAP_cpd()
)
# P(s2 | s1, a1)
cpd_s2_given_s1_a1 = TabularCPD(
    variable='s2',
    variable_card=4,
    evidence=['s1', 'a1'],
    evidence_card=[2, 2],
    values=[
        [0.99, 0.01, 0.0, 0.0],  
        [0.01, 0.99, 0.0, 0.0],  
        [0.0, 0.0, 0.99, 0.01],  
        [0.0, 0.0, 0.01, 0.99]   
    ]
)
# P(a2 | s2)
cpd_a2_given_s2 = TabularCPD(variable='a2', 
                            variable_card=2,
                            evidence=['s2'],
                            evidence_card=[4],
                            values=alpha_a2.get_MAP_cpd())
# P(s3 | s2, a2) with 8 states in s3
cpd_s3_given_s2_a2 = TabularCPD(
    variable='s3',
    variable_card=8,
    evidence=['s2', 'a2'],
    evidence_card=[4, 2],
    values=[
        [0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  
        [0.01, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  
        [0.0, 0.0, 0.99, 0.01, 0.0, 0.0, 0.0, 0.0],  
        [0.0, 0.0, 0.01, 0.99, 0.0, 0.0, 0.0, 0.0],  
        [0.0, 0.0, 0.0, 0.0, 0.99, 0.01, 0.0, 0.0],  
        [0.0, 0.0, 0.0, 0.0, 0.01, 0.99, 0.0, 0.0],  
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99, 0.01],  
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.99]   
    ]
)
model.add_cpds(cpd_a0, cpd_s1_given_a0, cpd_a1_given_s1, cpd_s2_given_s1_a1, cpd_a2_given_s2, cpd_s3_given_s2_a2)

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