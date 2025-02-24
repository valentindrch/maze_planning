import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation
from scipy.stats import entropy

from sampling import Sampler

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

class PlanningModel():

    def __init__(self, alpha, rho, kappa):
        self.alpha = alpha  # learning rate
        self.rho = rho  # focus/planning utility
        self.kappa = kappa # max reward focus utility
        self.model = BayesianNetwork([
            ('a0', 's1'),
            ('s1', 'r1'),
            ('s1', 'a1'),
            ('s1', 's2'),
            ('a1', 's2'),
            ('s2', 'r2'),
            ('s2', 'a2'),
            ('s2', 's3'),
            ('a2', 's3'),
            ('s3', 'r3'),
            ('s3', 'o3')
        ])
        self.node_list = [node for node in self.model.nodes if node not in ['r1', 'r2', 'r3', 'o3']]

        # Initialize the Dirichlet priors
        self.alpha_a0 = Dirichlet(shape=(2, 1), params=np.array([[self.alpha], [self.alpha]]))  # Dirichlet prior for a0
        self.alpha_a1 = Dirichlet(shape=(2, 2), params=np.array([[self.alpha, self.alpha], 
                                                                 [self.alpha, self.alpha]]))  # Dirichlet prior for a1
        self.alpha_a2 = Dirichlet(shape=(2, 4), params=np.array([[self.alpha, self.alpha, self.alpha, self.alpha], 
                                                                 [self.alpha, self.alpha, self.alpha, self.alpha]]))  # Dirichlet prior for a2
        # P(a0)
        cpd_a0 = TabularCPD(variable='a0', 
                            variable_card=2, 
                            values=self.alpha_a0.get_MAP_cpd())
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
        # P(s1 | r1)
        cpd_r1_given_s1 = TabularCPD(
            variable='r1',
            variable_card=2,
            evidence=['s1'],
            evidence_card=[2],
            values=[
                [0.5, 0.5],  
                [0.5, 0.5]   
            ]
        )
        # P(a1 | s1)
        cpd_a1_given_s1 = TabularCPD(
            variable='a1',
            variable_card=2,
            evidence=['s1'],
            evidence_card=[2],
            values=self.alpha_a1.get_MAP_cpd()
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
        # P(s2 | r2)
        cpd_r2_given_s2 = TabularCPD(
            variable='r2',
            variable_card=2,
            evidence=['s2'],
            evidence_card=[4],
            values=[
                [0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5] 
            ]
        )
        # P(a2 | s2)
        cpd_a2_given_s2 = TabularCPD(variable='a2', 
                                    variable_card=2,
                                    evidence=['s2'],
                                    evidence_card=[4],
                                    values=self.alpha_a2.get_MAP_cpd())
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
        # P(s3 | r3)
        cpd_r3_given_s3 = TabularCPD(
            variable='r3',
            variable_card=2,
            evidence=['s3'],
            evidence_card=[8],
            values=[
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            ]           
        )
        self.model.add_cpds(cpd_a0, cpd_s1_given_a0, cpd_a1_given_s1, cpd_s2_given_s1_a1, cpd_a2_given_s2, cpd_s3_given_s2_a2, cpd_r1_given_s1, cpd_r2_given_s2, cpd_r3_given_s3)

        # Create function to change goal
    
    def def_goal(self, index, maxima):
    
        # Make cpd array
        arr = np.zeros((2, 8))
        arr[0, :] = 1 - self.rho
        arr[0, index] = self.rho
        arr[1, :] = self.rho
        arr[1, index] = 1 - self.rho

        # Add to observational variable to model
        cpd_o3_given_s3 = TabularCPD(
            variable='o3',
            variable_card=2,
            evidence=['s3'],
            evidence_card=[8],
            values=arr,
        )
        self.model.add_cpds(cpd_o3_given_s3)

        # make cpd arrays for r1, r2, r3
        r1 = [[0.5, 0.5], [0.5, 0.5]]
        r2 = [[0.5] * 4, [0.5] * 4]
        r3 = [[0.5] * 8, [0.5] * 8]
        
        for state in maxima:
            if state in ['l', 'r']:
                r1[0][0 if state == 'l' else 1] = self.kappa
                r1[1][0 if state == 'l' else 1] = 1 - self.kappa
            elif state in ['ll', 'lr', 'rl', 'rr']:
                idx = ['ll', 'lr', 'rl', 'rr'].index(state)
                r2[0][idx] = self.kappa
                r2[1][idx] = 1 - self.kappa
            elif state in ['lll', 'llr', 'lrl', 'lrr', 'rll', 'rlr', 'rrl', 'rrr']:
                idx = ['lll', 'llr', 'lrl', 'lrr', 'rll', 'rlr', 'rrl', 'rrr'].index(state)
                r3[0][idx] = self.kappa
                r3[1][idx] = 1 - self.kappa
        
        # Add to model
        cpd_r1_given_s1 = TabularCPD(
            variable='r1',
            variable_card=2,
            evidence=['s1'],
            evidence_card=[2],
            values=r1
        )
        cpd_r2_given_s2 = TabularCPD(
            variable='r2',
            variable_card=2,
            evidence=['s2'],
            evidence_card=[4],
            values=r2
        )
        cpd_r3_given_s3 = TabularCPD(
            variable='r3',
            variable_card=2,
            evidence=['s3'],
            evidence_card=[8],
            values=r3
        )
        self.model.add_cpds(cpd_r1_given_s1, cpd_r2_given_s2, cpd_r3_given_s3)
        
        
    def plan(self, goal, maxima):

        # Define p(o3 | s3)
        self.def_goal(goal, maxima)

        # Inference
        inference = BeliefPropagation(self.model)
        self.prior = inference.query(self.node_list).values
        self.posterior = inference.query(self.node_list, evidence={'r1' : 0, 'r2' : 0, 'r3' : 0, 'o3': 0}).values

        # Get relevant values
        self.posterior_a = self.posterior.sum(axis=(1, 3, 5))
        self.posterior_s3 = self.posterior.sum(axis=(0, 1, 2, 3, 4))
        self.path_pred = self.posterior_a.ravel()

        self.posterior_a0 = self.posterior.sum(axis=tuple(range(1, 6)))  # 'a0'
        self.posterior_a1 = self.posterior.sum(axis=(0, 3, 4, 5)).T  # sorry, making this way harder to read (['a1', 's1'])
        self.posterior_a2 = self.posterior.sum(axis=(0, 1, 2, 5)).T

    def sample(self, goal, threshold=1e-2, prior=None):

        # Define p(o3 | s3)
        self.def_goal(goal)

        # Inference
        if prior is None:
            inference = BeliefPropagation(self.model)
            self.prior = inference.query(self.node_list).values
        else:
            self.prior = prior  # enable to set prior by hand
        sampling = Sampler(self.model, self.prior, threshold)
        self.posterior, self.n = sampling.query()

        # Get relevant values
        self.posterior_a = self.posterior.sum(axis=(1, 3, 5))
        self.posterior_s3 = self.posterior.sum(axis=(0, 1, 2, 3, 4))
        self.path_pred = self.posterior_a.ravel()

    def learn(self):

        self.alpha_a0.infer(self.posterior_a0.reshape(-1, 1))
        self.alpha_a1.infer(self.posterior_a1)
        self.alpha_a2.infer(self.posterior_a2)

        # Update the CPDs of the model
        cpd_a0 = TabularCPD('a0', 2, self.alpha_a0.get_MAP_cpd())
        cpd_a1_given_s1 = TabularCPD(
            variable='a1',
            variable_card=2,
            evidence=['s1'],
            evidence_card=[2],
            values=self.alpha_a1.get_MAP_cpd()
        )
        cpd_a2_given_s2 = TabularCPD(
            variable='a2', 
            variable_card=2, 
            evidence=['s2'],
            evidence_card=[4],
            values=self.alpha_a2.get_MAP_cpd()
        )
        self.model.add_cpds(cpd_a0, cpd_a1_given_s1, cpd_a2_given_s2)

    def get_it_measures(self):

        self.complexity = entropy(self.posterior.flatten(), self.prior.flatten())
        self.error = entropy(self.posterior_s3) + entropy(self.posterior_s3, self.model.get_cpds()[8].values[0, :])
        self.surprise = self.complexity + self.error

        return self.complexity, self.error, self.surprise