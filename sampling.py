from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation
import numpy as np
from scipy.stats import entropy

# Define sampler
class Sampler():

    def __init__(self, model, prior, threshold=1e-4):
        self.model = model
        self.prior = prior / prior.sum()  # make sure it sums to 1
        self.flat_prior = self.prior.ravel()
        self.threshold = threshold
        
        self.sampled_dir = self.prior.copy()
        self.trace = [self.prior.copy()]
        self.sampled_belief = self.prior.copy()

    def query(self, query_vars, evidence):

        kl_divs = np.ones(50)  # specify convergence window (no large influence)
        
        while sum(kl_divs) > self.threshold:

            flat_sample_idx = np.random.choice(len(self.flat_prior), p=self.flat_prior)
            sample_idx = np.unravel_index(flat_sample_idx, self.prior.shape)
            p_o3 = self.model.get_cpds('o3').values[:, sample_idx[-1]]  # only evaluate p(o | s) with last state
            accept = np.random.choice([0, 1], p=1-p_o3)  # reverse probability because 0 = reward

            if accept:
                flattened_belief_tm1 = self.sampled_belief.copy()[self.sampled_belief != 0].flatten()  # exclude zeros for klD calculation
                self.sampled_dir[sample_idx] += 1
                self.sampled_belief = self.sampled_dir / self.sampled_dir.sum()
                
                flattened_belief = self.sampled_belief[self.sampled_belief != 0].flatten()  # exclude zeros for klD calculation
                kl_div = entropy(flattened_belief, flattened_belief_tm1)
                kl_divs = kl_divs[1:]
                kl_divs = np.concatenate((kl_divs, kl_div.reshape(1)))

            #print(sum(kl_divs))
            self.trace.append(self.sampled_belief)

        return self.sampled_belief, len(self.trace)