from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation
import pandas as pd
import numpy as np
from loading_reward_data import reward_processing
from loading_experiment_data import load_maze_data
import logging
from scipy.stats import linregress

# Suppress warnings from pgmpy
logging.getLogger("pgmpy").setLevel(logging.ERROR)

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


def trial_learning(model, alpha_a0, alpha_a1, alpha_a2, reward_o3):
    # P(o3 | s3)
    cpd_o3_given_s3 = TabularCPD(
        variable='o3',
        variable_card=2,
        evidence=['s3'],
        evidence_card=[8],
        values = reward_o3
    )
    
    model.add_cpds(cpd_o3_given_s3)

    # Perform inference
    inference = BeliefPropagation(model)

    # Compute model priors for all states of s3; prior prediction of all paths
    prior_s3 = inference.query(['s3'])

    # Infer posteriors P(a0 | o3=0), P(a1 | o3=0), P(a2 | o3=0)
    posterior_a0 = inference.query(['a0'], evidence={'o3': 0})  # we always observe B=0 (like "reward")
    posterior_a1 = inference.query(['a1', 's1'], evidence={'o3': 0})
    posterior_a2 = inference.query(['a2', 's2'], evidence={'o3': 0})

    # Compute model posteriors for all states of s3; posterior prediction of all paths
    posterior_s3 = inference.query(['s3'], evidence={'o3': 0})

    # Infer hyperprior P(a0), P(a1|s1), P(a2|s2) based on the posteriors; update the Dirichlet priors
    alpha_a0.infer(posterior_a0.values.reshape(2, 1))
    alpha_a1.infer(posterior_a1.values.reshape(2, 2))
    alpha_a2.infer(posterior_a2.values.reshape(2, 4))

    # Update the CPDs of the model
    cpd_a0 = TabularCPD('a0', 2, alpha_a0.get_MAP_cpd())
    cpd_a1_given_s1 = TabularCPD(
        variable='a1',
        variable_card=2,
        evidence=['s1'],
        evidence_card=[2],
        values=alpha_a1.get_MAP_cpd()
    )
    cpd_a2_given_s2 = TabularCPD(variable='a2', 
                                variable_card=2, 
                                evidence=['s2'],
                                evidence_card=[4],
                                values=alpha_a2.get_MAP_cpd())

    model.add_cpds(cpd_a0, cpd_a1_given_s1, cpd_a2_given_s2)

    return prior_s3, posterior_s3

# Initialization of the model
# Learning and evaluation (prediction) of the model
def evaluate_model(rewards_participant, param):
    # Create network structure
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

    # P(o3 | s3)
    cpd_o3_given_s3 = TabularCPD(
        variable='o3',
        variable_card=2,
        evidence=['s3'],
        evidence_card=[8],
        values = rewards_participant[0]['distribution']
    )
    
    model.add_cpds(cpd_o3_given_s3)

    # ----------------------------------------------------------------------------------------------------------------------
    # Iteration and Evaluation
    # ----------------------------------------------------------------------------------------------------------------------

    # Determine the habitual path of the participant
    habitual_path = None
    for i in range(len(rewards_participant)):
        if rewards_participant[i]['extra_elements'][0] == 0:
            habitual_path = rewards_participant[i]['extra_elements'][1]
            break
    #print(f'Habitual Path: {habitual_path} ')

    # Initialize arrays to store the predicted probabilities of the optimal path for each distance to the habitual path
    post_dist_0 = np.array([])
    post_dist_1 = np.array([])
    post_dist_2 = np.array([])
    post_dist_3 = np.array([])

    prior_dist_0 = np.array([])
    prior_dist_1 = np.array([])
    prior_dist_2 = np.array([])
    prior_dist_3 = np.array([])

    # Store counts of how often a path was the optimal path
    counts = np.zeros(8)

    for i in range(len(rewards_participant)):
        # Optimal path of the trial:
        optimal_path = rewards_participant[i]['extra_elements'][1]

        counts[optimal_path] += 1

         # Let the model learn from the reward of trial i
        reward_trial = rewards_participant[i]['distribution']
        prior_prediction, posterior_prediction = trial_learning(model, alpha_a0, alpha_a1, alpha_a2, reward_trial)

        # Prior and Posterior model predictions for the optimal path
        prior_optimal_path = prior_prediction.values[optimal_path]
        post_optimal_path = posterior_prediction.values[optimal_path] 

        # Distance to the habitual path
        dist = rewards_participant[i]['extra_elements'][0]

        # Sort the predicted probabilities of the optimal path based on the distance to the habitual path 
        if dist == 0 and i > 85:
            post_dist_0 = np.append(post_dist_0, post_optimal_path)
            prior_dist_0 = np.append(prior_dist_0, prior_optimal_path)

        if dist == 1:
            post_dist_1 = np.append(post_dist_1, post_optimal_path)
            prior_dist_1 = np.append(prior_dist_1, prior_optimal_path)

        if dist == 2:
            post_dist_2 = np.append(post_dist_2, post_optimal_path)
            prior_dist_2 = np.append(prior_dist_2, prior_optimal_path)

        if dist == 3:
            post_dist_3 = np.append(post_dist_3, post_optimal_path)
            prior_dist_3 = np.append(prior_dist_3, prior_optimal_path)

        """ # Speichere die Daten dieses Trials in der Liste
        data.append({
            'Trial': i + 1,
            'Distance': dist,
            'Optimal_Path': optimal_path,
            'Prior_Optimal_Path': prior_optimal_path,
            'Posterior_Optimal_Path': post_optimal_path
        })
         """
    return habitual_path, post_dist_0, post_dist_1, post_dist_2, post_dist_3, prior_dist_0, prior_dist_1, prior_dist_2, prior_dist_3, counts
        

""" data = []
 """

probabilities = [0.886]
parameters = [12.0]
distances = np.array([1, 2, 3, 4])

# Initialize a matrix to store the results len(probabilities) x len(parameters)
matrix = np.zeros((len(probabilities), len(parameters)))

# Log diffs in csv file
data = []

# Find regression line for the following four data points
exp_data = np.array([0.7427441, 0.6653386, 0.4615385, 0.4559387])
slope, intercept, r_value, p_value, std_err = linregress(distances, exp_data)
regression_line_exp = slope * distances + intercept

# Grid Search for the best parameters
for probability in probabilities:
    for parameter in parameters:
        # Load the reward data of all participants
        rewards = reward_processing(probability) 
        # Only use the first participant for now
        rewards = {key: rewards[key] for key in list(rewards.keys())[18:22]}

        # Initialize arrays to store the probabilities of choosing the optimal path for each distance to the habitual path
        all_post_0 = np.array([])
        all_post_1 = np.array([])
        all_post_2 = np.array([])
        all_post_3 = np.array([])

        all_prior_0 = np.array([])
        all_prior_1 = np.array([])
        all_prior_2 = np.array([])
        all_prior_3 = np.array([])

        # Iterate over all participants to evaluate the model
        for key in rewards.keys():
            rewards_participant = rewards[key]
            habitual_path, post_dist_0, post_dist_1, post_dist_2, post_dist_3, prior_dist_0, prior_dist_1, prior_dist_2, prior_dist_3, counts = evaluate_model(rewards_participant, parameter)
            
            all_post_0 = np.append(all_post_0, post_dist_0)
            all_post_1 = np.append(all_post_1, post_dist_1)
            all_post_2 = np.append(all_post_2, post_dist_2)
            all_post_3 = np.append(all_post_3, post_dist_3)

            all_prior_0 = np.append(all_prior_0, prior_dist_0)
            all_prior_1 = np.append(all_prior_1, prior_dist_1)
            all_prior_2 = np.append(all_prior_2, prior_dist_2)
            all_prior_3 = np.append(all_prior_3, prior_dist_3)

        # Mittelwerte der Wahrscheinlichkeiten für jede Distanz
        means = [np.mean(all_post_0), np.mean(all_post_1), np.mean(all_post_2), np.mean(all_post_3)]

        # Lineare Regression durchführen
        slope, intercept, r_value, p_value, std_err = linregress(distances, means)

        # Regressionslinie berechnen
        regression_line = slope * distances + intercept

        # Berechne die Differenz zwischen der Regressionslinie und der experimentellen Daten
        diff = np.sum(np.abs(regression_line - regression_line_exp))

        # Save the difference in the matrix
        matrix[probabilities.index(probability), parameters.index(parameter)] = diff

        print(f'Probability: {probability}, Parameter: {parameter}, Difference: {diff}')
        print(f'Progress of Grid Search: {np.round((probabilities.index(probability) * len(parameters) + parameters.index(parameter)) / (len(probabilities) * len(parameters)) * 100, 2)}%')
         
        # Speichere die Daten in der Liste
        data.append({
            'Probability': probability,
            'Parameter': parameter,
            'Difference': diff
        })

# Save the matrix in a csv file
df = pd.DataFrame(matrix)
df.to_csv('diff_matrix.csv', index=False)

# Extract top 10 minimum values of the matrix and the values for the best probability and parameter
min_values = np.sort(matrix.flatten())[:10]
min_indices = np.unravel_index(np.argsort(matrix, axis=None), matrix.shape)






