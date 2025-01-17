from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation
import pandas as pd
import numpy as np
from loading_reward_data import reward_processing, load_reward_data
from loading_experiment_data import load_maze_data
import logging
from scipy.stats import linregress
import random
from sklearn.model_selection import KFold

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
def train_model(rewards_participant, param):
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

    # Initialize an array of tuples to store the model predictions for the optimal path and its distance to the habitual path
    post_dist = np.array([])

    # Initialize an array of tuples to store the model predictions for the optimal path and its distance to the habitual path
    post_dist_0 = np.array([])
    post_dist_1 = np.array([])
    post_dist_2 = np.array([])
    post_dist_3 = np.array([])

    prior_dist_0 = np.array([])
    prior_dist_1 = np.array([])
    prior_dist_2 = np.array([])
    prior_dist_3 = np.array([])

    for i in range(len(rewards_participant)):
        # Optimal path of the trial:
        optimal_path = rewards_participant[i]['extra_elements'][1]

         # Let the model learn from the reward of trial i
        reward_trial = rewards_participant[i]['distribution']
        prior_prediction, posterior_prediction = trial_learning(model, alpha_a0, alpha_a1, alpha_a2, reward_trial)

        # Prior and Posterior model predictions for the optimal path
        prior_optimal_path = prior_prediction.values[optimal_path]
        post_optimal_path = posterior_prediction.values[optimal_path] 

        # Distance to the habitual path
        dist = rewards_participant[i]['extra_elements'][0]

        post_dist = np.append(post_dist, post_optimal_path)

        # Sort the predicted probabilities of the optimal path based on the distance to the habitual path 
        if dist == 0:
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

    return post_dist, post_dist_0, post_dist_1, post_dist_2, post_dist_3
        
# Splits the experimental data of one participant into training and testing sets
# For every distance to the habitual path {0, 1, 2, 3} the test set needs contain 3 trials for each distance > 0 and 
# 9 trials for distance 0. These trials should be selected from different blocks of the experimental data.
# Such that we sample one trial with each distance 1,2 & 3from the first third of trials, one from the second third and one from the last third.
# Same with the 9 trials for distance 0.
# Exp_data_participant is the experimental data of one participant. Its a list of dictionaries. Each dictionary contains the data of one trial.
def split_data(exp_data_participant):
    """
    Splits experimental data into training and testing sets based on the described rules.

    Parameters:
    - exp_data_participant (list): A list of 100 dictionaries, where each dictionary contains the data for one trial.

    Returns:
    - dict: A dictionary with 'train_set' and 'test_set' keys containing the respective trials.
    """
    # Initialize containers for training and testing sets
    train_set = []
    test_set = []

    # Group trials by distance
    distance_groups = {0: [], 1: [], 2: [], 3: []}
    for trial in exp_data_participant:
        distance = int(trial['distance'])  # Convert distance to an integer
        if distance in distance_groups:  # Ensure valid distance
            distance_groups[distance].append(trial)

    # Ensure groups are distributed across thirds of the data
    third_size = len(exp_data_participant) // 3

    # Divide data into thirds
    thirds = [
        exp_data_participant[:third_size],
        exp_data_participant[third_size:2 * third_size],
        exp_data_participant[2 * third_size:]
    ]

    # Sample test set
    for distance, count in [(0, 3), (1, 3), (2, 3), (3, 3)]:
        sampled_trials = []
        for third in thirds:
            # Filter trials in this third by the current distance
            available_trials = [trial for trial in third if int(trial['distance']) == distance]
            sampled_trials.extend(random.sample(available_trials, min(count // 3, len(available_trials))))
        
        # Add sampled trials to test set
        test_set.extend(sampled_trials)

        # Remove sampled trials from the distance group
        distance_groups[distance] = [trial for trial in distance_groups[distance] if trial not in sampled_trials]

    # Remaining trials form the training set
    for trials in distance_groups.values():
        train_set.extend(trials)

    return train_set, test_set

# Log-Likelihood Berechnung
def compute_log_likelihood(predictions, actual_choices):
    """
    Berechnet die Log-Likelihood zwischen Modellvorhersagen und experimentellen Daten.

    Parameters:
        predictions (array): Wahrscheinlichkeiten, die das Modell für jede Option vorhersagt.
        actual_choices (array): Binäre Werte (0 oder 1), die angeben, ob eine Option gewählt wurde.

    Returns:
        log_likelihood (float): Die berechnete Log-Likelihood.
    """
    # Vermeiden von log(0), indem minimale Werte hinzugefügt werden
    epsilon = 1e-9
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    
    # Berechnung der Log-Likelihood
    log_likelihood = np.sum(actual_choices * np.log(predictions) + (1 - actual_choices) * np.log(1 - predictions))

    # Normalise by number of trials
    log_likelihood /= len(actual_choices)

    return log_likelihood


def k_cross_validation(predictions, actual_choices, k=5):
    """
    Perform k-fold cross-validation and compute the mean log-likelihood.

    Parameters:
    - predictions (array-like): Array of predicted probabilities for each trial.
    - actual_choices (array-like): Array of actual choices (0 or 1) for each trial.
    - k (int): Number of folds for cross-validation.

    Returns:
    - float: Mean log-likelihood across all folds.
    """
    
    # Initialize the KFold object
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    normalized_log_likelihoods = []
    
    # Perform k-fold cross-validation
    for train_idx, val_idx in kf.split(predictions):
        # Split the predictions and actual choices into training and validation sets
        train_predictions, val_predictions = predictions[train_idx], predictions[val_idx]
        train_choices, val_choices = actual_choices[train_idx], actual_choices[val_idx]
        
        # Compute log-likelihood for the validation set
        log_likelihood = np.sum(
            np.log(val_predictions) * val_choices + 
            np.log(1 - val_predictions) * (1 - val_choices)
        )
    
        # Normalize by the number of validation trials
        normalized_log_likelihood = log_likelihood / len(val_idx)
        normalized_log_likelihoods.append(normalized_log_likelihood)
    
    # Compute the mean normalized log-likelihood across folds
    mean_normalized_log_likelihood = np.mean(normalized_log_likelihoods)
    return mean_normalized_log_likelihood



# ----------------------------------------------------------------------------------------------------------------------
# Define Grid Search parameters
probabilities = [0.95, 0.9, 0.85]
parameters = [5.0, 10.0, 15.0]

# Initialize a matrix to store the results len(probabilities) x len(parameters)
matrix = np.zeros((len(probabilities), len(parameters)))

# Load experimental data
maze_data = load_maze_data('maze_data.csv')
   

exp_data_participant = maze_data['1']
train_set, test_set = split_data(exp_data_participant)


reward_data = load_reward_data(r'C:\Users\valen\OneDrive\Dokumente\7Semester\Bachelorarbeit\maze_planning\reward_data')
key = 'rewards_1'

# Grid Search
for probability in probabilities:
    for parameter in parameters:
        rewards = reward_processing(reward_data, key, probability)

        # Train the model
        post_dist, post_dist_0, post_dist_1, post_dist_2, post_dist_3 = train_model(rewards, parameter)

        # Compute the log-likelihood between post_dist and train_set 
        # Post_dist consists of 100 tuples of the form (post_optimal_path, dist)
        # We need to extract the trials from the Post_dist that are in the train_set
        predictions = np.array([])
        actual_choices = np.array([])

        for trial in train_set:
            index = int(trial['X'])
            predictions = np.append(predictions, post_dist[index])
            actual_choices = np.append(actual_choices, int(trial['optimality'] == 'TRUE'))

        # K-Cross-Validation
        mean_log_likelihood = k_cross_validation(predictions, actual_choices, k=5)

        # Save the log-likelihood in the matrix at the corresponding position of the probability and parameter
        matrix[probabilities.index(probability), parameters.index(parameter)] = mean_log_likelihood

print(matrix)

# Save the probability and parameter with the best log-likelihood
best_log_likelihood = np.max(matrix)
best_indices = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)
best_probability = probabilities[best_indices[0]]
best_parameter = parameters[best_indices[1]]

print(f"Best probability: {best_probability}")
print(f"Best parameter: {best_parameter}")

# Validate with test set
rewards = reward_processing(reward_data, key, best_probability)
post_dist, post_dist_0, post_dist_1, post_dist_2, post_dist_3 = train_model(rewards, best_parameter)

predictions = np.array([])
actual_choices = np.array([])
for trial in test_set:
    index = int(trial['X'])
    predictions = np.append(predictions, post_dist[index])
    actual_choices = np.append(actual_choices, int(trial['optimality'] == 'TRUE'))

log_likelihood = compute_log_likelihood(predictions, actual_choices)

print(f"Log-likelihood on test set: {log_likelihood}")







