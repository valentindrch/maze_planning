from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation
import pandas as pd
import numpy as np
from loading_reward_data import reward_processing
import logging

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

    # Infer posteriors P(a0 | o3=0), P(a1 | o3=0), P(a2 | o3=0)
    inference = BeliefPropagation(model)
    posterior_a0 = inference.query(['a0'], evidence={'o3': 0})  # we always observe B=0 (like "reward")
    posterior_a1 = inference.query(['a1', 's1'], evidence={'o3': 0})
    posterior_a2 = inference.query(['a2', 's2'], evidence={'o3': 0})

    # Infer hyperprior P(a0), P(a1|s1), P(a2|s2) based on the posteriors
    alpha_a0.infer(posterior_a0.values.reshape(2, 1))
    alpha_a1.infer(posterior_a1.values.reshape(2, 2))
    alpha_a2.infer(posterior_a2.values.reshape(2, 4))

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


# MAP prediction function
def map_prediction(model):
    inference = BeliefPropagation(model)
    posterior_s3 = inference.query(['s3'], evidence={'o3': 0})
    return posterior_s3

# For the data of one participant, the model learns 
# Furthermore, for each trial, the probability of choosing the optimal path is predicted and stored
def evaluate_model(reward_o3):
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

    param = 2.0
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


    # ----------------------------------------------------------------------------------------------------------------------
    # Learning and Evaluation
    # ----------------------------------------------------------------------------------------------------------------------

    # Determine the habitual path of the participant
    habitual_path = None
    for i in range(len(reward_o3)):
        if reward_o3[i]['extra_elements'][0] == 0:
            habitual_path = reward_o3[i]['extra_elements'][1]
            break

    # Iterate over trials to learn the hyperpriors
    # Extract the reward data from the trials
    distance_0 = np.array([])
    distance_1 = np.array([])
    distance_2 = np.array([])
    distance_3 = np.array([])

    for i in range(len(reward_o3)):
        trial_learning(model, alpha_a0, alpha_a1, alpha_a2, reward_o3[i]['distribution'])

        # Save the predicted probability of the optimal path at trial i
        optimal_path = reward_o3[i]['extra_elements'][1]
        prediction = map_prediction(model)
        prob_optimal_path = prediction.values[optimal_path]

        # Sort the predicted probabilities of the optimal path based on the distance to the habitual path 
        if reward_o3[i]['extra_elements'][0] == 0 and i > 85:
            distance_0 = np.append(distance_0, prob_optimal_path)

        if reward_o3[i]['extra_elements'][0] == 1:
            distance_1 = np.append(distance_1, prob_optimal_path)

        if reward_o3[i]['extra_elements'][0] == 2:
            distance_2 = np.append(distance_2, prob_optimal_path)

        if reward_o3[i]['extra_elements'][0] == 3:
            distance_3 = np.append(distance_3, prob_optimal_path)

    return habitual_path, distance_0, distance_1, distance_2, distance_3
        

# Load the reward data of all participants
rewards = reward_processing() 
# Only use the first participant for now
rewards = {key: rewards[key] for key in list(rewards.keys())[:1]}

# Initialize arrays to store the probabilities of choosing the optimal path for each distance to the habitual path
all_probs_0 = np.array([])
all_probs_1 = np.array([])
all_probs_2 = np.array([])
all_probs_3 = np.array([])

# Iterate over all participants to evaluate the model
for key in rewards.keys():
    reward_o3 = rewards[key]
    habitual_path, temp_0, temp_1, temp_2, temp_3 = evaluate_model(reward_o3)
    print(habitual_path)
    all_probs_0 = np.append(all_probs_0, temp_0)
    all_probs_1 = np.append(all_probs_1, temp_1)
    all_probs_2 = np.append(all_probs_2, temp_2)
    all_probs_3 = np.append(all_probs_3, temp_3)


# Calculate the mean probability of choosing the optimal path for each distance to the habitual path
prob_distance_0 = np.mean(all_probs_0)
prob_distance_1 = np.mean(all_probs_1)
prob_distance_2 = np.mean(all_probs_2)
prob_distance_3 = np.mean(all_probs_3)
print(f'Mean probability to choose the optimal path when distance to habitual path is 0: {prob_distance_0}')
print(f'Mean probability to choose the optimal path when distance to habitual path is 1: {prob_distance_1}')
print(f'Mean probability to choose the optimal path when distance to habitual path is 2: {prob_distance_2}')
print(f'Mean probability to choose the optimal path when distance to habitual path is 3: {prob_distance_3}')


# Plot the probabilities of choosing the optimal path based on the distance to the habitual path
# The individual data points should be shown
# A line should be drawn connecting the mean probabilities for each distance
# The x-axis should represent the distance to the habitual path
# The y-axis should represent the probability of choosing the optimal path
import matplotlib.pyplot as plt
# Plot the probabilities of choosing the optimal path based on the distance to the habitual path
plt.figure(figsize=(10, 6))

# Plot individual data points
plt.scatter(np.zeros_like(all_probs_0), all_probs_0, color='blue', label='Distance 0', alpha=0.6)
plt.scatter(np.ones_like(all_probs_1), all_probs_1, color='green', label='Distance 1', alpha=0.6)
plt.scatter(np.full_like(all_probs_2, 2), all_probs_2, color='orange', label='Distance 2', alpha=0.6)
plt.scatter(np.full_like(all_probs_3, 3), all_probs_3, color='red', label='Distance 3', alpha=0.6)

# Plot mean probabilities
plt.plot([0, 1, 2, 3], [prob_distance_0, prob_distance_1, prob_distance_2, prob_distance_3], color='black', marker='o', linestyle='-', linewidth=2, markersize=8, label='Mean Probability')

# Labels and title
plt.xlabel('Distance to Habitual Path')
plt.ylabel('Probability of Choosing Optimal Path')
plt.title('Probability of Choosing Optimal Path Based on Distance to Habitual Path')
plt.legend()
plt.grid(True)
plt.show()