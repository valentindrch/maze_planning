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
def evaluate_model(rewards_participant):
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

    param = 0.5
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
    print(f'Habitual Path: {habitual_path} ')

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

        if i % 20 == 0 or i == len(rewards_participant) - 1:
            print(f'Trial {i + 1}:')
            print(np.round(prior_prediction.values, 3), np.round(posterior_prediction.values, 3))
            print("")


        # Distance to the habitual path
        dist = rewards_participant[i]['extra_elements'][0]

        if post_optimal_path > 0.9 and dist == 3:
            print(f'High probability for trial {i + 1} and distance 3: {post_optimal_path}')
            print(f'Prior probability was: {prior_optimal_path}')
            print(f'Optimal Path was: {optimal_path}')

        """ if dist == 1:
            print(f'Probability for trial {i + 1} and distance 1: {post_optimal_path}')
            print(f'Prior probability was: {prior_optimal_path}')
            print(f'Optimal Path was: {optimal_path}') """

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

        # Speichere die Daten dieses Trials in der Liste
        data.append({
            'Trial': i + 1,
            'Distance': dist,
            'Optimal_Path': optimal_path,
            'Prior_Optimal_Path': prior_optimal_path,
            'Posterior_Optimal_Path': post_optimal_path
        })
        
    return habitual_path, post_dist_0, post_dist_1, post_dist_2, post_dist_3, prior_dist_0, prior_dist_1, prior_dist_2, prior_dist_3, counts
        

data = []

# Load the reward data of all participants
rewards = reward_processing() 
# Only use the first participant for now
rewards = {key: rewards[key] for key in list(rewards.keys())[3:4]}

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
    print(f'---------------------------------------------------------------------------------------')
    print(f'Participant: {key}')
    rewards_participant = rewards[key]
    habitual_path, post_dist_0, post_dist_1, post_dist_2, post_dist_3, prior_dist_0, prior_dist_1, prior_dist_2, prior_dist_3, counts = evaluate_model(rewards_participant)
    
    all_post_0 = np.append(all_post_0, post_dist_0)
    all_post_1 = np.append(all_post_1, post_dist_1)
    all_post_2 = np.append(all_post_2, post_dist_2)
    all_post_3 = np.append(all_post_3, post_dist_3)

    all_prior_0 = np.append(all_prior_0, prior_dist_0)
    all_prior_1 = np.append(all_prior_1, prior_dist_1)
    all_prior_2 = np.append(all_prior_2, prior_dist_2)
    all_prior_3 = np.append(all_prior_3, prior_dist_3)

    print(f'Path was optimal path: {counts}')

# Calculate the mean probability of choosing the optimal path for each distance to the habitual path
mean_prediction_0 = np.mean(all_post_0)
mean_prediction_1 = np.mean(all_post_1)
mean_prediction_2 = np.mean(all_post_2)
mean_prediction_3 = np.mean(all_post_3)
print(f'Mean probability to choose the optimal path when distance to habitual path is 0: {mean_prediction_0}')
print(f'Mean probability to choose the optimal path when distance to habitual path is 1: {mean_prediction_1}')
print(f'Mean probability to choose the optimal path when distance to habitual path is 2: {mean_prediction_2}')
print(f'Mean probability to choose the optimal path when distance to habitual path is 3: {mean_prediction_3}')

# Calculate the mean prior probability of choosing the optimal path for each distance to the habitual path
mean_prior_0 = np.mean(all_prior_0)
mean_prior_1 = np.mean(all_prior_1)
mean_prior_2 = np.mean(all_prior_2)
mean_prior_3 = np.mean(all_prior_3)
print(f'Mean prior probability to choose the optimal path when distance to habitual path is 0: {mean_prior_0}')
print(f'Mean prior probability to choose the optimal path when distance to habitual path is 1: {mean_prior_1}')
print(f'Mean prior probability to choose the optimal path when distance to habitual path is 2: {mean_prior_2}')
print(f'Mean prior probability to choose the optimal path when distance to habitual path is 3: {mean_prior_3}')


# Erstelle eine DataFrame aus den gesammelten Daten
results_df = pd.DataFrame(data)

# Exportiere die Tabelle in eine CSV-Datei
results_df.to_csv('trial_results.csv', index=False)

# Plotting the results
import matplotlib.pyplot as plt
# Plot the probabilities of choosing the optimal path based on the distance to the habitual path
plt.figure(figsize=(10, 6))

# Plot individual data points
plt.scatter(np.zeros_like(all_post_0), all_post_0, color='blue', label='Posterior', alpha=0.6)
plt.scatter(np.ones_like(all_post_1), all_post_1, color='green', label='Posterior', alpha=0.6)
plt.scatter(np.full_like(all_post_2, 2), all_post_2, color='orange', label='Posterior', alpha=0.6)
plt.scatter(np.full_like(all_post_3, 3), all_post_3, color='red', label='Posterio', alpha=0.6)

# Include Prior probabilities
plt.scatter(np.zeros_like(all_prior_0), all_prior_0, color='blue', marker='x', label='Prior')
plt.scatter(np.ones_like(all_prior_1), all_prior_1, color='green', marker='x', label='Prior')
plt.scatter(np.full_like(all_prior_2, 2), all_prior_2, color='orange', marker='x', label='Prior')
plt.scatter(np.full_like(all_prior_3, 3), all_prior_3, color='red', marker='x', label='Prior')

# Plot mean probabilities
plt.plot([0, 1, 2, 3], [mean_prediction_0, mean_prediction_1, mean_prediction_2, mean_prediction_3], color='black', marker='o', linestyle='-', linewidth=2, markersize=8, label='Mean Probability')

# Plot mean prior probabilities
plt.plot([0, 1, 2, 3], [mean_prior_0, mean_prior_1, mean_prior_2, mean_prior_3], color='black', marker='x', linestyle='--', linewidth=2, markersize=8, label='Mean Prior Probability')

# Labels and title
plt.xlabel('Distance to Habitual Path')
plt.ylabel('Probability of Choosing Optimal Path')
plt.title('Probability of Choosing Optimal Path Based on Distance to Habitual Path')
plt.legend()
plt.grid(True)
plt.show()