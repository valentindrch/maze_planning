from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation
import pandas as pd
import numpy as np
import logging
from scipy.stats import linregress
import random
from sklearn.model_selection import KFold
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
from loading_reward_data import reward_processing, load_reward_data
from loading_experiment_data import load_maze_data

# Remove all existing handlers
for handler in logging.getLogger().handlers[:]:
    logging.getLogger().removeHandler(handler)
import pickle
from scipy.stats import entropy

# Suppress warnings from pgmpy
logging.getLogger("pgmpy").setLevel(logging.ERROR)

# --- Logging Configuration ---
# Create a logs directory if it doesn't exist
if not os.path.exists("complete_logs"):
    os.makedirs("complete_logs")

# Configure logging to log to both file and console
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(message)s",
    handlers=[
        logging.FileHandler("complete_logs/participant_analysis.log"),  # Write to log file
        logging.StreamHandler()  # Print to console
    ]
)

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
    nodes = [node for node in model.nodes if node != 'o3']
    inference = BeliefPropagation(model)
    prior = inference.query(nodes).values
    posterior = inference.query(nodes, evidence={'o3': 0}).values  # might be a bit more efficient to only call `query` once
    posterior_a0 = posterior.sum(axis=tuple(range(1, 6)))  # 'a0'
    posterior_a1 = posterior.sum(axis=(0, 3, 4, 5)).T  # sorry, making this way harder to read (['a1', 's1'])
    posterior_a2 = posterior.sum(axis=(0, 1, 2, 5)).T  # ['a2', 's2']
    posterior_s3 = posterior.sum(axis=tuple(range(0, 5)))  # 's3'

    # Infer hyperprior P(a0), P(a1|s1), P(a2|s2) based on the posteriors
    alpha_a0.infer(posterior_a0.reshape(2, 1))
    alpha_a1.infer(posterior_a1.reshape(2, 2))
    alpha_a2.infer(posterior_a2.reshape(2, 4))

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

    return prior, posterior, posterior_s3

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

    post_dist = []  # Start with an empty list
    priors = []
    it_measures = {'complexity': [], 'error': [], 'surprise': []}

    for i in range(len(rewards_participant)):
        # Optimal path of the trial:
        optimal_path = rewards_participant[i]['extra_elements'][1]

         # Let the model learn from the reward of trial i
        reward_trial = rewards_participant[i]['distribution']
        prior_prediction, posterior_prediction, posterior_s3 = trial_learning(model, alpha_a0, alpha_a1, alpha_a2, reward_trial)
        priors.append(prior_prediction)

        # Calculate IT metrics
        complexity = entropy(posterior_prediction.flatten(), prior_prediction.flatten())
        error = entropy(posterior_s3) + entropy(posterior_s3, model.get_cpds()[6].values[0, :])
        it_measures['complexity'].append(complexity)
        it_measures['error'].append(error)
        it_measures['surprise'].append(complexity + error)

        # Prior and Posterior model predictions for the optimal path
        prior_optimal_path = prior_prediction.values[optimal_path]
        post_optimal_path = posterior_prediction.values[optimal_path] 

        # Distance to the habitual path
        dist = rewards_participant[i]['extra_elements'][0]

        post_dist.append((post_optimal_path, dist))  # Append a tuple

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

    return np.array(priors), habitual_path, post_dist, post_dist_0, post_dist_1, post_dist_2, post_dist_3, it_measures
        
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
    for distance, count in [(0, 10), (1, 3), (2, 3), (3, 3)]:
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

def run_partipant(exp_key, rewards_key, reward_data, exp_data_participant, matrix, probabilities, parameters, all_matrix, all_post_0, all_post_1, all_post_2, all_post_3):
    train_set, test_set = split_data(exp_data_participant)
    # Grid Search
    for probability in probabilities:
        for parameter in parameters:
            rewards = reward_processing(reward_data, rewards_key, probability)

            # Train the model
            post_dist, post_dist_0, post_dist_1, post_dist_2, post_dist_3  = train_model(rewards, parameter)

            # Compute the log-likelihood between post_dist and train_set 
            # Post_dist consists of 100 tuples of the form (post_optimal_path, dist)
            # We need to extract the trials from the Post_dist that are in the train_set
            predictions = np.array([])
            actual_choices = np.array([])

            

            for trial in train_set:
                idx = int(trial['X'])
                model_trial = post_dist[idx]  # Access the first trial
                posterior, distance = model_trial
                
                predictions = np.append(predictions, posterior)
                actual_choices = np.append(actual_choices, int(trial['optimality'] == 'TRUE'))

            # K-Cross-Validation
            mean_log_likelihood = k_cross_validation(predictions, actual_choices, k=5)

            # Save the log-likelihood in the matrix at the corresponding position of the probability and parameter
            matrix[probabilities.index(probability), parameters.index(parameter)] = mean_log_likelihood

            # Print the current probability and parameter and status of the grid search as a percentage
            print(f"Probability: {probability}, Parameter: {parameter}, Progress: {np.sum(matrix != 0) / matrix.size * 100:.2f}%")

            

    # Add the matrix to the all_matrix such that the log-likelihoods of all participants are stored
    # We want the entire matrix to be added on top of the all_matrix
    all_matrix += matrix

    

    # Save the probability and parameter with the best log-likelihood
    training_log_likelihood = np.max(matrix)
    training_indices = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)
    training_probability = probabilities[training_indices[0]]
    training_parameter = parameters[training_indices[1]]

    # Validate with test set
    rewards = reward_processing(reward_data, rewards_key, training_probability)
    post_dist, post_dist_0, post_dist_1, post_dist_2, post_dist_3 = train_model(rewards, training_parameter)


    # Extract post_dists for overall performance
    all_post_0 = np.concatenate([all_post_0, post_dist_0])
    all_post_1 = np.concatenate([all_post_1, post_dist_1])
    all_post_2 = np.concatenate([all_post_2, post_dist_2])
    all_post_3 = np.concatenate([all_post_3, post_dist_3])

    # Compute the log-likelihood of Test-Set
    predictions = np.array([])
    actual_choices = np.array([])
    for trial in test_set:
        idx = int(trial['X'])
        model_trial = post_dist[idx]  # Access the first trial
        posterior, distance = model_trial
        predictions = np.append(predictions, posterior)
        actual_choices = np.append(actual_choices, int(trial['optimality'] == 'TRUE'))

    test_log_likelihood = compute_log_likelihood(predictions, actual_choices)

    # Log the results
    logging.info(f"Participant {exp_key}:")
    logging.info(f"  Best Probability: {training_probability}")
    logging.info(f"  Best Parameter: {training_parameter}")
    logging.info(f"  Log-Likelihood on the training set: {training_log_likelihood}")
    logging.info(f"  Log-Likelihood on the test set: {test_log_likelihood}")
    

    # Combine the data for the logistic regression
    distances_post = np.concatenate([np.full_like(post_dist_0, 0),
                                np.full_like(post_dist_1, 1),
                                np.full_like(post_dist_2, 2),
                                np.full_like(post_dist_3, 3)])  # X-axis (distances)

    posteriors = np.concatenate([post_dist_0, post_dist_1, post_dist_2, post_dist_3])  # Y-axis (probabilities)


    # Fit a logistic regression model
    X_post = sm.add_constant(distances_post)  # Add an intercept term
    model_post = sm.GLM(posteriors, X_post, family=sm.families.Binomial())
    result_post = model_post.fit()


    # Go over the experimental data to fit a logistic regression model
    # Flatten experimental data
    distances_exp = []
    optimality = []

    for trial in exp_data_participant:
        distances_exp.append(trial["distance"])
        optimality.append(int(trial['optimality'] == 'TRUE'))

    distances_exp = np.array(distances_exp, dtype=float)
    optimality = np.array(optimality, dtype=float)

    # Add intercept term
    X_exp = sm.add_constant(distances_exp)

    # Fit logistic regression model for experimental data
    model_exp = sm.GLM(optimality, X_exp, family=sm.families.Binomial())
    result_exp = model_exp.fit()

    # --- Step 3: Generate Predictions for Plotting ---
    # Generate a range of distances
    x_range = np.linspace(0, 3, 100)
    X_range = sm.add_constant(x_range)

    # Predict probabilities
    y_range_post = result_post.predict(X_range)  # Posterior data
    y_range_exp = result_exp.predict(X_range)  # Experimental data

    # --- Step 4: Plot Results ---
    plt.figure(figsize=(10, 6))

    # Plot experimental data
    plt.scatter(distances_exp, optimality, color='blue', label='Experimental Data', alpha=0.6)
    plt.plot(x_range, y_range_exp, color='blue', label='Logistic Regression (Experimental)', linestyle='--')

    # Plot posterior data
    plt.scatter(distances_post, posteriors, color='red', label='Posterior Data', alpha=0.6)
    plt.plot(x_range, y_range_post, color='red', label='Logistic Regression (Posterior)', linestyle='-')

    # Labels, legend, and title
    plt.xlabel('Distance')
    plt.ylabel('Probability')
    plt.legend()
    plt.title('Logistic Regression: Experimental vs Posterior Data')

    logging.info(f"  Regression Summary (Model Data):\n{result_post.summary()}")
    logging.info(f"  Regression Summary (Experimental Data):\n{result_exp.summary()}")

    # Save the plot
    if not os.path.exists("complete_plots"):
        os.makedirs("complete_plots")
    plot_path = f"complete_plots/{exp_key}_logistic_regression.png"
    plt.savefig(plot_path)
    plt.close()

    logging.info(f"Finished with Participant {exp_key}.")

    return all_matrix, all_post_0, all_post_1, all_post_2, all_post_3

    

# ----------------------------------------------------------------------------------------------------------------------
# Define Grid Search parameters
probabilities = [0.99, 0.975, 0.95, 0.92, 0.91, 0.9, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.82, 0.8, 0.78, 0.75, 0.7]	
parameters = [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 30.0, 50.0]


# Load data
maze_data = load_maze_data('maze_data.csv')
reward_data = load_reward_data(r'C:\Users\valen\OneDrive\Dokumente\7Semester\Bachelorarbeit\maze_planning\reward_data')

""" # Only take the first2 participants for testing
maze_data = {key: maze_data[key] for key in list(maze_data.keys())[:1]} """

# Initialize matrix to store post_dist_0, ..., post_dist_3 depending on the probability and parameter
# The matrix has the shape len(probabilities) x len(parameters) x 4
all_matrix = np.zeros((len(probabilities), len(parameters)))
                  
all_post_0 = np.empty((0,), dtype=float)
all_post_1 = np.empty((0,), dtype=float)
all_post_2 = np.empty((0,), dtype=float)
all_post_3 = np.empty((0,), dtype=float)

# Iterate over exp_data
for exp_key in maze_data:
    # Generate corresponding key for reward_data
    reward_key = f"rewards_{exp_key}"
    exp_data_participant = maze_data[exp_key]
    
    matrix = np.zeros((len(probabilities), len(parameters)))
    
    # Check if the reward key exists in reward_data
    if reward_key in reward_data:
        # Access data from both dictionaries
        all_matrix, all_post_0, all_post_1, all_post_2, all_post_3 = run_partipant(exp_key, reward_key, reward_data, exp_data_participant, matrix, probabilities, parameters, all_matrix, all_post_0, all_post_1, all_post_2, all_post_3)
    else: 
        logging.warning(f"Reward key {reward_key} not found in reward_data")
        continue

    logging.info("Percent processed: {:.2f}%".format((list(maze_data.keys()).index(exp_key) + 1) / len(maze_data) * 100))

logging.info("All participants processed.")

# Get the maximum of the all_matrix
max_log_likelihood = np.max(all_matrix)
max_indices = np.unravel_index(np.argmax(all_matrix, axis=None), all_matrix.shape)
best_probability = probabilities[max_indices[0]]
best_parameter = parameters[max_indices[1]]

logging.info(f"Best Probability over all Participants: {best_probability}")
logging.info(f"Best Parameter over all Participants: {best_parameter}")

# Combine the data
distances_all = np.concatenate([np.full_like(all_post_0, 0),
                            np.full_like(all_post_1, 1),
                            np.full_like(all_post_2, 2),
                            np.full_like(all_post_3, 3)])  # X-axis (distances)

posteriors_all = np.concatenate([all_post_0, all_post_1, all_post_2, all_post_3])  # Y-axis (probabilities)

# Fit a logistic regression model
X_all = sm.add_constant(distances_all)  # Add an intercept term
model_all = sm.GLM(posteriors_all, X_all, family=sm.families.Binomial())
result_all = model_all.fit()


# Flatten experimental data
distances_exp = []
optimality = []

for exp_key in maze_data:
    exp_data_participant = maze_data[exp_key]
    for trial in exp_data_participant:
        distances_exp.append(trial["distance"])
        optimality.append(int(trial['optimality'] == 'TRUE'))

distances_exp = np.array(distances_exp, dtype=float)
optimality = np.array(optimality, dtype=float)

# Add intercept term
X_exp = sm.add_constant(distances_exp)

# Fit logistic regression model for experimental data
model_exp = sm.GLM(optimality, X_exp, family=sm.families.Binomial())
result_exp = model_exp.fit()

# --- Step 3: Generate Predictions for Plotting ---
# Generate a range of distances
x_range = np.linspace(0, 3, 100)
X_range = sm.add_constant(x_range)

# Predict probabilities
y_range_post = result_all.predict(X_range)  # Posterior data
y_range_exp = result_exp.predict(X_range)  # Experimental data

# --- Step 4: Plot Results ---
plt.figure(figsize=(10, 6))

# Plot experimental data
plt.scatter(distances_exp, optimality, color='blue', label='Experimental Data', alpha=0.6)
plt.plot(x_range, y_range_exp, color='blue', label='Logistic Regression (Experimental)', linestyle='--')

# Plot posterior data
plt.scatter(distances_all, posteriors_all, color='red', label='Posterior Data', alpha=0.6)
plt.plot(x_range, y_range_post, color='red', label='Logistic Regression (Posterior)', linestyle='-')

# Labels, legend, and title
plt.xlabel('Distance')
plt.ylabel('Probability')
plt.legend()
plt.title('Logistic Regression: Experimental vs Posterior Data')

logging.info(f"  Regression Summary (Model Data):\n{result_all.summary()}")
logging.info(f"  Regression Summary (Experimental Data):\n{result_exp.summary()}")

# Save the plot
if not os.path.exists("complete_plots"):
    os.makedirs("_complete_plots")
plot_path = f"complete_plots/all_logistic_regression.png"
plt.savefig(plot_path)
plt.close()
