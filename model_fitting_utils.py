import random
import numpy as np
from sklearn.model_selection import KFold


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

def transform_to_binary_tree(matrix):
    tree = {
        'Level 0': [0],  # Proxy reward for the empty layer 0
        'Level 1': [matrix[0][0], matrix[4][0]],  # state s1: l & r
        'Level 2': [matrix[0][1], matrix[2][1], matrix[4][1], matrix[6][1]],  # state s2: ll, lr, rl, rr
        'Level 3': [matrix[0][2], matrix[1][2], matrix[2][2], matrix[3][2],
                    matrix[4][2], matrix[5][2], matrix[6][2], matrix[7][2]]  # state s3: lll, llr, ..., rrr
    }
    return tree

def max_rewards(rewards):
    '''
    Input: A List of 8 arrays which depict the reward distribution for a specific trial.
    Output: Returns the states on which a maximum reward lies. 

    Example:
    Input: 
            [[3. 0. 1.]
            [3. 0. 1.]
            [3. 2. 2.]
            [3. 2. 2.]
            [3. 2. 2.]
            [3. 2. 3.]
            [3. 2. 2.]
            [3. 2. 2.]]

    Output: l in s1, r in s1, rlr in s3
    '''
    tree = transform_to_binary_tree(rewards)
    max_reward = max(sum(tree.values(), []))

    state_labels = {
        'Level 1': ['l', 'r'],
        'Level 2': ['ll', 'lr', 'rl', 'rr'],
        'Level 3': ['lll', 'llr', 'lrl', 'lrr', 'rll', 'rlr', 'rrl', 'rrr']
    }

    max_states = []
    for level, rewards in tree.items():
        if level == 'Level 0':
            continue
        for i, reward in enumerate(rewards):
            if reward == max_reward:
                max_states.append((state_labels[level][i], reward))

    return max_states

