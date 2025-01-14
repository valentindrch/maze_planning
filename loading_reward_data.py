import os
import pickle
import numpy as np
import json



def load_reward_data(folder_path):
    reward_data = {}
    for filename in os.listdir(folder_path):
        file_extension = os.path.splitext(filename)[1]
        if file_extension == '' or file_extension == '.pkl':
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
                    reward_data[filename] = data
            except Exception as e:
                print(f"Failed to load file: {file_path}, error: {e}")
    return reward_data

def softmax(vector, temperature=1.0):
    return np.exp(vector/temperature) / np.sum(np.exp(vector/temperature), axis=0)

# Replaces the softmax function with a function which gives the optimal path .99 and all the other .01
def one_hot_encoding(vector, probability):
    optimal_path = np.argmax(vector)
    softmax_values = np.zeros(len(vector))
    softmax_values[optimal_path] = probability
    softmax_values[softmax_values == 0] = 1 - probability
    return softmax_values

def reward_processing(probability):
    folder_path = r'C:\Users\valen\OneDrive\Dokumente\7Semester\Bachelorarbeit\maze_planning\reward_data'
    reward_data = load_reward_data(folder_path)

    # Generalized processing for all keys in reward_data
    reward_distributions = {}

    for subject_key, subject_data in reward_data.items():
        processed_trials = []
        
        for item in subject_data:
            trial_array = item[0]  # The main NumPy array
            extra_elements = item[1:]  # The extra elements
            
            # Sum each row of the trial array
            row_sums = np.sum(trial_array, axis=1)
            
            # Compute softmax and its complement
            encoded_values = one_hot_encoding(row_sums, probability)
            # encoded_values = softmax(row_sums)
            complement_values = 1 - encoded_values
            
            # Combine softmax values and complements, keeping the extra elements
            processed_trial = {
                "distribution": np.stack([encoded_values, complement_values]),
                "extra_elements": extra_elements
            }
            
            processed_trials.append(processed_trial)
        
        # Store the processed trials for the current subject
        reward_distributions[subject_key] = processed_trials

    return reward_distributions
    """ if key in reward_distributions:
        return reward_distributions[key]
    else:
        print(f"Subject key '{key}' not found in reward data")
        return None """


""" # Generalized processing for all keys in reward_data
reward_distributions = {}

for subject_key, subject_data in reward_data.items():
    # Extract only the NumPy arrays
    trials = [item[0] for item in subject_data]

    # Sum each row of trials
    sums = np.array([np.sum(trial, axis=1) for trial in trials])

    # Compute reward distributions (values and complements)
    reward_distributions[subject_key] = np.array([
        np.concatenate([[softmax(trial)], [1 - softmax(trial)]]) for trial in sums
    ])
 """

""" # Determine Habitual Path of a Participant
def habitual_path(data):
    for i in range(len(data)):
        if data[i]['extra_elements'][0] == 0:
            return data[i]['extra_elements'][1]
        else:
            pass

# List the habitual paths of all participants
def all_habitual_paths(reward_data):
    # List the subject key and its habitual path
    return {subject_key: habitual_path(data) for subject_key, data in reward_data.items()}
 """

