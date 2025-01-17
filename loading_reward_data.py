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

def reward_processing(reward_data, key, probability):
    #folder_path = r'C:\Users\valen\OneDrive\Dokumente\7Semester\Bachelorarbeit\maze_planning\reward_data'

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

    if key in reward_distributions:
        return reward_distributions[key]
    else: 
        print(f"Key {key} not found in reward data")
        return None
    
# Write rewards_0 to a json file

def write_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

