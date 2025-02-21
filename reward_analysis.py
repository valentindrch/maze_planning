import os
import pickle
import numpy as np
import json
import pandas as pd

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

def find_max_reward_paths(paths):
    # Find the maximum singular reward
    max_reward = np.max(paths)

    # Identify the paths that contain the maximum singular reward
    max_indices = [i for i, path in enumerate(paths) if max_reward in path]

    # Count occurrences based on binary tree structure for the first two columns
    unique_counts = 0
    seen_rewards = set()
    for col in range(paths.shape[1]):
        if col < 2:  # For the first two layers, count each reward only once
            unique_values = set(paths[:, col])
            if max_reward in unique_values and max_reward not in seen_rewards:
                unique_counts += 1
                seen_rewards.add(max_reward)
        else:  # For the third layer, count all occurrences
            unique_counts += np.sum(paths[:, col] == max_reward)

    return max_indices, max_reward, unique_counts



reward_data = load_reward_data(r'C:\Users\valen\OneDrive\Dokumente\7Semester\Bachelorarbeit\maze_planning\reward_data')

analysis_data = {'id': [], 'trials': [], 'optimal_path': [], 'highest_reward_path': [], 'number_max_rewards': [], 'in_optimal_path': [], 'in_optimal_path_percentage': [], 'only_one_max_reward_percentage': []}

# List of Reward Keys
reward_keys = list(reward_data.keys())

print(reward_data['rewards_1'][2][0])


# Iterate through reward data
for key in reward_keys:
    data = reward_data[key]
    for i in range(len(data)):
        analysis_data['id'].append(key)
        analysis_data['trials'].append(i)
        analysis_data['optimal_path'].append(data[i][2])
        max_indices, max_reward, unique_counts = find_max_reward_paths(data[i][0])
        analysis_data['highest_reward_path'].append(max_indices)
        analysis_data['number_max_rewards'].append(unique_counts)
        # Check if max_indices contains the optimal path and add TRUe or FALSE to the list
        analysis_data['in_optimal_path'].append(data[i][2] in max_indices)


# Calculate the percentage of trials where the highest reward path contains the optimal path
analysis_data['in_optimal_path_percentage'] = np.mean(analysis_data['in_optimal_path'])

# Calculate the percentage of trials where the highest reward path contains only one maximum reward
analysis_data['only_one_max_reward_percentage'] = np.mean(np.array(analysis_data['number_max_rewards']) == 1)


# Convert analysis_data to a DataFrame
df = pd.DataFrame(analysis_data)

# Save the DataFrame to a CSV file
df.to_csv('./data_files/reward_analysis.csv', index=False)


a=0