import pickle
import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window=10):
    return np.convolve(data, np.ones(window)/window, mode='valid')

# Load the IT measures from the pickle file
with open('it_measures.pkl', 'rb') as f:
    it_measures = pickle.load(f)

# Create a figure
plt.figure(figsize=(10, 6))

# Smooth and plot each measure
window = 20  # default window size
plt.plot(moving_average(it_measures['complexity'], window), label='Complexity', color='blue')
plt.plot(moving_average(it_measures['error'], window), label='Error', color='red')
plt.plot(moving_average(it_measures['surprise'], window), label='Surprise', color='green')

# Add labels and title
plt.xlabel('Trial')
plt.ylabel('Information (bits)')
plt.title('Information-Theoretic Measures over Trials (Smoothed)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()