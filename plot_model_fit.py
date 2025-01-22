import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data and calculate statistics
df = pd.read_csv('maze_data_fitted.csv')
df = df.iloc[30:, :]

grouped = df.groupby('distance')
means = grouped[['optimality', 'full_prediction']].mean()
stds = grouped[['optimality', 'full_prediction']].sem()

prop_ones = grouped['optimality'].apply(lambda x: (x == 1).mean())
prop_zeros = grouped['optimality'].apply(lambda x: (x == 0).mean())

# Create single plot
fig, ax1 = plt.subplots(figsize=(8, 4))

# Plot bars
ax1.bar(prop_ones.index, -0.2*prop_ones.values, bottom=1, color='green', alpha=0.3)
ax1.bar(prop_zeros.index, 0.2*prop_zeros.values, color='red', alpha=0.3)

# Plot main data with higher zorder
ax1.errorbar(means.index, means['optimality'], yerr=stds['optimality'],
            label='Human Data', marker='o', capsize=5, linestyle='-', zorder=3)
ax1.errorbar(means.index, means['full_prediction'], yerr=stds['full_prediction'],
            label='Model Prediction', marker='s', capsize=5, linestyle='-', zorder=3)

# Setup main axis
ax1.set_xlabel('Distance from Habitual Path')
ax1.set_ylabel('Probability of Optimal Choice')
ax1.set_title('Model Fit to Human Data')
ax1.set_ylim(0, 1)
ax1.set_xticks(means.index)
ax1.legend(loc='center left')
ax1.grid(True, zorder=0)

# Add twin axis for proportions
ax2 = ax1.twinx()
ax2.set_ylim(0, 1)
ax2.set_yticks([0, .1, .2, .8, .9, 1])
ax2.set_yticklabels(['0', '0.5', '1', '1', '0.5', '0'])
ax2.set_ylabel('Proportion')

plt.tight_layout()
plt.show()