import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

# Load data and calculate statistics
df = pd.read_csv('maze_data_fitted.csv')
df_30 = df.loc[df['trial'] >= 30, :]

grouped = df_30.groupby('distance')
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

# Plot habit test trials ##########################################################################
# 1. Filter rows where habit_test_trial == 1
df_habit = df[df['habit_test_trial'] == 1]

# 2. For each (id, n_habit_test_trial), calculate:
#    - The proportion of habit_path == chosen_path
#    - The mean full_prior_habit
df_habit_by_id = df_habit.groupby(['id', 'n_habit_test_trial']).apply(
    lambda x: pd.Series({
        'prop_habit': (x['habit_path'] == x['chosen_path']).mean(),
        'mean_prior_habit': x['full_prior_habit'].mean()
    })
).reset_index()

# 3. Now group by n_habit_test_trial (across ids) to get overall means + SEM
stats_df = df_habit_by_id.groupby('n_habit_test_trial').agg({
    'prop_habit': ['mean', sem],
    'mean_prior_habit': ['mean', sem]
})
stats_df.columns = [
    'prop_habit_mean', 'prop_habit_sem',
    'prior_mean', 'prior_sem'
]
stats_df = stats_df.reset_index()

# 4. Plot with twin axes
fig, ax1 = plt.subplots(figsize=(6,4))

# Plot proportion of habit path chosen
color1 = 'tab:blue'
ax1.errorbar(
    stats_df['n_habit_test_trial'], 
    stats_df['prop_habit_mean'], 
    yerr=stats_df['prop_habit_sem'], 
    fmt='o-', 
    color=color1, 
    capsize=5, 
    label='Proportion habit path chosen'
)
ax1.set_xlabel('n_habit_test_trial')
ax1.set_ylabel('Proportion of Habit == Chosen Path', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 1)
ax1.grid(True)
ax1.axhline(0.125, color='gray', linestyle='--')

# Twin axis for mean_prior_habit
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.errorbar(
    stats_df['n_habit_test_trial'], 
    stats_df['prior_mean'], 
    yerr=stats_df['prior_sem'],
    fmt='s-', 
    color=color2, 
    capsize=5, 
    label='Mean full_prior_habit'
)
ax2.set_ylabel('Mean Full Prior Habit', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, 1)

# Optional: Add a combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.title('Habit Path vs. Chosen Path in Habit Test Trials (Mean ± SEM)')
plt.tight_layout()
plt.show()