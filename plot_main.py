import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem

# Create 2x2 subplot layout
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))

# Plot 1: Main effect (distance plot)
df = pd.read_csv('./data_files/maze_data_fitted.csv')
df_30 = df.loc[df['trial'] >= 30, :]
grouped = df_30.groupby('distance')
means = grouped[['optimality', 'full_prediction']].mean()
stds = grouped[['optimality', 'full_prediction']].sem()

ax1.errorbar(means.index, means['optimality'], yerr=stds['optimality'],
            label='data', marker='o', capsize=5, linestyle='-', zorder=3)
ax1.errorbar(means.index, means['full_prediction'], yerr=stds['full_prediction'],
            label='model', marker='o', capsize=5, linestyle='-', zorder=3)
ax1.set_xlabel('Distance from Habitual Path')
ax1.set_ylabel('Probability of Optimal Choice')
ax1.set_title('Probability of Detecting Optimal Path')
ax1.set_ylim(0, 1)
ax1.set_xticks(means.index)
ax1.legend()
ax1.grid(True)

# Plot 2: Habit test trials
df_habit = df[df['habit_test_trial'] == 1]
df_habit_by_id = df_habit.groupby(['id', 'n_habit_test_trial']).apply(
    lambda x: pd.Series({
        'prop_habit': (x['habit_path'] == x['chosen_path']).mean(),
        'mean_prior_habit': x['full_prior_habit'].mean()
    })
).reset_index()

stats_df = df_habit_by_id.groupby('n_habit_test_trial').agg({
    'prop_habit': ['mean', sem],
    'mean_prior_habit': ['mean', sem]
}).reset_index()

ax2.errorbar(stats_df['n_habit_test_trial'], stats_df['prop_habit']['mean'], 
             yerr=stats_df['prop_habit']['sem'], fmt='o-', capsize=5, label='data')
ax2.errorbar(stats_df['n_habit_test_trial'], stats_df['mean_prior_habit']['mean'],
             yerr=stats_df['mean_prior_habit']['sem'], fmt='s-', capsize=5, label='model')
ax2.axhline(y=0.125, color='gray', linestyle='--')

ax2.set_xlabel('Trials')
ax2.set_ylabel('Probability of Habitual Choice')
ax2.set_ylim((0, 1))
ax2.set_xticks([1, 2, 3, 4])
ax2.set_xticklabels(['20', '40', '60', '80'])
ax2.set_title('Behavior in "Blind" Trials')
ax2.legend()
ax2.grid(True)

# Plot 3: Parameter scatter
results = pd.read_csv('grid_search_result.csv')
opt_params = results.loc[results.groupby(['id', 'model_type'])['ll'].idxmax()]
sns.regplot(x='alpha', y='rho', data=opt_params, ci=95, 
            scatter_kws={'alpha':0.6}, ax=ax3)
ax3.set_xscale('log')
ax3.set_xlabel(r'$\alpha$ (learning rate)')
ax3.set_ylabel(r'$\rho$ (focus)')
ax3.set_title('Model Parameters')
ax3.set_ylim((.65, 1))
ax3.grid(True)

# Plot 4: Time course coefficients
coefs = pd.read_csv('./data_files/coefs.csv')
coefs = coefs[coefs['coef'] >= -4]  # exclude outliers

for coef_type, group in coefs.groupby('type'):
    # Calculate mean and sem for each period
    means = group.groupby('period')['coef'].mean()
    sems = group.groupby('period')['coef'].apply(sem)
    
    ax4.errorbar(
        means.index,
        means.values,
        yerr=sems.values,
        marker='o',
        capsize=5,
        label=coef_type
    )

ax4.set_xlabel('Trials')
ax4.set_ylabel('Coefficient')
ax4.set_title('Time Course of Coefficients')
ax4.set_ylim((-1, 0))
ax4.set_xticks([0, 25, 50, 75])
ax4.set_xticklabels(['1-25', '26-50', '51-75', '76-100'])
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show()

fig.savefig('./figs/main_plot.pdf')

# Main Effect: Max Reward on Optimal Path {true, false} 
df_rewards = pd.read_csv('./data_files/reward_analysis.csv')

