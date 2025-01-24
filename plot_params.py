import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
results = pd.read_csv('grid_search_result.csv')
opt_params = results.loc[results.groupby(['id', 'model_type'])['ll'].idxmax()]

# Create a JointGrid for scatter + marginal histograms
g = sns.JointGrid(data=opt_params, x='alpha', y='rho', height=6)
# Plot regression line + scatter in the joint area
g.plot_joint(
    sns.regplot, 
    ci=95,
    scatter_kws={'alpha':0.6}
)
# Set x-axis to log scale
g.ax_joint.set_xscale('log')

# Plot marginal histograms
g.plot_marginals(sns.histplot, kde=False, bins=20, alpha=0.6)

g.ax_joint.set_xlabel('Alpha (log scale)')
g.ax_joint.set_ylabel('Rho')
g.ax_joint.grid(True)
g.fig.suptitle('Scatter Plot of Alpha and Rho with Marginal Histograms', y=1.02)

plt.show()