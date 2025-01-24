import pandas as pd
import numpy as np


df = pd.read_csv('./data_files/maze_data_fitted.csv')
df = df.loc[df['subset'] == 'test', :]

bic_full = 2 * np.log(len(df)) - 2 * df['full_ll'].sum()
bic_plan = 1 * np.log(len(df)) - 2 * df['plan_ll'].sum()
bic_learn = 1 * np.log(len(df)) - 2 * df['learn_ll'].sum()

a = 1