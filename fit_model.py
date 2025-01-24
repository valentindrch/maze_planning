import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import logging

from model import PlanningModel
from model_fitting_utils import split_data, k_cross_validation

# Suppress warnings from pgmpy
logging.getLogger("pgmpy").setLevel(logging.ERROR)

# Define Grid Search parameters
alphas = np.logspace(-4, 4, 20, base=np.e).round(2)  # from .02 to 55 ish
rhos = np.linspace(.7, .99, 20)

# Load data
exp_data = pd.read_csv('maze_data.csv')
exp_data['subset'] = ''
exp_data['full_prediction'] = np.nan
exp_data['full_prior_habit'] = np.nan
exp_data['full_ll'] = np.nan
exp_data['full_alpha'] = np.nan
exp_data['full_rho'] = np.nan
exp_data['learn_prediction'] = np.nan
exp_data['learn_ll'] = np.nan
exp_data['learn_alpha'] = np.nan
exp_data['plan_prediction'] = np.nan
exp_data['plan_ll'] = np.nan
exp_data['plan_rho'] = np.nan
ids = np.unique(exp_data['id'])

# Define train and test sets (in original dataframe)
for i in ids:
    exp_data_participant = exp_data.loc[exp_data['id'] == i, :].to_dict(orient='records')
    train_set, test_set = split_data(exp_data_participant)

    exp_data.loc[(exp_data['id'] == i) & 
                 (exp_data['trial'].isin(pd.DataFrame(train_set)['trial']._values)), 
                 'subset'] = 'train'
    exp_data.loc[(exp_data['id'] == i) & 
                 (exp_data['trial'].isin(pd.DataFrame(test_set)['trial']._values)), 
                 'subset'] = 'test'

def run_model(data, alpha, rho, model_type='full'):

    # Init model
    model = PlanningModel(alpha=alpha, rho=rho)

    # Iterate through trials
    predictions = {'trial': [], 'prediction': [], 'prior': []}
    for trial in data['trial']._values:
        
        goal = data.loc[data['trial'] == trial, 'optimal_path']

        match model_type:
            case 'full':
                model.plan(goal)
                model.learn()
                predictions['prediction'].append(float(model.path_pred[goal]))
            case 'learn_only':
                model.plan(goal)
                model.learn()
                predictions['prediction'].append(float(model.prior.sum(axis=(1, 3, 5)).ravel()[goal]))
            case 'plan_only':
                model.plan(goal)
                predictions['prediction'].append(float(model.path_pred[goal]))

        prior_a = model.prior.sum(axis=(1, 3, 5)).ravel()
        prior_habit = prior_a[np.unique(data['habit_path'])[0]]
        predictions['prior'].append(prior_habit)

        predictions['trial'].append(trial)

    return pd.DataFrame(predictions)

def fit_partipant(id, alphas, rhos):

    exp_data_id = exp_data.loc[(exp_data['id'] == id), :]
    actual_choices = exp_data_id['optimality'].astype(int)._values
    
    # Grid Search
    result = {'id': [], 'rho': [], 'alpha': [], 'll': [], 'model_type': []}
    for rho in rhos:
        for alpha in alphas:

            predictions = run_model(exp_data_id, alpha, rho)
            mean_log_likelihood = k_cross_validation(predictions['prediction']._values, actual_choices, k=5)

            result['id'].append(id)
            result['rho'].append(rho)
            result['alpha'].append(alpha)
            result['ll'].append(mean_log_likelihood)
            result['model_type'].append('full')
    
    
    for alpha in alphas:

            predictions = run_model(exp_data_id, alpha=alpha, rho=.95, model_type='learn_only')
            mean_log_likelihood = k_cross_validation(predictions['prediction']._values, actual_choices, k=5)

            result['id'].append(id)
            result['rho'].append(.95)
            result['alpha'].append(alpha)
            result['ll'].append(mean_log_likelihood)
            result['model_type'].append('learn_only')

    for rho in rhos:

            predictions = run_model(exp_data_id, alpha=1, rho=rho, model_type='plan_only')
            mean_log_likelihood = k_cross_validation(predictions['prediction']._values, actual_choices, k=5)

            result['id'].append(id)
            result['rho'].append(rho)
            result['alpha'].append(1)
            result['ll'].append(mean_log_likelihood)
            result['model_type'].append('plan_only')

    return pd.DataFrame(result)

def predict(parameters):

    for i in ids:

        exp_data_id = exp_data.loc[(exp_data['id'] == i), :]
        true_vals = exp_data_id['optimality'].astype(int)._values

        # Full model
        alpha = parameters.loc[(parameters['id'] == i) & (parameters['model_type'] == 'full'), 'alpha']._values[0]
        rho = parameters.loc[(parameters['id'] == i) & (parameters['model_type'] == 'full'), 'rho']._values[0]

        predictions = run_model(exp_data_id, alpha, rho)
        pred_vals = predictions['prediction']._values

        exp_data.loc[exp_data['id'] == i, 'full_prediction'] = pred_vals
        exp_data.loc[exp_data['id'] == i, 'full_prior_habit'] = predictions['prior']._values
        exp_data.loc[exp_data['id'] == i, 'full_ll'] = np.log(pred_vals) * true_vals + np.log(1 - pred_vals) * (1 - true_vals)
        exp_data.loc[exp_data['id'] == i, 'full_alpha'] = alpha
        exp_data.loc[exp_data['id'] == i, 'full_rho'] = rho

        # Learning-only model
        alpha = parameters.loc[(parameters['id'] == i) & (parameters['model_type'] == 'learn_only'), 'alpha']._values[0]
        rho = parameters.loc[(parameters['id'] == i) & (parameters['model_type'] == 'learn_only'), 'rho']._values[0]

        predictions = run_model(exp_data_id, alpha, rho, model_type='learn_only')
        pred_vals = predictions['prediction']._values

        exp_data.loc[exp_data['id'] == i, 'learn_prediction'] = pred_vals
        exp_data.loc[exp_data['id'] == i, 'learn_ll'] = np.log(pred_vals) * true_vals + np.log(1 - pred_vals) * (1 - true_vals)
        exp_data.loc[exp_data['id'] == i, 'learn_alpha'] = alpha
        exp_data.loc[exp_data['id'] == i, 'learn_rho'] = rho

        # Planning-only model
        alpha = parameters.loc[(parameters['id'] == i) & (parameters['model_type'] == 'plan_only'), 'alpha']._values[0]
        rho = parameters.loc[(parameters['id'] == i) & (parameters['model_type'] == 'plan_only'), 'rho']._values[0]

        predictions = run_model(exp_data_id, alpha, rho, model_type='plan_only')
        pred_vals = predictions['prediction']._values

        exp_data.loc[exp_data['id'] == i, 'plan_prediction'] = pred_vals
        exp_data.loc[exp_data['id'] == i, 'plan_ll'] = np.log(pred_vals) * true_vals + np.log(1 - pred_vals) * (1 - true_vals)
        exp_data.loc[exp_data['id'] == i, 'plan_alpha'] = alpha
        exp_data.loc[exp_data['id'] == i, 'plan_rho'] = rho

# Fit parameters
if False:
    results = Parallel(n_jobs=9)(
            delayed(fit_partipant)(id, alphas, rhos) 
            for id in ids
        )
    results = pd.concat(results)
    results = results.reset_index(drop=True)
    results.to_csv('grid_search_result.csv')
else:
    results = pd.read_csv('grid_search_result.csv')

# Make predictions
opt_params = results.loc[results.groupby(['id', 'model_type'])['ll'].idxmax()]
predict(opt_params)
exp_data.to_csv('maze_data_fitted.csv')


a = 0