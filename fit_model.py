import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import logging

from model import PlanningModel
from model_fitting_utils import split_data, k_cross_validation, load_reward_data, max_rewards_trial

# Suppress warnings from pgmpy
logging.getLogger("pgmpy").setLevel(logging.ERROR)


# Define Grid Search parameters
alphas = np.logspace(-4, 4, 20, base=np.e).round(2)  # from .02 to 55 ish
rhos = np.linspace(.7, .99, 20)
kappas = np.linspace(.5, .9, 20)

# Load reward data
reward_data = load_reward_data('./data_files/reward_data')

# Load data
exp_data = pd.read_csv('./data_files/maze_data.csv')
exp_data['subset'] = ''
exp_data['full_prediction'] = np.nan
exp_data['full_prior_habit'] = np.nan
exp_data['full_surprise'] = np.nan
exp_data['full_cost'] = np.nan
exp_data['full_error'] = np.nan
exp_data['full_ll'] = np.nan
exp_data['full_alpha'] = np.nan
exp_data['full_rho'] = np.nan
exp_data['full_kappa'] = np.nan
exp_data['learn_prediction'] = np.nan
exp_data['learn_ll'] = np.nan
exp_data['learn_alpha'] = np.nan
exp_data['learn_kappa'] = np.nan
exp_data['plan_prediction'] = np.nan
exp_data['plan_surprise'] = np.nan
exp_data['plan_cost'] = np.nan
exp_data['plan_error'] = np.nan
exp_data['plan_ll'] = np.nan
exp_data['plan_rho'] = np.nan
exp_data['plan_kappa'] = np.nan
ids = np.unique(exp_data['id'])
ids = ids[:1]

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

def run_model(data, reward_data, alpha, rho, kappa, model_type='full'):

    # Init model
    model = PlanningModel(alpha=alpha, rho=rho, kappa=kappa)

    # Iterate through trials
    predictions = {'trial': [], 'prediction': [], 'prior': [], 'surprise': [], 'cost': [], 'error': []}
    for trial in data['trial']._values:
        
        goal = data.loc[data['trial'] == trial, 'optimal_path']

        # Get reward maxima for each trial
        maxima = max_rewards_trial(reward_data[trial][0])

        match model_type:
            case 'full':
                model.plan(goal, maxima)
                cost, error, surprise =  model.get_it_measures()
                model.learn()
                predictions['prediction'].append(float(model.path_pred[goal]))
                predictions['surprise'].append(surprise)
                predictions['cost'].append(cost)
                predictions['error'].append(error)
            case 'learn_only':
                model.plan(goal, maxima)
                cost, error, surprise =  model.get_it_measures()  # TODO  
                model.learn()
                predictions['prediction'].append(float(model.prior.sum(axis=(1, 3, 5)).ravel()[goal]))
                predictions['surprise'].append(surprise)
                predictions['cost'].append(cost)
                predictions['error'].append(error)
            case 'plan_only':
                model.plan(goal, maxima)
                cost, error, surprise =  model.get_it_measures()
                predictions['prediction'].append(float(model.path_pred[goal]))
                predictions['surprise'].append(surprise)
                predictions['cost'].append(cost)
                predictions['error'].append(error)

        prior_a = model.prior.sum(axis=(1, 3, 5)).ravel()
        prior_habit = prior_a[np.unique(data['habit_path'])[0]]
        predictions['prior'].append(prior_habit)

        predictions['trial'].append(trial)

    return pd.DataFrame(predictions)

def fit_partipant(id, alphas, rhos, kappas):

    exp_data_id = exp_data.loc[(exp_data['id'] == id), :]
    reward_data_id = reward_data[str(id)]
    actual_choices = exp_data_id['optimality'].astype(int)._values
    
    # Grid Search
    result = {'id': [], 'rho': [], 'alpha': [], 'kappa': [], 'll': [], 'model_type': []}

    print(f'Fitting full model for participant {id}')

    for rho in rhos:
        print(f'Progress: {round((rho / len(rhos)) * 100, 2)}%')
        for alpha in alphas:
            for kappa in kappas:
                predictions = run_model(exp_data_id, reward_data_id, alpha, rho, kappa)
                mean_log_likelihood = k_cross_validation(predictions['prediction']._values, actual_choices, k=5)

                result['id'].append(id)
                result['rho'].append(rho)
                result['alpha'].append(alpha)
                result['kappa'].append(kappa)
                result['ll'].append(mean_log_likelihood)
                result['model_type'].append('full')

            

    print('Fitting learning model')
    
    for alpha in alphas:

            predictions = run_model(exp_data_id, reward_data_id, alpha=alpha, rho=.95, kappa=0.5, model_type='learn_only')
            mean_log_likelihood = k_cross_validation(predictions['prediction']._values, actual_choices, k=5)

            result['id'].append(id)
            result['rho'].append(.95)
            result['alpha'].append(alpha)
            result['kappa'].append(0.5)
            result['ll'].append(mean_log_likelihood)
            result['model_type'].append('learn_only')

    print('Fitting planning model')

    for rho in rhos:
        for kappa in kappas:
            
            predictions = run_model(exp_data_id, reward_data_id, alpha=1, rho=rho, kappa=kappa, model_type='plan_only')
            mean_log_likelihood = k_cross_validation(predictions['prediction']._values, actual_choices, k=5)

            result['id'].append(id)
            result['rho'].append(rho)
            result['alpha'].append(1)
            result['kappa'].append(kappa)
            result['ll'].append(mean_log_likelihood)
            result['model_type'].append('plan_only')

    return pd.DataFrame(result)

def predict(parameters):

    for i in ids:

        print(f'Predicting for participant {i}')
        print(f'Percentage done: {round((i / len(ids)) * 100, 2)}%')

        exp_data_id = exp_data.loc[(exp_data['id'] == i), :]
        reward_data_id = reward_data[str(i)]
        true_vals = exp_data_id['optimality'].astype(int)._values

        # Full model
        alpha = parameters.loc[(parameters['id'] == i) & (parameters['model_type'] == 'full'), 'alpha']._values[0]
        rho = parameters.loc[(parameters['id'] == i) & (parameters['model_type'] == 'full'), 'rho']._values[0]
        kappa = parameters.loc[(parameters['id'] == i) & (parameters['model_type'] == 'full'), 'kappa']._values[0]

        predictions = run_model(exp_data_id, reward_data_id, alpha, rho)
        pred_vals = predictions['prediction']._values

        exp_data.loc[exp_data['id'] == i, 'full_prediction'] = pred_vals
        exp_data.loc[exp_data['id'] == i, 'full_prior_habit'] = predictions['prior']._values
        exp_data.loc[exp_data['id'] == i, 'full_surprise'] = predictions['surprise']._values
        exp_data.loc[exp_data['id'] == i, 'full_cost'] = predictions['cost']._values
        exp_data.loc[exp_data['id'] == i, 'full_error'] = predictions['error']._values
        exp_data.loc[exp_data['id'] == i, 'full_ll'] = np.log(pred_vals) * true_vals + np.log(1 - pred_vals) * (1 - true_vals)
        exp_data.loc[exp_data['id'] == i, 'full_alpha'] = alpha
        exp_data.loc[exp_data['id'] == i, 'full_rho'] = rho
        exp_data.loc[exp_data['id'] == i, 'full_kappa'] = kappa

        # Learning-only model
        alpha = parameters.loc[(parameters['id'] == i) & (parameters['model_type'] == 'learn_only'), 'alpha']._values[0]
        rho = parameters.loc[(parameters['id'] == i) & (parameters['model_type'] == 'learn_only'), 'rho']._values[0]
        kappa = parameters.loc[(parameters['id'] == i) & (parameters['model_type'] == 'learn_only'), 'kappa']._values[0]
        

        predictions = run_model(exp_data_id, alpha, rho, model_type='learn_only')
        pred_vals = predictions['prediction']._values

        exp_data.loc[exp_data['id'] == i, 'learn_prediction'] = pred_vals
        exp_data.loc[exp_data['id'] == i, 'learn_ll'] = np.log(pred_vals) * true_vals + np.log(1 - pred_vals) * (1 - true_vals)
        exp_data.loc[exp_data['id'] == i, 'learn_alpha'] = alpha
        exp_data.loc[exp_data['id'] == i, 'learn_rho'] = rho
        exp_data.loc[exp_data['id'] == i, 'learn_kappa'] = kappa

        # Planning-only model
        alpha = parameters.loc[(parameters['id'] == i) & (parameters['model_type'] == 'plan_only'), 'alpha']._values[0]
        rho = parameters.loc[(parameters['id'] == i) & (parameters['model_type'] == 'plan_only'), 'rho']._values[0]
        kappa = parameters.loc[(parameters['id'] == i) & (parameters['model_type'] == 'plan_only'), 'kappa']._values[0]

        predictions = run_model(exp_data_id, alpha, rho, model_type='plan_only')
        pred_vals = predictions['prediction']._values

        exp_data.loc[exp_data['id'] == i, 'plan_prediction'] = pred_vals
        exp_data.loc[exp_data['id'] == i, 'plan_surprise'] = predictions['surprise']._values
        exp_data.loc[exp_data['id'] == i, 'plan_cost'] = predictions['cost']._values
        exp_data.loc[exp_data['id'] == i, 'plan_error'] = predictions['error']._values
        exp_data.loc[exp_data['id'] == i, 'plan_ll'] = np.log(pred_vals) * true_vals + np.log(1 - pred_vals) * (1 - true_vals)
        exp_data.loc[exp_data['id'] == i, 'plan_alpha'] = alpha
        exp_data.loc[exp_data['id'] == i, 'plan_rho'] = rho
        exp_data.loc[exp_data['id'] == i, 'plan_kappa'] = kappa

# Fit parameters
if True:
    results = Parallel(n_jobs=3)(
            delayed(fit_partipant)(id, alphas, rhos, kappas) 
            for id in ids
        )
    results = pd.concat(results)
    results = results.reset_index(drop=True)
    results.to_csv('./data_files/grid_search_result.csv')
else:
    results = pd.read_csv('./data_files/grid_search_result.csv')

# Make predictions
opt_params = results.loc[results.groupby(['id', 'model_type'])['ll'].idxmax()]
predict(opt_params)
exp_data.to_csv('./data_files/maze_data_fitted.csv')


a = 0