import numpy as np
import matplotlib.pyplot as plt
import random 
import pickle
from reservoirpy.nodes import Reservoir, Ridge, ESN
from functools import partial
from sklearn.model_selection import ParameterGrid
import os

from  generate_data import generate_data, multi_harmonic, plot_train_data, plot_prediction
from train_model import grid_search, t_plus_1, forecast
import noise 
from scipy import signal


data_config = {
    'num_series': 250,                  # Number of series to generate (default: 5)
    'T': 2*np.pi,                     # Period of the function (default: 2*np.pi)
    'low': 0,                         # Lower bound for the x-axis (default: 0)
    'train_T': 20,                    # Length of training data (default: 10)
    'rate': 300,                      # Rate of sampling (default: 100)
    'warmup': 1,                    # Percentage of data to use as warmup (default: 0.5)
    'forecast': 3,                    # Number of points to forecast (default: 3)
    'amp_noise': .5,                 # Amplitude of added noise (default: 0.3)
    'same_start': False               # Whether all series should start at the same point (default: False)
}

# Best params for sin for t+1
param_grid = {
    'nodes': [10],  
    'lr': [.2],  
    'sr': [1.0],  
    'ridge': [1e-7],
    #'ridge': [1e-6]  
}

param_grid = {
    'nodes': [10, 100, 500],  
    'lr': [.2, 0.5, 0.8],  
    #'lr': [0.5],
    #'sr': [.8] ,
    'sr': [.5 ,0.8, 1.0],  
    #'ridge': [1e-7],
    'ridge': [1e-8 ,1e-7, 1e-6]  
}

if __name__ == '__main__':

    five_harmonic = partial(multi_harmonic, num_harmonics=5)
    fifteen_harmonic = partial(multi_harmonic, num_harmonics=15)


    five_harmonic_data = generate_data(five_harmonic, noise.sine_noise, data_config)
    fifteen_harmonic_data = generate_data(fifteen_harmonic, noise.sine_noise, data_config)

    # ! #################################### SAWTOOTH Forecase upgrade
    param_grid = {
    'nodes': [100, 500, 1000],  
    'lr': [.2, 0.5, 0.8],  
    #'lr': [0.5],
    #'sr': [.8] ,
    'sr': [.5 ,0.8, 1.0],  
    #'ridge': [1e-7],
    'ridge': [1e-8 ,1e-7]  
    }
    # Make new filename if running a new test
    save_file = 'results/f15har_nsin_amp5_ssF_tp1_ba.pkl'

    # Create directories if they don't exist
    # Create directories if they don't exist
    save_dir = os.path.dirname(save_file)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # check if the saved file exists
    try:
        with open(save_file, 'rb') as f:
            grid_search_results = pickle.load(f)
        best_params = grid_search_results['best_params']
        best_log_lik = grid_search_results['best_log_lik']
        Y_test = grid_search_results['Y_test']
        Y_pred = grid_search_results['Y_pred']
        model = grid_search_results['model']
        sigma = grid_search_results['sigma']
        print("Loaded grid search results from file.")
    except FileNotFoundError:
        print("Running grid search...")
        best_params, best_log_lik, Y_test, Y_pred, model, sigma = grid_search(fifteen_harmonic_data, param_grid, t_plus_1, save_file, False)
        print("Grid search completed and results saved.")

    train, validate, test = fifteen_harmonic_data
    #plot_prediction(X_warmup, Y_test, Y_pred, sigma)
    #print("Best params: ", best_params)

    results = forecast(model, validate, test, upgrade= True)

    model, Y_test, Y_pred, log_lik, sigma = results

    plot_prediction(test[0], Y_test, Y_pred, sigma)
    

"""

    # ! #################################### SAWTOOTH T+1 BASELINE 
    # Make new filename if running a new test
    
    save_file = 'results/f5har_nsin_amp5_ssF_tp1_ba.pkl'

    # Create directories if they don't exist
    save_dir = os.path.dirname(save_file)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # check if the saved file exists
    try:
        with open(save_file, 'rb') as f:
            grid_search_results = pickle.load(f)
        best_params = grid_search_results['best_params']
        best_log_lik = grid_search_results['best_log_lik']
        Y_test = grid_search_results['Y_test']
        Y_pred = grid_search_results['Y_pred']
        model = grid_search_results['model']
        sigma = grid_search_results['sigma']
        print("Loaded grid search results from file.")
    except FileNotFoundError:
        print("Running grid search...")
        best_params, best_log_lik, Y_test, Y_pred, model, sigma = grid_search(five_harmonic_data, param_grid, t_plus_1, save_file, False)
        print("Grid search completed and results saved.")
    

    # ! #################################### SAWTOOTH T+1 Upgrade 
    # Make new filename if running a new test
    save_file = 'results/f7har_nsin_amp5_ssF_tp1_ba.pkl'

    # Create directories if they don't exist
    save_dir = os.path.dirname(save_file)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # check if the saved file exists
    try:
        with open(save_file, 'rb') as f:
            grid_search_results = pickle.load(f)
        best_params = grid_search_results['best_params']
        best_log_lik = grid_search_results['best_log_lik']
        Y_test = grid_search_results['Y_test']
        Y_pred = grid_search_results['Y_pred']
        model = grid_search_results['model']
        sigma = grid_search_results['sigma']
        print("Loaded grid search results from file.")
    except FileNotFoundError:
        print("Running grid search...")
        best_params, best_log_lik, Y_test, Y_pred, model, sigma = grid_search(seven_harmonic_data, param_grid, t_plus_1, save_file, False)
        print("Grid search completed and results saved.")
    

        print("Grid search completed and results saved.")
    # ! #################################### SAWTOOTH Forecast BASELINE  

    param_grid = {
    'nodes': [100, 500, 1000],  
    'lr': [.2, 0.5, 0.8],  
    #'lr': [0.5],
    #'sr': [.8] ,
    'sr': [.5 ,0.8, 1.0],  
    #'ridge': [1e-7],
    'ridge': [1e-8 ,1e-7, 1e-6]  
    }
    # Make new filename if running a new test
    save_file = 'results/f10har_nsin_amp5_ssF_tp1_ba.pkl'

    # Create directories if they don't exist
    save_dir = os.path.dirname(save_file)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # check if the saved file exists
    try:
        with open(save_file, 'rb') as f:
            grid_search_results = pickle.load(f)
        best_params = grid_search_results['best_params']
        best_log_lik = grid_search_results['best_log_lik']
        Y_test = grid_search_results['Y_test']
        Y_pred = grid_search_results['Y_pred']
        model = grid_search_results['model']
        sigma = grid_search_results['sigma']
        print("Loaded grid search results from file.")
    except FileNotFoundError:
        print("Running grid search...")
        best_params, best_log_lik, Y_test, Y_pred, model, sigma = grid_search(ten_harmonic_data, param_grid, t_plus_1, save_file, False)
        print("Grid search completed and results saved.")
    

"""
