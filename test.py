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
    'num_series': 10,                  # Number of series to generate (default: 5)
    'T': 2*np.pi,                     # Period of the function (default: 2*np.pi)
    'low': 0,                         # Lower bound for the x-axis (default: 0)
    'train_T': 20,                    # Length of training data (default: 10)
    'rate': 300,                      # Rate of sampling (default: 100)
    'warmup': 1,                    # Percentage of data to use as warmup (default: 0.5)
    'forecast': 3,                    # Number of points to forecast (default: 3)
    'amp_noise': .1,                 # Amplitude of added noise (default: 0.3)
    'same_start': True               # Whether all series should start at the same point (default: False)
}

param_grid = {
    'nodes': [100],  
    #'lr': [0.5, 0.7, 1.0],  
    'lr': [0.5],
    #'sr': [.8] ,
    'sr': [.5 ,0.8, 1.0],  
    'ridge': [1e-7],
    #'ridge': [1e-9, 1e-8 ,1e-7]  
}
if __name__ == '__main__':

    data = generate_data(np.sin, noise.sine_noise, data_config)

    # make new filename if running a new test
    # save_file = 'grid_search_results.pkl'

    # Make new filename if running a new test
    save_file = 'results/grid_search_results.pkl'

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
        best_params, best_log_lik, Y_test, Y_pred, model, sigma = grid_search(data, param_grid, forecast, save_file, True)
        print("Grid search completed and results saved.")


    ((X_train, Y_train),(Val_warmup, Y_validate),(X_warmup, Y_test)) = data
    plot_prediction(X_warmup, Y_test, Y_pred, sigma)
    print(best_log_lik)
