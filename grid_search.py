import numpy as np
import matplotlib.pyplot as plt

from reservoirpy.nodes import Reservoir, Ridge, ESN
from functools import partial
from sklearn.model_selection import ParameterGrid
from scipy import signal

from generate_data import multi_series, plot_prediction, multi_harmonic

def curve_fit(dataset, nodes = 100, lr = .5, sr = .9, ridge = 1e-8):
    """
    Fits an echo state network to a training dataset

    Inputs:
    - Dataset:        
         - (train, test) Datasets
        Where 
            - train = (X_train, Y_train)
            - test = (X_warmup, Y_test)

    Parameters: 
    - nodes: # Nodes in reservoir
    - lr: The learning rate of the reservoir
    - sr: The spectral radius of the reservoir
    - ridge: The ridge of the readout

    Returns:
    - Y_test: The test data
    - Y_pred: The predicted data
    - Y_train: The training data
    """
    # Prep data
    (X_train, Y_train), (X_warmup, Y_test) = dataset

    # Make a reservoir and readout, and link them together to make esn
    reservoir = Reservoir(nodes, lr = lr, sr = sr) 
    readout = Ridge(ridge = ridge)
    model = reservoir >> readout

    # Train the model
    model.fit(X_train, Y_train)

    return model, X_warmup, Y_test

def t_plus_1(model, X_warmup, Y_test):

    test_points = Y_test.shape[0]
    X_test = Y_test[:test_points - 1]
    Y_test = Y_test[1 : test_points]

    # Reset model, run it on warmup values
    model.run(X_warmup, reset=True)

    Y_pred = model.run(X_test)

    # TODO change to log likelihood
    # loss = log_likelihood(Y_pred=Y_pred, sigma=np.full_like(Y_pred, 1), Y_test=Y_test)
    loss = np.sum(np.square(Y_test - Y_pred))

    return model, Y_test, Y_pred, loss


def forecast(model, X_warmup, Y_test):

    num_forecast = Y_test.shape[0]
    Y_pred = np.empty((num_forecast,Y_test.shape[1]))

    # Reset model, run it on warmup values
    warmup_y = model.run(X_warmup, reset=True)
    x = warmup_y[-1].reshape(1, -1)

    for i in range(num_forecast):
        x = model(x)
        Y_pred[i] = x

    # TODO change to log likelihood
    # loss = log_likelihood(Y_pred=Y_pred, sigma=np.full_like(Y_pred, 1), Y_test=Y_test)
    loss = np.sum(np.square(Y_test - Y_pred))

    return model, Y_test, Y_pred, loss


def grid_search(dataset, param_grid, prediction_task):
    """
    Preforms a grid search over lr, sr, ridge parameters, returns optimal values based on log likelihood.

    Inputs:
    - Dataset:        
        - (train, test) Datasets

        Where 
            - train = (X_train, Y_train)
            - test = (X_warmup, Y_test)

    Parameters: 
    - param_grid: A dictionary containing range of param values to try,
                Keys:
                -'lr'       Learning Rate
                -'sr'       Spectral Radius
                -'ridge'    Ridge
                -'nodes'    # Nodes

    Returns:
    - results: best_params, best_loss, Y_test, Y_pred, Y_train, model
    """

    # Create a grid of parameters to try
    grid = ParameterGrid(param_grid)

    # Initialize the best loss to a very large number and best_params to None
    best_loss = float('inf')
    best_params = None

    # Loop over the grid
    for params in grid:

        losses = []
        for _ in range(3):

            # Perform wave_fit three times for robustness
            model_iter, X_warmup, Y_test = curve_fit(dataset, nodes=params['nodes'], lr=params['lr'], sr=params['sr'], ridge=params['ridge'])
            _, _, Y_pred_iter, loss_iter = prediction_task(model_iter, X_warmup, Y_test)

            losses.append(loss_iter) 

        
        # Take the median loss value, to handle random initialization
        loss = np.median(losses)

        # If the current loss is lower than the best loss, update the best loss and best parameters
        if loss < best_loss: # was results [3]
            best_loss = loss#[3]
            best_params = params
            model, Y_pred = model_iter, Y_pred_iter
    return best_params, best_loss, Y_test, Y_pred, model

def log_likelihood(Y_pred, sigma, Y_test):
    """
        Computes the log-likelihood of the test data given predicted values and
        standard deviations

        Parameters:
        - Y_pred : Prediction values
        - sigma : Standard deviations
        - Y_test : Test values

        Returns:
        - Log-likelihood of test data

    """
    log_likelihoods = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((Y_test - Y_pred) / sigma)**2
    log_likelihood = np.sum(log_likelihoods)
    return log_likelihood


# TESTING CODE
f = partial(multi_harmonic, num_harmonics = 1)

dataset =  multi_series(function = signal.sawtooth, num_series = 1, train_T = 30, warmup = 1, rate = 300, same_start = False)
(X_train, Y_train), (X_warmup, Y_test) = dataset

param_grid = {
    'nodes': [100],  
    #'lr': [0.1, 0.5, 0.7, 1.0],  
    'lr': [0.1],
    'sr': [.5] ,
    #'sr': [.5 ,0.8, 1.0],  
    'ridge': [1e-9],
    #'ridge': [1e-9, 1e-8, 1e-7]  
}
best_params, best_loss, Y_test, Y_pred, model = grid_search(dataset, param_grid, t_plus_1)
plot_prediction(X_warmup, Y_test, Y_pred, sigma = 1)
