import numpy as np
import matplotlib.pyplot as plt
import pickle

from reservoirpy.nodes import Reservoir, Ridge, ESN
from functools import partial
from sklearn.model_selection import ParameterGrid
from scipy import signal

from generate_data import multi_series, multi_harmonic

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
    (X_train, Y_train), validate, test = dataset

    # Make a reservoir and readout, and link them together to make esn
    reservoir = Reservoir(nodes, lr = lr, sr = sr) 
    readout = Ridge(ridge = ridge)
    

    # ! This paralellizes the code, remove if you want to run model outsite of the if statement in train_model
    #model = reservoir >> readout
    model = ESN(reservoir=reservoir, readout=readout, workers=-1)

    # Train the model
    model.fit(X_train, Y_train)

    return model, validate, test

def base_sigma_t_plus_1(model, validate):

    Val_warmup, Y_validate = validate

    num_series = Y_validate.shape[0]
    N = Y_validate.size 
    sigma = 0

    for series in range (num_series):
        # Reset model, run it on warmup values
        model.run(Val_warmup[series,:,:], reset=True)
        Val_pred = model.run(Y_validate[series, :,:])

        sigma += np.sum((Val_pred - Y_validate[series, :,:])**2)

    sigma /= N
    sigma_vec = np.ones((Y_validate.shape[1] - 1 ,1)) * sigma

    return sigma_vec

def upgrade_sigma_t_plus_1(model, validate):

    Val_warmup, Y_validate = validate

    num_series = Y_validate.shape[0]
    sigma = np.zeros((Y_validate.shape[1],1))

    for series in range (num_series):
        # Reset model, run it on warmup values
        model.run(Val_warmup[series,:,:], reset=True)
        Val_pred = model.run(Y_validate[series, :,:])

        sigma += (Val_pred - Y_validate[series, :,:])**2

    sigma /= num_series

    # Skipping the first value of sigma to align shapes
    return sigma[1:]


def base_sigma_forecast(model, validate):

    Val_warmup, Y_validate = validate

    N = Y_validate.size 
    num_series = Y_validate.shape[0]
    num_forecast = Y_validate.shape[1]
    sigma = 0 

    for series in range (num_series):
        # Reset model, run it on warmup values
        Val_pred = np.zeros((num_forecast,1))
        warmup_y = model.run(Val_warmup[series,:,:], reset=True)
        x = warmup_y[-1].reshape(1, -1)

        for i in range(num_forecast):
                x = model(x)
                Val_pred[i] = x
    
        sigma += np.sum((Val_pred - Y_validate[series, :,:])**2)
    
    sigma /= N
    sigma_vec = np.ones((Y_validate.shape[1] ,1)) * sigma

    return sigma_vec    


def upgrade_sigma_forecast(model, validate):

    Val_warmup, Y_validate = validate


    return 0




def t_plus_1(model, validate, test):

    sigma = upgrade_sigma_t_plus_1(model, validate)

    X_warmup, Y_test = test

    test_points = Y_test.shape[0]
    X_test = Y_test[:test_points - 1]
    Y_test = Y_test[1 : test_points]

    # Reset model, run it on warmup values
    model.run(X_warmup, reset=True)

    Y_pred = model.run(X_test)

    loss = log_likelihood(Y_pred=Y_pred, sigma=np.full_like(Y_pred, 1), Y_test=Y_test)

    return model, Y_test, Y_pred, loss, sigma


def forecast(model, validate, test):

    sigma = base_sigma_forecast(model,validate)

    X_warmup, Y_test = test

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

    return model, Y_test, Y_pred, loss, sigma


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
            model_iter, validate, test = curve_fit(dataset, nodes=params['nodes'], lr=params['lr'], sr=params['sr'], ridge=params['ridge'])
            _, _, Y_pred_iter, loss_iter, sigma_iter = prediction_task(model_iter, validate, test)

            losses.append(loss_iter) 

        
        # Take the median loss value, to handle random initialization
        loss = np.median(losses)

        # If the current loss is lower than the best loss, update the best loss and best parameters
        if loss < best_loss:
            best_loss = loss
            best_params = params
            model, Y_pred, sigma = model_iter, Y_pred_iter, sigma_iter
    
    results = {
        'best_params': best_params,
        'best_loss': best_loss,
        'Y_test': Y_test,
        'Y_pred': Y_pred,
        'model': model,
        'sigma': sigma
    }

    with open(save_file, 'wb') as f:
        pickle.dump(results, f)
    
    Y_test = test[1]

    return best_params, best_loss, Y_test, Y_pred, model, sigma


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
train, validate, test = dataset

param_grid = {
    'nodes': [100],  
    #'lr': [0.1, 0.5, 0.7, 1.0],  
    'lr': [0.1],
    'sr': [.5] ,
    #'sr': [.5 ,0.8, 1.0],  
    'ridge': [1e-9],
    #'ridge': [1e-9, 1e-8, 1e-7]  
}
#best_params, best_loss, Y_test, Y_pred, model = grid_search(dataset, param_grid, t_plus_1)
#plot_prediction(X_warmup, Y_test, Y_pred, sigma = 1)
