import numpy as np
import matplotlib.pyplot as plt
import pickle

from reservoirpy.nodes import Reservoir, Ridge, ESN
from functools import partial
from sklearn.model_selection import ParameterGrid
from scipy import signal
from concurrent.futures import ProcessPoolExecutor

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
    """
    Produces a constant sigma vector for the t_plus_1 prediction 

    Inputs 
    - Model: Trained ESN
    - Validate: Validation training set (Val_warmup, Y_validate)

    Outputs:
    - Constant Sigma Vector
    """

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
    sigma = np.sqrt(sigma)

    sigma_vec = np.ones((Y_validate.shape[1] - 1 ,1)) * sigma

    return sigma_vec

def upgrade_sigma_t_plus_1(model, validate):
    """
    Produces a time dependent sigma vector for the t_plus_1 prediction 

    Inputs 
    - Model: Trained ESN
    - Validate: Validation training set (Val_warmup, Y_validate)

    Outputs:
    - Time dependent Sigma Vector
    """

    Val_warmup, Y_validate = validate

    num_series = Y_validate.shape[0]
    sigma = np.zeros((Y_validate.shape[1],1))

    for series in range (num_series):
        # Reset model, run it on warmup values
        model.run(Val_warmup[series,:,:], reset=True)
        Val_pred = model.run(Y_validate[series, :,:])

        sigma += (Val_pred - Y_validate[series, :,:])**2

    sigma /= num_series
    sigma = np.sqrt(sigma)

    # Skipping the first value of sigma to align shapes
    return sigma[1:]


def base_sigma_forecast(model, validate):
    """
    Produces a constant sigma vector for the forecast prediction 

    Inputs 
    - Model: Trained ESN
    - Validate: Validation training set (Val_warmup, Y_validate)

    Outputs:
    - Constant Sigma Vector
    """
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
    sigma = np.sqrt(sigma)

    sigma_vec = np.ones((Y_validate.shape[1] ,1)) * sigma

    return sigma_vec    


def upgrade_sigma_forecast(model, validate):
    """
    Produces a time dependent sigma vector for the forecast prediction 

    Inputs 
    - Model: Trained ESN
    - Validate: Validation training set (Val_warmup, Y_validate)

    Outputs:
    - Time dependent Sigma Vector
    """

    Val_warmup, Y_validate = validate

    num_series = Y_validate.shape[0]
    num_forecast = Y_validate.shape[1]
    sigma = np.zeros((Y_validate.shape[1],1))

    for series in range (num_series):
        # Reset model, run it on warmup values
        Val_pred = np.zeros((num_forecast,1))
        warmup_y = model.run(Val_warmup[series,:,:], reset=True)
        x = warmup_y[-1].reshape(1, -1)

        for i in range(num_forecast):
                x = model(x)
                Val_pred[i] = x

        sigma += (Val_pred - Y_validate[series, :, :])**2
    
    sigma /= num_series
    sigma = np.sqrt(sigma)

    return sigma


def t_plus_1(model, validate, test, upgrade = False):
    """
    Given a trained model, completes prediction task of given f(t) predict f(t+1) for a series of values

    Inputs: 
    - Model: Trained ESN
    - Validate: Validation training set (Val_warmup, Y_validate)
    - Test: Testing data (X_warmup, Y_test)
    - Upgrade: Boolean determining if we should use upgraded (time dependent) sigma or not

    Outputs:
    - Time dependent Sigma Vector
    """
    if upgrade:
        sigma = upgrade_sigma_t_plus_1(model, validate)
    else:
        sigma = base_sigma_t_plus_1(model, validate)

    X_warmup, Y_test = test

    test_points = Y_test.shape[0]
    X_test = Y_test[:test_points - 1]
    Y_test = Y_test[1 : test_points]

    # Reset model, run it on warmup values
    model.run(X_warmup, reset=True)

    Y_pred = model.run(X_test)

    log_lik = log_likelihood(Y_pred=Y_pred, sigma=np.full_like(Y_pred, 1), Y_test=Y_test)
    # loss = np.mean((Y_test - Y_pred) ** 2)

    return model, Y_test, Y_pred, log_lik, sigma


def forecast(model, validate, test, upgrade = False):
    """
    Given a trained model, completes prediction task of forcasting time series data

    Inputs: 
    - Model: Trained ESN
    - Validate: Validation training set (Val_warmup, Y_validate)
    - Test: Testing data (X_warmup, Y_test)
    - Upgrade: Boolean determining if we should use upgraded (time dependent) sigma or not

    Outputs:
    - Time dependent Sigma Vector
    """
    if upgrade:
        sigma = upgrade_sigma_forecast(model, validate)
    else:
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

    
    log_lik = log_likelihood(Y_pred=Y_pred, sigma=np.full_like(Y_pred, 1), Y_test=Y_test)
    #loss = np.sum(np.square(Y_test - Y_pred))

    return model, Y_test, Y_pred, log_lik, sigma


def grid_search(dataset, param_grid, prediction_task, save_file, upgrade = False):
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
    - results: best_params, best_log_lik, Y_test, Y_pred, Y_train, model
    """

    # Create a grid of parameters to try
    grid = ParameterGrid(param_grid)

    # Initialize the best log_lik to a very large number and best_params to None
    # ? best_loss = float('inf')
    best_log_lik = float('-inf')
    best_params = None

    # Loop over the grid
    for params in grid:

        log_likes = []
        for _ in range(3):

            # Perform wave_fit three times for robustness
            model_iter, validate, test = curve_fit(dataset, nodes=params['nodes'], lr=params['lr'], sr=params['sr'], ridge=params['ridge'])
            _, _, Y_pred_iter, log_lik_iter, sigma_iter = prediction_task(model_iter, validate, test, upgrade)

            log_likes.append(log_lik_iter) 

        
        # Take the median log_lik value, to handle random initialization
        log_lik = np.median(log_likes)

        # If the current log_lik is lower than the best log_lik, update the best log_lik and best parameters
        #if loss < best_loss:
        if log_lik > best_log_lik:
            best_log_lik = log_lik
            best_params = params
            model, Y_pred, sigma = model_iter, Y_pred_iter, sigma_iter
    
    Y_test = test[1]
    
    results = {
        'best_params': best_params,
        'best_log_lik': best_log_lik,
        'Y_test': Y_test,
        'Y_pred': Y_pred,
        'model': model,
        'sigma': sigma
    }

    with open(save_file, 'wb') as f:
        pickle.dump(results, f)
    

    return best_params, best_log_lik, Y_test, Y_pred, model, sigma


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
    log_likelihood = np.sum(log_likelihoods) / Y_test.size[0]
    return log_likelihood
