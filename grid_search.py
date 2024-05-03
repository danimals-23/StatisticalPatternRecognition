import numpy as np
import matplotlib.pyplot as plt
import random 
from reservoirpy.nodes import Reservoir, Ridge, ESN
from functools import partial
from sklearn.model_selection import ParameterGrid

from generate_data import multi_series, plot_prediction, multi_harmonic


# TODO Implement likleihood.
def wave_forecast(dataset, nodes = 100, lr = .5, sr = .9, ridge = 1e-8):
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

    # Make container for predictions
    num_forecast = Y_test.shape[0]
    Y_pred = np.empty((num_forecast,Y_test.shape[1]))

    # Make a reservoir and readout, and link them together to make esn
    reservoir = Reservoir(nodes, lr = lr, sr = sr) 
    readout = Ridge(ridge = ridge)
    model = reservoir >> readout

    # Train the model
    model.fit(X_train, Y_train)

    # Reset model, run it on warmup values
    warmup_y = model.run(X_warmup, reset=True)
    x = warmup_y[-1].reshape(1, -1)

    for i in range(num_forecast):
        x = model(x)
        Y_pred[i] = x

    #TODO Change this to likelihood (once we have embedded variance into system)
    # TODO: Potentially change this to median loss value across the dataset, could give us better results. 
    loss = np.sum(np.square(Y_test - Y_pred))

    return model, Y_test, Y_pred, loss


def grid_search(dataset, param_grid):
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
        for _ in range(3):  # Perform wave_fit three times for robustness
            result = wave_fit(dataset, nodes=params['nodes'], lr=params['lr'], sr=params['sr'], ridge=params['ridge'])
            losses.append(result[3]) 
        
        # Take the median loss value, to handle random initialization
        loss = np.median(losses)

        # If the current loss is lower than the best loss, update the best loss and best parameters
        if result[3] < best_loss:
            best_loss = result[3]
            best_params = params
            model, Y_test, Y_pred= result[:3]
    return best_params, best_loss, Y_test, Y_pred, model


f = partial(multi_harmonic, num_harmonics = 5)

dataset =  multi_series(function = f, num_series = 1, train_T = 30, warmup = 1, rate = 300, same_start = False)
(X_train, Y_train), (X_warmup, Y_test) = dataset

print(X_train.shape, Y_train.shape, X_warmup.shape, Y_test.shape)

#model, Y_test, Y_pred, Y_train, loss = wave_forecast(dataset, nodes = 250)


param_grid = {
    'nodes': [1500],  
    #'lr': [0.1, 0.3, 0.7, 1.0],  
    'lr': [0.5, 0.7, 1.0],
    # 'sr': [0.1, 0.3, 0.7, 1.0],  
    'sr': [.5, 0.7, 1.0],
    # 'ridge': [1e-9, 1e-8, 1e-7]  
    'ridge': [1e-9, 1e-7]  
}

best_params, best_loss, Y_test, Y_pred, Y_train, model = grid_search(dataset, param_grid)

plot_prediction(X_warmup, Y_test, Y_pred, sigma = 1)