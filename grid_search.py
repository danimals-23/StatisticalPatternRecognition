import numpy as np
import matplotlib.pyplot as plt
import random 
from reservoirpy.nodes import Reservoir, Ridge, ESN

# TODO Test more serieously on data with multiple series.
# TODO Implement likleihood.
def wave_fit(dataset, nodes = 100, lr = .5, sr = .9, ridge = 1e-7):
    """
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
    loss = np.sum(np.square(Y_test - Y_pred))

    return model, Y_test, Y_pred, Y_train, loss

# TODO finish testing. 
def grid_search(dataset, param_grid):
    """
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
        # Call wave_fit with the current parameters
        result = wave_fit(dataset, nodes=params['nodes'], lr=params['lr'], sr=params['sr'], ridge=params['ridge'])

        # If the current loss is lower than the best loss, update the best loss and best parameters
        if result[4] < best_loss:
            best_loss = result[4]
            best_params = params
            model, Y_test, Y_pred, Y_train = result[:4]
    return best_params, best_loss, Y_test, Y_pred, Y_train, model

