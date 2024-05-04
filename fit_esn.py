import numpy as np
import matplotlib.pyplot as plt
import random 
from reservoirpy.nodes import Reservoir, Ridge, ESN
from functools import partial
from sklearn.model_selection import ParameterGrid

from  generate_data import generate_data, multi_harmonic, plot_train_data
import noise 
import grid_search


data_config = {
    'num_series': 10,                  # Number of series to generate (default: 5)
    'T': 2*np.pi,                     # Period of the function (default: 2*np.pi)
    'low': 0,                         # Lower bound for the x-axis (default: 0)
    'train_T': 10,                    # Length of training data (default: 10)
    'rate': 100,                      # Rate of sampling (default: 100)
    'warmup': 0.5,                    # Percentage of data to use as warmup (default: 0.5)
    'forecast': 3,                    # Number of points to forecast (default: 3)
    'amp_noise': .75,                 # Amplitude of added noise (default: 0.3)
    'same_start': False               # Whether all series should start at the same point (default: False)
}


data = generate_data(multi_harmonic, noise.sine_noise, data_config)

(X_train, Y_train), (X_warmup, Y_test) = data

plot_train_data(X_train, Y_train, single_series = True)


# ? Old code, ignore
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
    # ?: Potentially change this to median loss value across dataset, could give better results. 
    loss = np.sum(np.square(Y_test - Y_pred))

    return model, Y_test, Y_pred, loss

