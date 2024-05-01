import numpy as np
import matplotlib.pyplot as plt
import random 

def single_series(function, T, low, train_T, rate, warmup = .5, forecast = 5, amp_noise = 1, same_start = True):
    """
    Inputs:
        - function: function used to generate data 
        - T :       Period of function
        - low:      where to start linspace 
        - train_T:  number of periods to train on
        - rate:     number of training samples per period 
        - warmup:   number of periods to use in warmup
        - forecast: number of periods to forecast
        - amp_noise:    magnitude of amplitude noise 
        - same_start:   False means linspace starts at random value in (low, low + T)

    Outputs: 
        - (train, test)

        Where 
            - train = (X_train, Y_train)
            - test = (X_warmup, Y_test)
    """

    # Set high to be train_T * T higher than low
    high = low + T * train_T
    num_points = rate * train_T
    warmup_points = int(np.round(warmup * rate))
    forecast_points = int(np.round(forecast * rate))

    if not same_start:
        low = low + random.uniform(0, T)

    # Sample points from the function and evaluate
    x_values = np.linspace(low, high, num_points)
    X = function(x_values).reshape(-1, 1) 

    # Create the training data
    X_train = X[:num_points - 1]
    Y_train = X[1:]

    # TODO Add noise to Y

    # Create the warmup and test data
    X_warmup = X_train[:warmup_points]
    Y_test = X_train[warmup_points: warmup_points + forecast_points]

    dataset = ((X_train, Y_train), (X_warmup, Y_test))

    return dataset

def multi_series(function, num_series, T, low, train_T, rate, warmup = .5, forecast = 5, amp_noise = 1, same_start = False):
    """
    Inputs:
        - function: function used to generate data 
        - num_sets: number of training sets
        - T :       Period of function
        - low:      where to start linspace 
        - train_T:  number of periods to train on
        - rate:     number of training samples per period 
        - warmup:   number of periods to use in warmup
        - forecast: number of periods to forecast
        - amp_noise:    magnitude of amplitude noise 
        - same_start:   False means linspace starts at random value in (low, low + T)

    Outputs: 
        - (train, test)

        Where 
            - train = (X_train, Y_train)
            - test = (X_warmup, Y_test)
    """
    num_points = rate * train_T
    warmup_points = int(np.round(warmup * rate))
    num_forecast = forecast * rate 

    # * Make containers, 1 dim for now
    X_train = np.zeros((num_series, num_points - 1, 1))
    Y_train = np.zeros((num_series, num_points - 1, 1))
    X_warmup = np.zeros((warmup_points, 1))
    Y_test = np.zeros((num_forecast, 1))

    print(X_warmup.shape)
    print(Y_test.shape)
    for i in range (num_series):
        dataset = single_series(function, T, low, train_T, rate, warmup, forecast, amp_noise, same_start)
        X_train[i, :, 0] = dataset[0][0][:, 0]
        Y_train[i, :, 0] = dataset[0][1][:, 0]

        # * Generate the warmup and test data
        if i == 0:
            X_warmup = dataset[1][0]
            Y_test = dataset[1][1]
        
    dataset = ((X_train, Y_train), (X_warmup, Y_test))

    return dataset

