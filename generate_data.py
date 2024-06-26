import numpy as np
import matplotlib.pyplot as plt
import random 
from functools import partial
import noise

def single_series(function, T, low, train_T, rate, warmup = .5, forecast = 5, amp_noise = 1, same_start = True):
    """
    Given an inputted function, generates a single series of data

    Inputs:
        - function: function used to generate data 
    
    Parameters: 
        - T :       Period of function
        - low:      where to start linspace 
        - train_T:  number of periods to train on
        - rate:     number of training samples per period 
        - warmup:   number of periods to use in warmup
        - forecast: number of periods to forecast
        - amp_noise:    magnitude of amplitude noise 
        - same_start:   False means linspace starts at random value in (low, low + T)

    Outputs: 
        - (train, test) Datasets

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

    # Create the warmup and test data
    X_warmup = X_train[:warmup_points]
    Y_test = X_train[warmup_points: warmup_points + forecast_points]

    dataset = ((X_train, Y_train), (X_warmup, Y_test))

    return dataset

def multi_series(function = np.sin, num_series = 5, T = 2*np.pi, low = 0, train_T = 10, rate = 10, warmup = .5, forecast = 3, amp_noise = 1, same_start = False):
    """
    Given an input function, generates multiple series of data.

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
    - (train, validate ,test)

        Where 
        - train = (X_train, Y_train)
        - validate = (Val_warmup, Y_validate)
        - test = (X_warmup, Y_test)
    """
    num_points = rate * train_T
    warmup_points = int(np.round(warmup * rate))
    num_forecast = forecast * rate 

    # * Make containers to store the data
    # Train
    X_train = np.zeros((num_series, num_points - 1, 1))
    Y_train = np.zeros((num_series, num_points - 1, 1))
    # Validation
    Val_warmup = np.zeros((num_series, warmup_points, 1))
    Y_validate = np.zeros((num_series, num_forecast, 1))
    # Test
    X_warmup = np.zeros((warmup_points, 1))
    Y_test = np.zeros((num_forecast, 1))

    # Generate Training, Validation data 
    for i in range (num_series):

        # Training data 
        dataset = single_series(function, T, low, train_T, rate, warmup, forecast, amp_noise, same_start)
        X_train[i, :, 0] = dataset[0][0][:, 0]
        Y_train[i, :, 0] = dataset[0][1][:, 0]

        # Valdiation Data 
        dataset = single_series(function, T, low, train_T, rate, warmup, forecast, amp_noise, same_start)
        Val_warmup[i, :, 0] = dataset[1][0][:, 0]
        Y_validate[i, :, 0] = dataset[1][1][:, 0]

    #  Generate the test data
    (_, _), (X_warmup, Y_test)  = single_series(function, T, low, train_T, rate, warmup, forecast, amp_noise, same_start)

    train = (X_train, Y_train)
    validate = (Val_warmup, Y_validate)
    test = (X_warmup, Y_test)

    # dataset = ((X_train, Y_train),(Val_warmup, Y_validate) ,(X_warmup, Y_test))
    return train, validate, test 

def generate_data(data_function = np.sin, noise_function = noise.sine_noise, config = None):
    """
    Given an input function, generates multiple series of data.
    """
    if config is None:
        config = {}
    
    # Unpack config dictionary or set default values
    num_series = config.get('num_series', 5)
    T = config.get('T', 2*np.pi)
    low = config.get('low', 0)
    train_T = config.get('train_T', 10)
    rate = config.get('rate', 100)
    warmup = config.get('warmup', 0.5)
    forecast = config.get('forecast', 3)
    amp_noise = config.get('amp_noise', .3)
    same_start = config.get('same_start', False)

    train, validate, test = multi_series(data_function, num_series, T, low, train_T, rate, warmup, forecast, amp_noise, same_start)

    # unpack 
    X_train, Y_train = train
    Val_warmup, Y_validate = validate
    X_warmup, Y_test = test

    # Generate Test Noise
    test_domain = T * (warmup +  forecast)
    test_t_vals = np.linspace(0, test_domain, Y_test.shape[0])  
    test_noise = noise_function(test_t_vals, amp_noise)
    Y_test += test_noise.reshape(-1,1)

    # Generate Training, Validation Noise
    train_domain = T * train_T
    t_values = np.linspace(0, train_domain, Y_train.shape[1])
    
    for i in range (num_series):
        # Train Noise
        train_noise = noise_function(t_values, amp_noise)
        Y_train[i,:,:] += train_noise.reshape(-1,1)

        # Validation Noise
        val_noise = noise_function(test_t_vals, amp_noise)
        Y_validate[i,:,:] += val_noise.reshape(-1,1)

    train = (X_train, Y_train)
    validate = (Val_warmup, Y_validate)
    test = (X_warmup, Y_test)

    # dataset = ((X_train, Y_train),(Val_warmup, Y_validate) ,(X_warmup, Y_test))
    return train, validate, test 


def multi_harmonic(t, num_harmonics = 5):
    """
    Generates a multi harmonic function

    Inputs: 
    - t:                Generally a linspace of values that we will be evaluating multi harmonic over 
    - num_harmonics:    The number of harmonics we want in the function

    Outputs: 
    - The multi harmonic evalutaed at every point in t. 
    """

    # Make evaluation container
    signal = np.zeros_like(t)

    for i in range(num_harmonics):
        # Randomly draw amplitude, phase, freq values
        amplitude = random.uniform(0.5, 2)
        phase = random.uniform(0, 2 * np.pi)
        freq = random.uniform(0.5, 3)

        # add them to multi harmonic signal
        signal += amplitude * np.sin(phase + freq *  t)

    return signal


def plot_train_data(X_train, Y_train, single_series = False, three_series = False, every_series = False):
    """
    Plots the training data of the ESN

    Inputs: 
    - t:                Generally a linspace of values that we will be evaluating multi harmonic over 
    - num_harmonics:    The number of harmonics we want in the function

    Parameters: 
    - single_series:    If true Plot X train and Y train of a single series
    - three_series:     If true Plot three sets of X_train
    - every_series:     If true Plot all X_train

    Outputs: 
    - None
    """
    # Plot X train and Y train of a single series
    if single_series:
        # ! This one does not have the desired offset, fix at some point. 
        plt.figure()  
        plt.title("X_train vs Y_train for single series")
        plt.xlabel("$t$")
        plt.ylabel("amplitude")
        plt.plot(X_train[0,:,0], label="X_train", color="blue")
        plt.plot(Y_train[0,:,0], label="Y_train", color="red")
        plt.show()  

    # Plot three sets of X_train
    if three_series and X_train.shape[0]: 
        plt.figure()  
        plt.title("3 sets in X_train")
        plt.xlabel("$t$")
        plt.ylabel("amplitude")
        for i in range (3):
            plt.plot(X_train[i,:,0])
        plt.show()  

    # Plot all X_train
    if every_series: 
        plt.figure()  
        plt.title("all series in X_train")
        plt.xlabel("$t$")
        plt.ylabel("amplitude")
        for i in range (X_train.shape[0]):
            plt.plot(X_train[i,:,0])
        plt.show()  
        

def plot_prediction(Warmup, Y_test, Y_pred, sigma):
    
    """
    Plots the predictions against the ground truth for a single time series.

    Parameters:
        Warmup: Array containing the warmup data.
        Y_test: Array containing test
        Y_pred: Array containing the predicted values for the test data.
        sigma: Standard deviation

    Returns:
        None: The function only plots the data, no return value.
    """

    warmup_len = Warmup.shape[0]

    warmup_index = range(warmup_len)
    index_Pred = range(warmup_len, warmup_len + Y_pred.shape[0])
    index_test = range(warmup_len, warmup_len + Y_test.shape[0])

    plt.figure()  
    plt.title("Y_test vs Y_pred for single series")
    plt.xlabel("$t$")
    plt.ylabel("amplitude")
    plt.plot(warmup_index, Warmup, label="Warmup", color="black")
    plt.plot(index_test, Y_test, label="Y_test", color="red")
    plt.plot(index_Pred, Y_pred, label="Y_pred", color="blue")
    

    Y_pred = Y_pred.flatten()
    sigma = sigma.flatten()
    # Shaded standard deviation regions
    plt.fill_between(index_Pred, Y_pred - 3 * sigma , Y_pred + 3 * sigma, color='blue', alpha=0.2, label="3sigma")
    
    plt.legend()
    plt.show()  
