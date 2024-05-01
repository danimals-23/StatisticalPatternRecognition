import numpy as np
import matplotlib.pyplot as plt
import random 

def single_series(function, T, low, train_T, rate, warmup = .5, forecast = 5, amp_noise = 1, same_start = True):
    """
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

    # TODO Add noise to Y

    # Create the warmup and test data
    X_warmup = X_train[:warmup_points]
    Y_test = X_train[warmup_points: warmup_points + forecast_points]

    dataset = ((X_train, Y_train), (X_warmup, Y_test))

    return dataset

def multi_series(function = np.sin, num_series = 5, T = 2*np.pi, low = 0, train_T = 10, rate = 10, warmup = .5, forecast = 5, amp_noise = 1, same_start = False):
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

def multi_harmonic(t, num_harmonics = 5):
    """
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
        

(X_train, Y_train), (X_warmup, Y_test) = multi_series()


# TODO: Plot Multi Harmonic 
# TODO: Figure out why the following function, there always to be convergence at the end.
plot_train_data(X_train, Y_train,every_series = True)

# TODO: Make a plot test vs prediction w/ warmup (where warmup is different color)
