import numpy as np

def g(t, C):
    """
        Generates time-dependent Gaussian noise, scaled by some constant
        C.

        Params:
            t - array-like input-values for time
            C - float, scaling constant

        Return:
            array-like: Gaussian noise scaled by constant, following
                        the formula: g(t, C) = C(1/t + 0.1) * N(0, 1)
    """
    noise = (1/t + 0.1) * np.random.normal(0, 1, len(t))
    return C * noise


def f(t, A, v, C):
    """
        Generates the sine curve with added Gaussian noise

        Params:
            t - time values
            A - amplitude of sine curve
            v - frequency of sine curve
            C - scaling constant for the noise

        Return:
            Sine curve with time-dependent Gaussian noise scaled by a constant
    """
    sine_curve = A * np.sin(v * t)
    noise = g(t, C)
    return (sine_curve + noise)

def constant_noise(t, C = 1):
    """
        Generates constant Gaussian noise scaled by a constant
    """
    noise = np.random.normal(0, 1, len(t))
    return noise * C


def sine_noise(t, C = 1, x_factor=1.0):
    """
        Generates sine wave Gaussian noise scaled by a constant.

    Parameters:
        t : time values.
        C : scaling constant.
        x_factor : scaling factor for x in sine wave.

    Returns:
        Sine wave Gaussian noise scaled by the constant C.
    """
    noise = C * np.sin(x_factor * t) * np.random.normal(0, 1, len(t))
    return noise

def sigmoid_noise(t, C, slope=1.0, midpoint=0.0):
    """
        Generates sigmoidal Gaussian noise scaled by a constant.

    Parameters:
        t : time values.
        C : scaling constant.
        slope : slope of the sigmoid function.
        midpoint : midpoint of the sigmoid function.

    Returns:
        Sigmoidal Gaussian noise scaled by the constant C.
    """
    noise = C * (1 / (1 + np.exp(-slope * (t - midpoint)))) * np.random.normal(0, 1, len(t))
    return noise

def custom_noise(t, C):
    """
        Generates custom Gaussian noise scaled by a constant. This would follow
        the formula:
            noise = C * (0.5 * sin(t) * N(0,1) + e^{-0.1t} * N(0,1) + N(0,1))
        
    Parameters:
        t : time values.
        C : scaling constant.

    Returns:
        array-like: Custom Gaussian noise scaled by the constant C.
    """
    # random definitions of custom noise components
    sine_component = 0.5 * np.sin(t) * np.random.normal(0, 1, len(t))
    decay_component = np.exp(-0.1 * t) * np.random.normal(0, 1, len(t))
    random_component = np.random.normal(0, 1, len(t))
    
    noise = C * (sine_component + decay_component + random_component)
    
    return noise
