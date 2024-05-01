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