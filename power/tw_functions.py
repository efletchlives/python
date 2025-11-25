import numpy as np

def percent_error(approx,actual):
    return np.abs(np.abs(approx) - np.abs(actual))/np.abs(actual)