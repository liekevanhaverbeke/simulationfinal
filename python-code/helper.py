"""
File created by Maxim Frankish. Adapted by Samuel Bakker.
Contains distributions used in simulation.py.

Note: ofcourse you can also use the scipy.stats package to instantiate a distribution on which you then call .rvs().
But why would you, if this is more fun? ;)
"""

USE_ANTITHETIC = False  

import random
import math

def Exponential_distribution(lambda_value) -> float:
    """Exponential distribution

    Args:
        lambda_value (float): shape parameter
    Returns:
        float: random sample
    """
    j1 = random.randint(0, 1000) / 1000
    if USE_ANTITHETIC:
        j1 = 1 - j1
    if j1 == 0:
        j1 = 0.0001
    j2 = -math.log(j1) / lambda_value
    return j2

def Normal_distribution(mean, stdev) -> float:
    """Normal distribution.

    Args:
        mean (float): mean
        stdev (float): stddev

    Returns:
        float: value in minutes
    """
    do_loop = True
    while do_loop:
        v1 = random.random() * 2 - 1
        v2 = random.random() * 2 - 1
        t = v1 * v1 + v2 * v2
        if ((t >= 1) or (t == 0)):
            do_loop = True
        else:
            do_loop = False
    multiplier = math.sqrt(-2 * math.log(t) / t)
    x = v1 * multiplier * stdev + mean
    return x

def Bernouilli_distribution(prob) -> bool:
    """Bernouilli distribution

    Args:
        prob (double): probability of returning a 1

    Returns:
        bool: true if random number is smaller than prob, false otherwise
    """
    j1 = random.random()
    return j1 < prob
