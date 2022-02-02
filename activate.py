#! /usr/bin/env python3

#from functools import partial
#import numpy as np
#import scipy as sp

#from math import ceil

def identity(x): return x
def binary_step(x):
    if x < 0: return 0
    return 1
def sigmoid(x): return 1/(1+exp(-x))
# tanh
def relu(x): return max(0, x)
#def gelu(x): pass
def softplus(x): ln(1+exp(x))
def elu(a, x):
    if x <= 0: return a*(exp(x)-1)
    return x
#def selu(l, a, x): pass
def leaky_relu(x):
    if x < 0: return 0.01 * x
    return x
def prelu(a, x):
    if x < 0: return a * x
    return x
def silu(x): return x / (1 + exp(-x))
def gaussian(x): return exp(-(x**2))

