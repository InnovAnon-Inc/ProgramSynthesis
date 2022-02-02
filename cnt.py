#! /usr/bin/env python3

from functools import partial
import numpy as np

#
# TODO Turing-complete Neural Networks, including Cellular Automata Neural Network
#
class NCell(Cell): # Neural Cell
    pass
class MANN(CAType): # Memory-Augmented Neural Network
    #def __init__(self): pass
    def initialize(self, ndx, val=None):
        # TODO
        pass
    def update(self, cell, neighbors, inverse=False):
        if inverse: raise Exception() # not supported
        # TODO
        pass
def init_mann(gene):
    a = get_shape(gene)
    b = MANN()
    c = CAShape(a, b.initialize, b.update)
    return NCA(c.initial, c.step) # TODO read/write heads + memory space ?
def init_nn(gene):
    # TODO NN types:
    # - FFNN: Feed-Forward
    # -  CNN: Convolutional
    # -  RNN: Recurrent; LSTM; MANN
    # -  GNN: Graph
    # - CANN: Cellular Automata
    return init_mann(gene)

def main():
    a = random_bitstring() # TODO
    d = init_nn(a)

    for e in d.converge(): print(e)
    return 0

if __name__ == '__main__': exit(main())

