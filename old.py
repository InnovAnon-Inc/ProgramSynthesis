#! /usr/bin/env python3

# Speech-to-Text
# Parse(Speech-to-Text)
# => sentence type:
#    - statements => prolog
#    - queries
#    - imperatives
#    - ...

#from stat_parser import Parser
#parser = Parser()
#print(parser.parse("How can the net amount of entropy of the universe be massively decreased?"))

# input/padding -> CA -> no output ?
#                  ||       ||
#                  GA <- fitness?         cell has-a genome; genome => nn cell properties
#                  ||
#                  NN

from functools import partial
import numpy as np

def is_in_list(array_to_check, list_np_arrays): return np.any(np.all(array_to_check == list_np_arrays))
class CA(object): # universe (petri dish)
    def __init__(self, initial, step):
        self.state = initial()
        self.step  = step
    def forward(self):#, i):
        self.state = self.step(self.state)#, i)
        return self.state
        #initial = self.step(self.state)
        #return CA(initial, self.step)
    def backward(self):
        self.state = self.step(self.state, inverse=True)
        return self.state
        #return CA(initial, self.step)
    def converge(self, n=10):
        M, m = [], self.state
        t = 0
        while not is_in_list(m, M):
            yield m
            M.append(m)
            if len(M) > n: M = M[1:]
            m = self.forward()
            t = t + 1
        return t
    #def __eq__(self, o): return self.state.all() == o.state.all()

class CAShape(object): # topology; n-dimensional
    def __init__(self, shape, initialize, update):
        self.shape      = shape
        self.initialize = initialize
        self.update     = update
    def initial(self):
        S = self.shape
        I = np.ndindex(*S)
        A = tuple(map(self.initialize, I))
        B = np.array(A, dtype=object)
        return B.reshape(*S)
    def neighbors_helper(self, P): # von neumann
        for i, s in enumerate(self.shape):
            #p = (*P[:i], P[i], *P[i+1:])
            if P[i]-1 >= 0: yield (*P[:i], P[i]-1, *P[i+1:])
            if P[i]+1 <  s: yield (*P[:i], P[i]+1, *P[i+1:])
            # TODO (i-1) % S[i]
    def neighbors(self, cell):
        c = cell.item().pos
        return c, tuple(self.neighbors_helper(c))
    def update_helper(self, pn, state, inverse=False):
        pos, neighbors = pn
        f = lambda p: state[p]
        c = f(pos)
        N = tuple(map(f, neighbors))
        return self.update(c, N, inverse)
    def step(self, state, inverse=False):#padding, inverse=False):
        I = np.nditer(state, flags=["refs_ok",])
        N = tuple(map(self.neighbors, I))
        u = partial(self.update_helper, state=state, inverse=inverse)
        A = tuple(map(u, N))
        B = np.array(A, dtype=object)
        return B.reshape(*self.shape)



#
# dummy impl
#
class Cell(object):
    def __init__(self, pos, val):
        self.pos = pos
        self.val = val
    def __repr__(self): return "Cell(pos=%s)" % (self.pos,)
    def __eq__(self, o):
        if not isinstance(o, type(self)): return False
        if self.pos != o.pos: return False
        if self.val != o.val: return False
        return True
class CAType(object): # behavior
    #def __init__(self): pass
    def initialize(self, ndx, val=None):
        if val is None:
            if np.random.random() < 0.7: val = 0
            else:                        val = 1
        return Cell(ndx, val)
    def update(self, cell, neighbors, inverse=False):
        if inverse: raise Exception() # not supported
        f = lambda c: c.val
        V = tuple(map(f, neighbors))
        v = sum(V) + cell.val
        if 2 <= v and v <= 3: val = 1
        else:                 val = 0
        return Cell(cell.pos, val)
def init_ca(a):
    b = CAType()
    c = CAShape(a, b.initialize, b.update)
    return CA(c.initial, c.step)



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



#
# Genetic Cellular Automata; TODO fitness functions
#
def random_bitstring(n=10): return np.random.randint(2, size=(n,))
class GCell(Cell): # Genetic Cell
    def __init__(self, pos, val):#, data):
        Cell.__init__(self, pos, val, phenotype)
        #self.data = init_nn(self.val)
        self.data = phenotype(self.val)
    def update(self):
        self.converge()
        #return ???
        # TODO
class GCA(CAType): # Genetic Cellular Automata
    def __init__(self, abiogenesis, cull, spawn, phenotype):
        self.abiogenesis = abiogenesis
        self.cull        = cull
        self.spawn       = spawn
        self.phenotype   = phenotype
    def initialize(self, ndx):#, gene=None):
        #if gene is None:
        gene = self.abiogenesis()
        return GCell(ndx, gene, self.phenotype)
    def update(self, cell, neighbors, inverse=False):
        if inverse: raise Exception() # not supported
        #cell.update(inverse) # compute fitness
        if cell.val is None: # no life
            return self.spawn(cell, neighbors)
        if self.cull(cell, neighbors):
            #return self.initialize(cell.pos)
            #return GCell(cell.pos, None, None)
            return self.spawn(cell, neighbors)
        # no change
        return GCell(cell.pos, cell.val, self.phenotype)#, cell.data)
        #return cell # TODO defensive copy ?
import scipy as sp
class GCAProperties(object):
    #def __init__(self):
    def abiogenesis(self):
        if np.random.random() < 0.7: return None
        gene = random_bitstring()
        return self.initialize(gene)
    def cull(self, cell, neighbors):
        f = self.fitness(cell)
        F = map(self.fitness, neighbors)
        A = np.fromiter(F, dtype=float)
        p = sp.stats.percentileofscore(A, f, kind='rank')
        return p < .10
    def spawn(self, cell, neighbors):
        if cell is not None: raise Exception()

        pass
def init_gca(a):
    b = GCAProperties()
    c = GCA(b.abiogenesis, b.cull, b.spawn)
    d = CAShape(a, c.initialize, c.update)
    return CA(d.initial, d.step)



def main():
    #a = (10, 10)
    #d = init_ca(a)

    #a = random_bitstring() # TODO
    #d = init_nn(a)

    a = (10, 10)
    d = init_gca(a)

    for e in d.converge(): print(e)
    return 0
if __name__ == '__main__': exit(main())



#class Cell(object):
#    def __init__(self, bias, memory, activate, weights):
#        self.add      = add      # addition    function for affine transformation
#        self.dot      = dot      # dot product function for affine transformation
#        self.activate = activate # activation  function
#
#        self.bias     = bias     # bias/offset          for affine transformation
#        self.memory   = memory   # initial state / previous output
#        self.weights  = weights  # 
#    def forward(self, I): # TODO batch normalization ?
#        N = self.neighbors()
#        M = self.memory(I)
#        W = self.weights()
#        d = self.dot(M, W, N)
#        a = self.add(self.bias, d)
#        R = self.activate(a)
#        return R
#    def backward(self, O, l):
#        N = self.neighbors()
#        W = self.weights()
#        R = self.activate(a, inverse=True)
#        a = self.add(self.bias, R, inverse=True)
#        d = self.dot(a, W, inverse=True)
#        M = self.memory(d, inverse=True)
#        return M
#




# CA:
# - Cell: contains <scalar, vector or matrix>, bias/state (function)?, activation function, randomized weights?
# - Update(Cell a, Set<Cell> b):
#   - # TODO Set<Adj> b ==> Set<Cell>, Set<Weight> ==> backward()
#   - return activate(add(bias, dot(a, *b)))
#

# grammar => tree => recursive NN


