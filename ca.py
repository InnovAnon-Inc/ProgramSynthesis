#! /usr/bin/env python3

from functools import partial
import numpy as np

from dataclasses import dataclass

from util.array import is_in_list

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

from typing import Callable, Optional
@dataclass
class CAShape(object): # topology; n-dimensional
    #shape:list[int]
    #initialize:Callable[[object,int,Optional[int]], Cell]
    #update:Callable[[object,Cell,list[Cell],bool], Cell]
    shape      : list[int]
    initialize : Callable
    update     : Callable
    #def __init__(self, shape, initialize, update):
    #    self.shape      = shape
    #    self.initialize = initialize
    #    self.update     = update
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
@dataclass
class Cell(object):
    pos : list[int]
    val : int
    #def __init__(self, pos, val):
    #    self.pos = pos
    #    self.val = val
    #def __repr__(self): return "Cell(pos=%s)" % (self.pos,)
    def __eq__(self, o):
        if not isinstance(o, type(self)): return False
        if self.pos != o.pos: return False
        if self.val != o.val: return False
        return True

class CAType(object): # behavior
    #def __init__(self): pass
    def initialize(self, ndx, val=None):
        if val is None:
            p = np.random.random() < 0.7
            if p: val = 0
            else: val = 1
        return Cell(ndx, val)
    def update(self, cell, neighbors, inverse=False):
        if inverse: raise Exception() # not supported
        f = lambda c: c.val
        V = tuple(map(f, neighbors))
        v = sum(V) + cell.val
        p = 2 <= v and v <= 3
        if p: val = 1
        else: val = 0
        return Cell(cell.pos, val)

def init_ca(a):
    b =    CAType()
    c =    CAShape(a, b.initialize, b.update)
    return CA(c.initial, c.step)

def main():
    a = (10, 10)
    d = init_ca(a)
    for e in d.converge(): print(e)
    return 0

if __name__ == '__main__': exit(main())

