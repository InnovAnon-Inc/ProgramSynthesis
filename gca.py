#! /usr/bin/env python3

from functools import partial
import numpy as np
import scipy as sp

from scipy import stats

from math import ceil
from dataclasses import dataclass
from typing import Callable, Optional

from random import randint, random

from ca import Cell, CAType, CAShape, CA
from util.random import random_bitstring
from activate import relu

#
# Genetic Cellular Automata
#

@dataclass
class GCA(CAType): # Genetic Cellular Automata
    abiogenesis : Callable
    cull        : Callable
    spawn       : Callable
    copy        : Callable
    #def __init__(self, abiogenesis, cull, spawn, copy):
    #    self.abiogenesis = abiogenesis
    #    self.cull        = cull
    #    self.spawn       = spawn
    #    self.copy        = copy
    def initialize(self, ndx): return self.abiogenesis(ndx)
    def update(self, cell, neighbors, inverse=False):
        if inverse: raise Exception() # not supported
        if cell.val is None: # no life
            return self.spawn(cell, neighbors)
        if self.cull(cell, neighbors):
            return self.spawn(cell, neighbors)
        # no change
        return self.copy(cell)

@dataclass
class GCAProbabilities(object):
    initialize : Callable
    empty      : Callable
    breed      : Callable
    #def __init__(self, initialize, empty, breed):
    #    self.initialize = initialize
    #    self.empty      = empty
    #    self.breed      = breed
    def abiogenesis(self, ndx):
        p = np.random.random() < 0.7
        if p: return self.empty(ndx)
        return self.initialize(ndx)#, gene)
    def cull(self, cell, neighbors):
        f = lambda n: n.fitness
        F = map(f, neighbors)
        A = np.fromiter(F, dtype=float)
        f = cell.fitness
        p = sp.stats.percentileofscore(A, f, kind='rank')
        return p < .10
    def spawn(self, cell, neighbors):
        ndx = cell.pos
        
        f = lambda n: n.val is not None
        n = tuple(filter(f, neighbors))
        if len(n) == 0: return self.empty(ndx)

        p = np.random.random() < 0.5
        if p: return self.empty(ndx)
        return self.breed(ndx, n)

#@dataclass
class GCell(Cell): # Genetic Cell
    #phenotype : Callable
    #evaluate  : Callable
    def __init__(self, pos, val, phenotype, evaluate):
        Cell.__init__(self, pos, val)
        self.phenotype = phenotype
        self.evaluate  = evaluate
        #self.update()
    #def __post_init__(self):
        if self.val  is None: self.data    = None
        else:                 self.data    = self.phenotype(self.val)
        if self.data is None: self.fitness = None
        else:                 self.fitness = self.evaluate (self.data)
    def update(self): # TODO needs to return a new cell object ==> stateless
        if self.val  is None: self.data    = None
        else:                 self.data    = self.phenotype(self.val)
        if self.data is None: self.fitness = None
        else:                 self.fitness = self.evaluate (self.data)
        # TODO data, fitness
        return GCell(self.pos, self.val, self.phenotype, self.evaluate)

@dataclass
class GCAOpt(object):
    phenotype : Callable
    evaluate  : Callable
    #def __init__(self, phenotype, evaluate):
    #    self.phenotype = phenotype
    #    self.evaluate  = evaluate
    def initialize(self, ndx):
        val = random_bitstring()
        return GCell(ndx, val, self.phenotype, self.evaluate)
    def empty(self, ndx): return GCell(ndx, None, self.phenotype, self.evaluate)
    def copy(self, cell): return cell.update()
    def breed(self, ndx, neighbors):
        #t = np.partition(neighbors, -k, order=['fitness'])[-k:]
        t = tuple(filter(lambda c: c.val is not None, neighbors))
        k = ceil(.90 * len(neighbors))
        t = sorted(t, key=lambda c: c.fitness)[:k+1]

        f = lambda n: n.val
        V = tuple(map(f, t))
        v = self.crossover(V)
        return GCell(ndx, v, self.phenotype, self.evaluate)
    def crossover(self, genes):
        if len(genes) == 0: raise Exception()

        # TODO
        ret = []
        #a = add(map(sum, genes)) / add(map(len, genes))
        #while len(ret) < 0.8 * a
        while len(ret) == 0:
            for gene in genes:
                l = len(gene)
                #a = randint(0, l - 1)
                #b = randint(0, l - a - 1)
                a = randint(0, l)
                b = randint(0, l - a)
                k = gene[a:a+b]
                ret.extend(k)

        if len(genes) == 1: p = 0.30
        else:               p = 0.10
        p = random() < p
        if p:
            k = randint(0, 2)
            i = randint(0, len(gene)-1)
            #i = randint(0, len(gene))
            if k == 0: gene[i] = not gene[i]
            if k == 1: gene    = np.insert(gene, i, randint(0, 1))
            if k == 2: gene    = np.delete(gene, i)
            
        return np.array(ret)

class GCADummy(object):
    def phenotype(self, gene):
        l = len(gene)
        s = sum(gene) + 1
        #v = np.packbits(gene)
        a = relu
        #return (l, s, v, a)
        #print("gene: %s, l: %s, s: %s, a: %s" % (gene, l, s, a,))
        return (l, s, a)
    def evaluate(self, phenotype):
        #l, s, v, a = phenotype
        l, s, a = phenotype
        #x = float(l)/(s+v)
        x = float(l)/(s)
        return a(x)

def init_gca(a):
    b =   GCADummy()
    c =   GCAOpt(b.phenotype, b.evaluate)
    d =   GCAProbabilities(c.initialize, c.empty, c.breed)
    e =   GCA(d.abiogenesis, d.cull, d.spawn, c.copy)
    f =    CAShape(a, e.initialize, e.update)
    return CA(f.initial, f.step)

def main():
    a = (10, 10)
    d = init_gca(a)
    for e in d.converge(): print(e)
    return 0

if __name__ == '__main__': exit(main())

