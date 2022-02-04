#! /usr/bin/env python3
""" Metamorpheus in the Underworld """

__copyright__  = "Copyright (C) 2022, Innovations Anonymous"
__version__    = "0.0.0"
__license__    = "Public Domain"
__status__     = "Development"

__author__     = "Brahmjot Singh"
__maintainer__ = "Brahmjot Singh"
__email__      = "InnovAnon-Inc@protonmail.com"
__contact__    = "(801) 448-7855"
__credits__    = [ ]

import ast
import inspect
import multiprocessing
import sys

class NN(object): pass
class Visitor(ast.NodeTransformer):
    #def __init__(self, nn:NN, *args, **kwargs):
    #    ast.NodeTransformer.__init__(self, *args, **kwargs)
    #    self.nn = nn
    # TODO visit node, use NN to decide what to replace it with ?
    #foreach node is ast:
    #  replace with random subtree
    pass

def Y(tree):
    print("enter Y(tree=%s)" % (tree,))
    code = compile(tree, filename="<ast>", mode='exec', optimize=2)
    ret  = exec(code, globals()) # TODO return value
    print("leave Y(ret=%s)" % (ret,))
    return ret

def main():
    print("enter main()")
    module  = sys.modules[__name__]
    #print(module)

    source  = inspect.getsource(module)
    #print(source)

    tree    = ast.parse(source)
    #print(tree)

    #nn      = NN()
    visitor = Visitor()#nn)
    tree    = visitor.visit(tree)
    tree    = ast.fix_missing_locations(tree)
    #print(tree)

    #process = multiprocessing.Process(target=Y, args=(tree,))
    ctx     = multiprocessing.get_context("spawn")
    process = ctx.Process(target=Y, args=(tree,))

    #code    = compile(tree, filename="<ast>", mode='exec', optimize=2)
    #ctx     = multiprocessing.get_context("spawn")
    #process = ctx.Process(target=exec, args=(code, globals())) # locals()

    process.daemon = True
    process.start()
    process.join()

    print("leave main()")
    return 0

if __name__ == '__main__':
    print("test")
    exit(main())

