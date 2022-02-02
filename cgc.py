#! /usr/bin/env python3

import ast
#from dask            import delayed
from itertools       import product as ip
#from joblib          import delayed, Parallel
from collections.abc import Iterable
#from pprint          import pprint 
from random          import randrange, choice, random, getrandbits
from string          import ascii_letters, digits, punctuation
from types           import GeneratorType
#import numpy as np
#from tatsu.ast import AST
#from tatsu.objectmodel import Node
#from tatsu.semantics import ModelBuilderSemantics
#import tatsu
#from tatsu.walkers import NodeWalker

from cg_abs          import CGAbs
#from cg_type import CGType
#from cg      import CG

#np.random.seed(1)

from functools import wraps
from dask import bag, delayed

# TODO
#def product(*args, repeat=1):
#    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
#    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
#    pools = [tuple(pool) for pool in args] * repeat
#    result = [[]]
#    for pool in pools:
#        result = [x+[y] for x in result for y in pool]
#    for prod in result:
#        yield tuple(prod)
#def product(*args, repeat=1):
#	r = ip(*args, repeat=repeat)
#	return bag.from_sequence(r)
def product(*funcs, repeat=None):
    if not funcs:
        yield ()
        return

    if repeat is not None:
        funcs *= repeat

    func, *rest = funcs

    for val in func():
        for res in product(*rest):
            yield (val, ) + res

from functools import partial
values = product(partial(gen1, arg1, arg2), partial(gen2, arg1))

#root = dbopen('test.fs')
def out_of_core(func):
	@wraps(func)
	def eager(*args, **kwargs):
		#print(func.__name__ + " was called")
		#r = delayed(func)(*args, **kwargs)
		#return bag.from_delayed(r)

		#root['A'] = A = ZBigArray((10,), object)
		#transaction.commit()
		#return A
		r = func(*args, **kwargs)
		return bag.from_sequence(r)
	return eager

def trace(func):
	@wraps(func)
	def log(*args, **kwargs):
		#i = '\t' * 
		#print("enter %s(%s, %s)" % (func, args, kwargs,), flush=True)
		print("enter %s" % (func.__name__,), flush=True)
		r = func(*args, **kwargs)
		#print("leave %s(%s, %s)" % (func, args, kwargs,), flush=True)
		return r
	return log

class CG(object):
	def __init__(self, max_rd=3):
		self.max_rd = max_rd

	@trace
	def build_module_ast(self):
		#pprint("build_module()")
		#A = delayed(self.make_Module)()
		A = self.make_Module()
		#pprint("build_module A: %s" % (A,))
		for a in A:#.compute():
			assert not isinstance(a, GeneratorType)
			#pprint("build_module a: %s" % (a,))
			a = ast.fix_missing_locations(a)
			#pprint("build_module a: %s" % (a,))
			yield a
	@trace
	def compile_module(self):
		A = self.build_module_ast()
		for a in A:
			assert a is not None
			try:
				b = compile(a, filename="", mode='exec', optimize=2)
				#pprint("compile_module b: %s" % (b,))
				yield a, b
			#except(SyntaxError, ValueError): pass
			#except TypeError as e: #pprint("TypeError: %s %s %s" % (e, a, b,))
			except SyntaxError: pass
			except ValueError as e:
				#pprint("ValueError: %s %s %s" % (e, a, b,))
				yield a, None
	@trace
	def exec_module(self):
		A = self.compile_module()
		for a, b in A:
			assert a is not None
			if b is None: continue # yield a, b, None
			try:
				c = exec(b)
				#pprint("exec_module b: %s" % (b,))
				yield a, b, c
			except Exception as e:
				#pprint("Error: %s %s %s %s" % (e, a, b, c,))
				yield a, b, None



	@trace
	def build_expression_ast(self):
		#pprint("build_expression()")
		A = self.make_Expression()
		for a in A:
			assert not isinstance(a, GeneratorType)
			#pprint("build_expression a: %s" % (a,))
			a = ast.fix_missing_locations(a)
			#pprint("build_expression a: %s" % (a,))
			yield a
	@trace
	def compile_expression(self):
		A = self.build_expression_ast()
		for a in A:
			try:
				a = compile(a, filename="", mode='eval', optimize=2)
				#pprint("compile_expression a: %s" % (a,))
				yield a
			except SyntaxError: pass
	@trace
	def exec_expression(self):
		A = self.compile_expression()
		for a in A:
			try:
				b = eval(a)
				#pprint("exec_expression b: %s" % (a,))
				yield a, b
			except: pass



	@out_of_core
	@trace
	def choice(self, C):
		#pprint("choice(C=%s)" % (C,))
		# TODO
		for c in C:
			#pprint("choice c: %s" % (c,))
			yield c

	#@delayed
	@out_of_core
	@trace
	def make_star(self, f, d):
		#pprint("make_star(f=%s)" % (f,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		N = 10 # TODO
		S = []
		#yield S
		#yield None
		yield []
		for n in range(N):
			#S.append(f(d+1)) # f() -> GeneratorType
			S.append(f(d)) # f() -> GeneratorType
			#yield S
			# TODO
			yield from product(*S)
			#yield from delayed(product)(*S)
			#for k in product(*S):
			#	assert not isinstance(k, GeneratorType)
			#	yield k
	@out_of_core
	@trace
	def make_optional(self, f, d):
		#pprint("make_optional(f=%s)" % (f,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield None
		#yield []
		#yield [f(d+1)] # TODO from?
		#yield [f(d)] # TODO from?
		yield from f(d)



	@out_of_core
	@trace
	def make_mod(self, d=0):
		#pprint("make_mod(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		choices = [
			self.make_Module,
			self.make_Interactive,
			self.make_Expression,
			self.make_FunctionType,
		]
		for c in self.choice(choices):
			assert not isinstance(c, GeneratorType)
			#pprint("make_mod c: %s" % (c,), indent=d)
			yield from c(d)

	#@delayed
	@out_of_core
	@trace
	def make_Module(self, d=0):
		#pprint("make_Module(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		body         = self.make_star(self.make_stmt, d)
		type_ignores = self.make_star(self.make_type_ignore, d)
		for b, ti in product(body, type_ignores):
			assert not isinstance(b,  GeneratorType)
			assert not isinstance(ti, GeneratorType)
			assert isinstance(b, Iterable)
			#pprint("make_Module b: %s, ti: %s" % (b, ti,), indent=d)
			assert len(b) == 0 or not isinstance(b[0], GeneratorType)
			yield ast.Module(body=list(b), type_ignores=list(ti))
	@out_of_core
	@trace
	def make_Interactive(self, d=0):
		#pprint("make_Interactive(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		body = self.make_star(self.make_stmt, d)
		for b in body:
			assert not isinstance(b,  GeneratorType)
			#pprint("make_Interactive b: %s" % (b,), indent=d)
			yield ast.Interactive(body=list(b))
	@out_of_core
	@trace
	def make_Expression(self, d=0):
		#pprint("make_Expression(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		body = self.make_star(self.make_stmt, d)
		for b in body:
			assert not isinstance(b,  GeneratorType)
			#pprint("make_Expression b: %s" % (b,), indent=d)
			yield ast.Expression(body=list(b))
	@out_of_core
	@trace
	def make_FunctionType(self, d=0):
		#pprint("make_FunctionType(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		argtypes = self.make_star(self.make_expr, d)
		returns  = self.make_expr(d) # TODO
		for a, r in product(argtypes, returns):
			assert not isinstance(a, GeneratorType)
			assert not isinstance(r, GeneratorType)
			#pprint("make_FunctionType a: %s, r: %s" % (a, r,), indent=d)
			yield ast.FunctionType(argtypes=a, returns=r)



	@out_of_core
	@trace
	def make_stmt(self, d):
		#pprint("make_stmt(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		choices = [
			self.make_FunctionDef,
			self.make_AsyncFunctionDef,
			self.make_ClassDef,
			self.make_Return,
			self.make_Delete,
			self.make_Assign,
			self.make_AugAssign,
			self.make_AnnAssign,
			self.make_For,
			self.make_AsyncFor,
			self.make_While,
			self.make_If,
			self.make_With,
			self.make_AsyncWith,
			self.make_Match,
			self.make_Raise,
			self.make_Try,
			self.make_Assert,
			self.make_Import,
			self.make_ImportFrom,
			self.make_Global,
			self.make_Nonlocal,
			self.make_Expr,
			self.make_Pass,
			self.make_Break,
			self.make_Continue,
		]
		for c in self.choice(choices):
			assert not isinstance(c, GeneratorType)
			#pprint("make_stmt c: %s" % (c,), indent=d)
			#for k in c(d):
			#	assert not isinstance(k, GeneratorType)
			#	#pprint("make_stmt k: %s" % (k,), indent=d)
			#	yield k
			yield from c(d)
	@out_of_core
	@trace
	def make_FunctionDef(self, d):
		#pprint("make_FunctionDef(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		name           = self.make_identifier(d)
		args           = self.make_arguments(d)
		body           = self.make_star(self.make_stmt, d)
		decorator_list = self.make_star(self.make_expr, d)
		returns        = self.make_optional(self.make_expr, d)
		type_comment   = self.make_optional(self.make_string, d)
		for n, a, b, dl, r, tc in product(name, args, body, decorator_list, returns, type_comment):
			assert not isinstance(n,  GeneratorType)
			assert not isinstance(a,  GeneratorType)
			assert not isinstance(b,  GeneratorType)
			assert not isinstance(dl, GeneratorType)
			assert not isinstance(r,  GeneratorType)
			assert not isinstance(tc, GeneratorType)
			#pprint("make_FunctionDef n: %s, a: %s, b: %s, dl: %s, r: %s, tc: %s" % (n, a, b, dl, r, tc,), indent=d)
			yield ast.FunctionDef(name=n, args=a, body=list(b), decorator_list=dl, returns=r, type_comment=tc)
	@out_of_core
	@trace
	def make_AsyncFunctionDef(self, d):
		#pprint("make_AsyncFunctionDef(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		name           = self.make_identifier(d)
		args           = self.make_arguments(d)
		body           = self.make_star(self.make_stmt, d)
		decorator_list = self.make_star(self.make_expr, d)
		returns        = self.make_optional(self.make_expr, d)
		type_comment   = self.make_optional(self.make_string, d)
		for n, a, b, dl, r, tc in product(name, args, body, decorator_list, returns, type_comment):
			assert not isinstance(n,  GeneratorType)
			assert not isinstance(a,  GeneratorType)
			assert not isinstance(b,  GeneratorType)
			assert not isinstance(dl, GeneratorType)
			assert not isinstance(r,  GeneratorType)
			assert not isinstance(tc, GeneratorType)
			#pprint("make_AsyncFunctionDef n: %s, a: %s, b: %s, dl: %s, r: %s, tc: %s" % (n, a, b, dl, r, tc,), indent=d)
			yield ast.AsyncFunctionDef(name=n, args=a, body=list(b), decorator_list=dl, returns=r, type_comment=tc)
	@out_of_core
	@trace
	def make_ClassDef(self, d):
		#pprint("make_ClassDef(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		name           = self.make_identifier(d)
		bases          = self.make_star(self.make_expr, d)
		keywords       = self.make_star(self.make_keyword, d)
		body           = self.make_star(self.make_stmt, d)
		decorator_list = self.make_star(self.make_expr, d)
		for n, ba, k, bo, dl in product(name, bases, keywords, body, decorator_list):
			assert not isinstance(n,  GeneratorType)
			assert not isinstance(ba, GeneratorType)
			assert not isinstance(k,  GeneratorType)
			assert not isinstance(bo, GeneratorType)
			assert not isinstance(dl, GeneratorType)
			#pprint("make_ClassDef n: %s, ba: %s, k: %s, bo: %s, dl: %s" % (n, ba, k, bo, dl,), indent=d)
			yield ast.ClassDef(name=n, bases=ba, keywords=k, body=list(bo), decorator_list=dl)
	@out_of_core
	@trace
	def make_Return(self, d):
		#pprint("make_Return(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		value = self.make_optional(self.make_expr, d)
		for v in value:
			assert not isinstance(v, GeneratorType)
			#pprint("make_Return v: %s" % (v,), indent=d)
			yield ast.Return(value=v)
	@out_of_core
	@trace
	def make_Delete(self, d):
		#pprint("make_Delete(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		targets = self.make_star(self.make_expr, d)
		for t in targets:
			assert not isinstance(t, GeneratorType)
			#pprint("make_Delete t: %s" % (t,), indent=d)
			yield ast.Delete(targets=t)
	@out_of_core
	@trace
	def make_Assign(self, d):
		#pprint("make_Assign(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		targets      = self.make_star(self.make_expr, d)
		value        = self.make_expr(d)
		type_comment = self.make_optional(self.make_string, d)
		for t, v, tc in product(targets, value, type_comment):
			assert not isinstance(t,  GeneratorType)
			assert not isinstance(v,  GeneratorType)
			assert not isinstance(tc, GeneratorType)
			#pprint("make_Assign t: %s, v: %s, tc: %s" % (t, v, tc,), indent=d)
			yield ast.Assign(targets=t, value=v, type_comment=tc)
	@out_of_core
	@trace
	def make_AugAssign(self, d):
		#pprint("make_AugAssign(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		target = self.make_expr(d)
		op     = self.make_operator(d)
		value  = self.make_expr(d)
		for t, o, v in product(target, op, value):
			assert not isinstance(t, GeneratorType)
			assert not isinstance(o, GeneratorType)
			assert not isinstance(v, GeneratorType)
			#pprint("make_AugAssign t: %s, o: %s, v: %s" % (t, o, v,), indent=d)
			yield ast.AugAssign(target=t, op=o, value=v)
	@out_of_core
	@trace
	def make_AnnAssign(self, d):
		#pprint("make_AnnAssign(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		target     = self.make_expr(d)
		annotation = self.make_expr(d)
		value      = self.make_optional(self.make_expr, d)
		simple     = self.make_int(d)
		for t, a, v, s in product(target, annotation, value, simple):
			assert not isinstance(t, GeneratorType)
			assert not isinstance(a, GeneratorType)
			assert not isinstance(v, GeneratorType)
			assert not isinstance(s, GeneratorType)
			#pprint("make_AnnAssign t: %s, a: %s, v: %s, s: %s" % (t, a, v, s,), indent=d)
			yield ast.AnnAssign(target=t, annotation=a, value=v, simple=s)
	@out_of_core
	@trace
	def make_For(self, d):
		#pprint("make_For(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		target       = self.make_expr(d)
		iter_        = self.make_expr(d)
		body         = self.make_star(self.make_stmt, d)
		orelse       = self.make_star(self.make_stmt, d)
		type_comment = self.make_optional(self.make_string, d)
		for t, i, b, o, tc in product(target, iter_, body, orelse, type_comment):
			assert not isinstance(t,  GeneratorType)
			assert not isinstance(i,  GeneratorType)
			assert not isinstance(b,  GeneratorType)
			assert not isinstance(o,  GeneratorType)
			assert not isinstance(tc, GeneratorType)
			#pprint("make_For t: %s, i: %s, b: %s, o: %s, tc: %s" % (t, i, b, o, tc,), indent=d)
			yield ast.For(target=t, iter=i, body=list(b), orelse=list(o), type_comment=tc)
	@out_of_core
	@trace
	def make_AsyncFor(self, d):
		#pprint("make_AsyncFor(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		target       = self.make_expr(d)
		iter_        = self.make_expr(d)
		body         = self.make_star(self.make_stmt, d)
		orelse       = self.make_star(self.make_stmt, d)
		type_comment = self.make_optional(self.make_string, d)
		for t, i, b, o, tc in product(target, iter_, body, orelse, type_comment):
			assert not isinstance(t,  GeneratorType)
			assert not isinstance(i,  GeneratorType)
			assert not isinstance(b,  GeneratorType)
			assert not isinstance(o,  GeneratorType)
			assert not isinstance(tc, GeneratorType)
			#pprint("make_AsyncFor t: %s, i: %s, b: %s, o: %s, tc: %s" % (t, i, b, o, tc,), indent=d)
			yield ast.AsyncFor(target=t, iter=i, body=list(b), orelse=list(o), type_comment=tc)
	@out_of_core
	@trace
	def make_While(self, d):
		#pprint("make_While(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		test   = self.make_expr(d)
		body   = self.make_star(self.make_stmt, d)
		orelse = self.make_star(self.make_stmt, d)
		for t, b, o in product(test, body, orelse):
			assert not isinstance(t, GeneratorType)
			assert not isinstance(b, GeneratorType)
			assert not isinstance(o, GeneratorType)
			#pprint("make_While t: %s, b: %s, o: %s" % (t, b, o,), indent=d)
			yield ast.While(test=t, body=list(b), orelse=list(o))
	@out_of_core
	@trace
	def make_If(self, d):
		#pprint("make_If(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		test   = self.make_expr(d)
		body   = self.make_star(self.make_stmt, d)
		orelse = self.make_star(self.make_stmt, d)
		for t, b, o in product(test, body, orelse):
			assert not isinstance(t, GeneratorType)
			assert not isinstance(b, GeneratorType)
			assert not isinstance(o, GeneratorType)
			#pprint("make_If t: %s, b: %s, o: %s" % (t, b, o,), indent=d)
			yield ast.If(test=t, body=list(b), orelse=list(o))
	@out_of_core
	@trace
	def make_With(self, d):
		#pprint("make_With(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		items        = self.make_star(self.make_withitem, d)
		body         = self.make_star(self.make_stmt, d)
		type_comment = self.make_optional(self.make_string, d)
		for i, b, tc in product(items, body, type_comment):
			assert not isinstance(i,  GeneratorType)
			assert not isinstance(b,  GeneratorType)
			assert not isinstance(tc, GeneratorType)
			#pprint("make_With i: %s, b: %s, tc: %s" % (i, b, tc,), indent=d)
			yield ast.With(items=i, body=list(b), type_comment=tc)
	@out_of_core
	@trace
	def make_AsyncWith(self, d):
		#pprint("make_AsyncWith(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		items        = self.make_star(self.make_withitem, d)
		body         = self.make_star(self.make_stmt, d)
		type_comment = self.make_optional(self.make_string, d)
		for i, b, tc in product(items, body, type_comment):
			assert not isinstance(i,  GeneratorType)
			assert not isinstance(b,  GeneratorType)
			assert not isinstance(tc, GeneratorType)
			#pprint("make_AsyncWith i: %s, b: %s, tc: %s" % (i, b, tc,), indent=d)
			yield ast.AsyncWith(items=i, body=list(b), type_comment=tc)
	@out_of_core
	@trace
	def make_Match(self, d):
		#pprint("make_Match(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		subject = self.make_expr(d)
		cases   = self.make_star(self.make_match_case, d)
		for s, c in product(subject, cases):
			assert not isinstance(s, GeneratorType)
			assert not isinstance(c, GeneratorType)
			#pprint("make_Match s: %s, c: %s" % (s, c,), indent=d)
			yield ast.Match(subject=s, cases=c)
	@out_of_core
	@trace
	def make_Raise(self, d):
		#pprint("make_Raise(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		exc   = self.make_optional(self.make_expr, d)
		cause = self.make_optional(self.make_expr, d)
		for e, c in product(exc, cause):
			assert not isinstance(e, GeneratorType)
			assert not isinstance(c, GeneratorType)
			#pprint("make_Raise e: %s, c: %s" % (e, c,), indent=d)
			yield ast.Raise(exc=e, cause=c)
	@out_of_core
	@trace
	def make_Try(self, d):
		#pprint("make_Try(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		body      = self.make_star(self.make_stmt, d)
		handlers  = self.make_star(self.make_excepthandler, d)
		orelse    = self.make_star(self.make_stmt, d)
		finalbody = self.make_star(self.make_stmt, d)
		for b, h, o, f in product(body, handlers, orelse, finalbody):
			assert not isinstance(b, GeneratorType)
			assert not isinstance(h, GeneratorType)
			assert not isinstance(o, GeneratorType)
			assert not isinstance(f, GeneratorType)
			#pprint("make_Try b: %s, h: %s, o: %s, f: %s" % (b, h, o, f,), indent=d)
			yield ast.Try(body=list(b), handlers=h, orelse=list(o), finalbody=list(f))
	@out_of_core
	@trace
	def make_Assert(self, d):
		#pprint("make_Assert(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		test = self.make_expr(d)
		msg  = self.make_optional(self.make_expr, d)
		for t, m in product(test, msg):
			assert not isinstance(t, GeneratorType)
			assert not isinstance(m, GeneratorType)
			#pprint("make_Assert t: %s, m: %s" % (t, m,), indent=d)
			yield ast.Assert(test=t, msg=m)
	@out_of_core
	@trace
	def make_Import(self, d):
		#pprint("make_Import(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		names = self.make_star(self.make_alias, d)
		for n in names:
			assert not isinstance(n, GeneratorType)
			#pprint("make_Import n: %s" % (n,), indent=d)
			yield ast.Import(names=list(n))
	@out_of_core
	@trace
	def make_ImportFrom(self, d):
		#pprint("make_ImportFrom(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		module = self.make_optional(self.make_identifier, d)
		names  = self.make_star(self.make_alias, d)
		level  = self.make_optional(self.make_int, d)
		for m, n, l in product(module, names, level):
			assert not isinstance(m, GeneratorType)
			assert not isinstance(n, GeneratorType)
			assert not isinstance(l, GeneratorType)
			#pprint("make_ImportFrom m: %s, n: %s, l: %s" % (m, n, l,), indent=d)
			yield ast.ImportFrom(module=m, names=list(n), level=l)
	@out_of_core
	@trace
	def make_Global(self, d):
		#pprint("make_Global(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		names = self.make_star(self.make_identifier, d)
		for n in names:
			assert not isinstance(n, GeneratorType)
			#pprint("make_Global n: %s" % (n,), indent=d)
			yield ast.Global(names=list(n))
	@out_of_core
	@trace
	def make_Nonlocal(self, d):
		#pprint("make_Nonlocal(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		names = self.make_star(self.make_identifier, d)
		for n in names:
			assert not isinstance(n, GeneratorType)
			#pprint("make_Nonlocal n: %s" % (n,), indent=d)
			yield ast.Nonlocal(names=list(n))
	@out_of_core
	@trace
	def make_Expr(self, d):
		#pprint("make_Expr(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		value = self.make_expr(d)
		for v in value:
			assert not isinstance(v, GeneratorType)
			#pprint("make_Expr v: %s" % (v,), indent=d)
			yield ast.Expr(value=v)
	@out_of_core
	@trace
	def make_Pass(self, d):
		#pprint("make_Pass(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.Pass()
	@out_of_core
	@trace
	def make_Break(self, d):
		#pprint("make_Break(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.Break()
	@out_of_core
	@trace
	def make_Continue(self, d):
		#pprint("make_Continue(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.Continue()



	@out_of_core
	@trace
	def make_expr(self, d):
		#pprint("make_expr(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		choices = [
			self.make_BoolOp,
			self.make_NamedExpr,
			self.make_BinOp,
			self.make_UnaryOp,
			self.make_Lambda,
			self.make_IfExp,
			self.make_Dict,
			self.make_Set,
			self.make_ListComp,
			self.make_SetComp,
			self.make_DictComp,
			self.make_GeneratorExp,
			self.make_Await,
			self.make_Yield,
			self.make_YieldFrom,
			self.make_Compare,
			self.make_Call,
			self.make_FormattedValue,
			self.make_JoinedStr,
			self.make_Constant,
			self.make_Attribute,
			self.make_Subscript,
			self.make_Starred,
			self.make_Name,
			self.make_List,
			self.make_Tuple,
			self.make_Slice,
		]
		for c in self.choice(choices):
			assert not isinstance(c, GeneratorType)
			#pprint("make_expr c: %s" % (c,), indent=d)
			yield from c(d)
	@out_of_core
	@trace
	def make_BoolOp(self, d):
		#pprint("make_BoolOp(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		op     = self.make_boolop(d)
		values = self.make_star(self.make_expr, d)
		for o, v in product(op, values):
			assert not isinstance(o, GeneratorType)
			assert not isinstance(v, GeneratorType)
			#pprint("make_BoolOp o: %s, v: %s" % (o, v,), indent=d)
			yield ast.BoolOp(op=o, values=v)
	@out_of_core
	@trace
	def make_NamedExpr(self, d):
		#pprint("make_NamedExpr(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		target = self.make_expr(d)
		value  = self.make_expr(d)
		for t, v in product(target, value):
			assert not isinstance(t, GeneratorType)
			assert not isinstance(v, GeneratorType)
			#pprint("make_NamedExpr t: %s, v: %s" % (t, v,), indent=d)
			yield ast.NamedExpr(target=t, value=v)
	@out_of_core
	@trace
	def make_BinOp(self, d):
		#pprint("make_BinOp(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		left  = self.make_expr(d)
		op    = self.make_operator(d)
		right = self.make_expr(d)
		for l, o, r in product(left, op, right):
			assert not isinstance(l, GeneratorType)
			assert not isinstance(o, GeneratorType)
			assert not isinstance(r, GeneratorType)
			#pprint("make_BinOp l: %s, o: %s, r: %s" % (l, o, r,), indent=d)
			yield ast.BinOp(left=l, op=o, right=r)
	@out_of_core
	@trace
	def make_UnaryOp(self, d):
		#pprint("make_UnaryOp(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		op      = self.make_unaryop(d)
		operand = self.make_expr(d)
		for o, a in product(op, operand):
			assert not isinstance(o, GeneratorType)
			assert not isinstance(a, GeneratorType)
			#pprint("make_UnaryOp o: %s, a: %s" % (o, a,), indent=d)
			yield ast.UnaryOp(op=o, operand=a)
	@out_of_core
	@trace
	def make_Lambda(self, d):
		#pprint("make_Lambda(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		args = self.make_arguments(d)
		body = self.make_expr(d)
		for a, b in product(args, body):
			assert not isinstance(a, GeneratorType)
			assert not isinstance(b, GeneratorType)
			#pprint("make_Lambda a: %s, b: %s" % (a, b,), indent=d)
			yield ast.Lambda(args=a, body=list(b))
	@out_of_core
	@trace
	def make_IfExp(self, d):
		#pprint("make_IfExp(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		test   = self.make_expr(d)
		body   = self.make_expr(d)
		orelse = self.make_expr(d)
		for t, b, o in product(test, body, orelse):
			assert not isinstance(t, GeneratorType)
			assert not isinstance(b, GeneratorType)
			assert not isinstance(o, GeneratorType)
			#pprint("make_IfExp t: %s, b: %s, o: %s" % (t, b, o,), indent=d)
			yield ast.IfExp(test=t, body=list(b), orelse=list(o))
	@out_of_core
	@trace
	def make_Dict(self, d):
		#pprint("make_Dict(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		keys   = self.make_star(self.make_expr, d)
		values = self.make_star(self.make_expr, d)
		for k, v in product(keys, values):
			assert not isinstance(k, GeneratorType)
			assert not isinstance(v, GeneratorType)
			#pprint("make_Dict k: %s, v: %s" % (k, v,), indent=d)
			yield ast.Dict(keys=k, values=v)
	@out_of_core
	@trace
	def make_Set(self, d):
		#pprint("make_Set(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		elts = self.make_star(self.make_expr, d)
		for e in elts:
			assert not isinstance(e, GeneratorType)
			#pprint("make_Set e: %s" % (e,), indent=d)
			yield ast.Set(elts=e)
	@out_of_core
	@trace
	def make_ListComp(self, d):
		#pprint("make_ListComp(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		elt        = self.make_expr(d)
		generators = self.make_star(self.make_comprehension, d)
		for e, g in product(elt, generators):
			assert not isinstance(e, GeneratorType)
			assert not isinstance(g, GeneratorType)
			#pprint("make_ListComp e: %s, g: %s" % (e, g,), indent=d)
			yield ast.ListComp(elt=e, generators=g)
	@out_of_core
	@trace
	def make_SetComp(self, d):
		#pprint("make_SetComp(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		elt        = self.make_expr(d)
		generators = self.make_star(self.make_comprehension, d)
		for e, g in product(elt, generators):
			assert not isinstance(e, GeneratorType)
			assert not isinstance(g, GeneratorType)
			#pprint("make_SetComp e: %s, g: %s" % (e, g,), indent=d)
			yield ast.SetComp(elt=e, generators=g)
	@out_of_core
	@trace
	def make_DictComp(self, d):
		#pprint("make_DictComp(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		key        = self.make_expr(d)
		value      = self.make_expr(d)
		generators = self.make_star(self.make_comprehension, d)
		for k, v, g in product(key, value, generators):
			assert not isinstance(k, GeneratorType)
			assert not isinstance(v, GeneratorType)
			assert not isinstance(g, GeneratorType)
			#pprint("make_DictComp k: %s, v: %s, g: %s" % (k, v, g,), indent=d)
			yield ast.DictComp(key=k, value=v, generators=g)
	@out_of_core
	@trace
	def make_GeneratorExp(self, d):
		#pprint("make_GeneratorExp(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		elt        = self.make_expr(d)
		generators = self.make_star(self.make_comprehension, d)
		for e, g in product(elt, generators):
			assert not isinstance(e, GeneratorType)
			assert not isinstance(g, GeneratorType)
			#pprint("make_GeneratorExp e: %s, g: %s" % (e, g,), indent=d)
			yield ast.GeneratorExp(elt=e, generators=g)
	@out_of_core
	@trace
	def make_Await(self, d):
		#pprint("make_Await(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		value = self.make_expr(d)
		for v in value:
			assert not isinstance(v, GeneratorType)
			#pprint("make_Await v: %s" % (v,), indent=d)
			yield ast.Await(value=v)
	@out_of_core
	@trace
	def make_Yield(self, d):
		#pprint("make_Yield(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		value = self.make_optional(self.make_expr, d)
		for v in value:
			assert not isinstance(v, GeneratorType)
			#pprint("make_Yield v: %s" % (v,), indent=d)
			yield ast.Yield(value=v)
	@out_of_core
	@trace
	def make_YieldFrom(self, d):
		#pprint("make_YieldFrom(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		value = self.make_expr(d)
		for v in value:
			assert not isinstance(v, GeneratorType)
			#pprint("make_YieldFrom v: %s" % (v,), indent=d)
			yield ast.YieldFrom(value=v)
	@out_of_core
	@trace
	def make_Compare(self, d):
		#pprint("make_Compare(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		left        = self.make_expr(d)
		ops         = self.make_star(self.make_cmpop, d)
		comparators = self.make_star(self.make_expr, d)
		for l, o, c in product(left, ops, comparators):
			assert not isinstance(l, GeneratorType)
			assert not isinstance(o, GeneratorType)
			assert not isinstance(c, GeneratorType)
			#pprint("make_Compare l: %s, o: %s, c: %s" % (l, o, c,), indent=d)
			yield ast.Compare(left=l, ops=o, comparators=c)
	@out_of_core
	@trace
	def make_Call(self, d):
		#pprint("make_Call(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		func     = self.make_expr(d)
		args     = self.make_star(self.make_expr, d)
		keywords = self.make_star(self.make_keyword, d)
		for f, a, k in product(func, args, keywords):
			assert not isinstance(f, GeneratorType)
			assert not isinstance(a, GeneratorType)
			assert not isinstance(k, GeneratorType)
			#pprint("make_Call f: %s, a: %s, k: %s" % (f, a, k,), indent=d)
			yield ast.Call(func=f, args=a, keywords=k)
	@out_of_core
	@trace
	def make_FormattedValue(self, d):
		#pprint("make_FormattedValue(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		value       = self.make_expr(d)
		conversion  = self.make_int(d)
		format_spec = self.make_optional(self.make_expr, d)
		for v, c, f in product(value, conversion, format_spec):
			assert not isinstance(v, GeneratorType)
			assert not isinstance(c, GeneratorType)
			assert not isinstance(f, GeneratorType)
			#pprint("make_FormattedValue v: %s, c: %s, f: %s" % (v, c, f,), indent=d)
			yield ast.FormattedValue(value=v, conversion=c, format_spec=f)
	@out_of_core
	@trace
	def make_JoinedStr(self, d):
		#pprint("make_JoinedStr(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		values = self.make_star(self.make_expr, d)
		for v in values:
			assert not isinstance(v, GeneratorType)
			#pprint("make_JoinedStr v: %s" % (v,), indent=d)
			yield ast.JoinedStr(values=v)
	@out_of_core
	@trace
	def make_Constant(self, d):
		#pprint("make_Constant(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		value = self.make_constant(d)
		kind  = self.make_optional(self.make_string, d)
		for v, k in product(value, kind):
			assert not isinstance(v, GeneratorType)
			assert not isinstance(k, GeneratorType)
			#pprint("make_Constant v: %s, k: %s" % (v, k,), indent=d)
			yield ast.Constant(value=v, kind=k)
	@out_of_core
	@trace
	def make_Attribute(self, d):
		#pprint("make_Attribute(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		value = self.make_expr(d)
		attr  = self.make_identifier(d)
		ctx   = self.make_expr_context(d)
		for v, a, c in product(value, attr, ctx):
			assert not isinstance(v, GeneratorType)
			assert not isinstance(a, GeneratorType)
			assert not isinstance(c, GeneratorType)
			#pprint("make_Attribute v: %s, a: %s, c: %s" % (v, a, c,), indent=d)
			yield ast.Attribute(value=v, attr=a, ctx=c)
	@out_of_core
	@trace
	def make_Subscript(self, d):
		#pprint("make_Subscript(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		value  = self.make_expr(d)
		slice_ = self.make_expr(d)
		ctx    = self.make_expr_context(d)
		for v, s, c in product(value, slice_, ctx):
			assert not isinstance(v, GeneratorType)
			assert not isinstance(s, GeneratorType)
			assert not isinstance(c, GeneratorType)
			#pprint("make_Subscript v: %s, s: %s, c: %s" % (v, s, c,), indent=d)
			yield ast.Subscript(value=v, slice=s, ctx=c)
	@out_of_core
	@trace
	def make_Starred(self, d):
		#pprint("make_Starred(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		value = self.make_expr(d)
		ctx   = self.make_expr_context(d)
		for v, c in product(value, ctx):
			assert not isinstance(v, GeneratorType)
			assert not isinstance(c, GeneratorType)
			#pprint("make_Starred v: %s, c: %s" % (v, c,), indent=d)
			yield ast.Starred(value=v, ctx=c)
	@out_of_core
	@trace
	def make_Name(self, d):
		#pprint("make_Name(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		id_ = self.make_identifier(d)
		ctx = self.make_expr_context(d)
		for i, c in product(id_, ctx):
			assert not isinstance(i, GeneratorType)
			assert not isinstance(c, GeneratorType)
			#pprint("make_Name i: %s, c: %s" % (i, c,), indent=d)
			yield ast.Name(id=i, ctx=c)
	@out_of_core
	@trace
	def make_List(self, d):
		#pprint("make_List(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		elts = self.make_star(self.make_expr, d)
		ctx  = self.make_expr_context(d)
		for e, c in product(elts, ctx):
			assert not isinstance(e, GeneratorType)
			assert not isinstance(c, GeneratorType)
			#pprint("make_List e: %s, c: %s" % (e, c,), indent=d)
			yield ast.List(elts=e, ctx=c)
	@out_of_core
	@trace
	def make_Tuple(self, d):
		#pprint("make_Tuple(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		elts = self.make_star(self.make_expr, d)
		ctx  = self.make_expr_context(d)
		for e, c in product(elts, ctx):
			assert not isinstance(e, GeneratorType)
			assert not isinstance(c, GeneratorType)
			#pprint("make_Tuple e: %s, c: %s" % (e, c,), indent=d)
			yield ast.Tuple(elts=e, ctx=c)
	@out_of_core
	@trace
	def make_Slice(self, d):
		#pprint("make_Slice(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		lower = self.make_optional(self.make_expr, d)
		upper = self.make_optional(self.make_expr, d)
		step  = self.make_optional(self.make_expr, d)
		for l, u, s in product(lower, upper, step):
			assert not isinstance(l, GeneratorType)
			assert not isinstance(u, GeneratorType)
			assert not isinstance(s, GeneratorType)
			#pprint("make_Slice l: %s, u: %s, s: %s" % (l, u, s,), indent=d)
			yield ast.Slice(lower=l, upper=u, step=s)



	@out_of_core
	@trace
	def make_expr_context(self, d):
		#pprint("make_expr_context(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		choices = [
			self.make_Load,
			self.make_Store,
			self.make_Del,
		]
		for c in self.choice(choices):
			assert not isinstance(c, GeneratorType)
			#pprint("make_expr_context c: %s" % (c,), indent=d)
			yield from c(d)
	@out_of_core
	@trace
	def make_Load(self, d):
		#pprint("make_Load(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.Load()
	@out_of_core
	@trace
	def make_Store(self, d):
		#pprint("make_Store(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.Store()
	@out_of_core
	@trace
	def make_Del(self, d):
		#pprint("make_Del(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.Del()



	@out_of_core
	@trace
	def make_boolop(self, d):
		#pprint("make_boolop(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		choices = [
			self.make_And,
			self.make_Or,
		]
		for c in self.choice(choices):
			assert not isinstance(c, GeneratorType)
			#pprint("make_boolop c: %s" % (c,), indent=d)
			yield from c(d)
	@out_of_core
	@trace
	def make_And(self, d):
		#pprint("make_And(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.And()
	@out_of_core
	@trace
	def make_Or(self, d):
		#pprint("make_Or(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.Or()



	@out_of_core
	@trace
	def make_operator(self, d):
		#pprint("make_operator(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		choices = [
			self.make_Add,
			self.make_Sub,
			self.make_Mult,
			self.make_MatMult,
			self.make_Div,
			self.make_Mod,
			self.make_Pow,
			self.make_LShift,
			self.make_RShift,
			self.make_BitOr,
			self.make_BitXor,
			self.make_BitAnd,
			self.make_FloorDiv,
		]
		for c in self.choice(choices):
			assert not isinstance(c, GeneratorType)
			#pprint("make_operator c: %s" % (c,), indent=d)
			yield from c(d)
	@out_of_core
	@trace
	def make_Add(self, d):
		#pprint("make_Add(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.Add()
	@out_of_core
	@trace
	def make_Sub(self, d):
		#pprint("make_Sub(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.Sub()
	@out_of_core
	@trace
	def make_Mult(self, d):
		#pprint("make_Mult(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.Mult()
	@out_of_core
	@trace
	def make_MatMult(self, d):
		#pprint("make_MatMult(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.MatMult()
	@out_of_core
	@trace
	def make_Div(self, d):
		#pprint("make_Div(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.Div()
	@out_of_core
	@trace
	def make_Mod(self, d):
		#pprint("make_Mod(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.Mod()
	@out_of_core
	@trace
	def make_Pow(self, d):
		#pprint("make_Pow(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.Pow()
	@out_of_core
	@trace
	def make_LShift(self, d):
		#pprint("make_LShift(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.LShift()
	@out_of_core
	@trace
	def make_RShift(self, d):
		#pprint("make_RShift(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.RShift()
	@out_of_core
	@trace
	def make_BitOr(self, d):
		#pprint("make_BitOr(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.BitOr()
	@out_of_core
	@trace
	def make_BitXor(self, d):
		#pprint("make_BitXor(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.BitXor()
	@out_of_core
	@trace
	def make_BitAnd(self, d):
		#pprint("make_BitAnd(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.BitAnd()
	@out_of_core
	@trace
	def make_FloorDiv(self, d):
		#pprint("make_FloorDiv(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.FloorDiv()



	@out_of_core
	@trace
	def make_unaryop(self, d):
		#pprint("make_unaryop(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		choices = [
			self.make_Invert,
			self.make_Not,
			self.make_UAdd,
			self.make_USub,
		]
		for c in self.choice(choices):
			assert not isinstance(c, GeneratorType)
			#pprint("make_unaryop c: %s" % (c,), indent=d)
			yield from c(d)
	@out_of_core
	@trace
	def make_Invert(self, d):
		#pprint("make_Invert(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.Invert()
	@out_of_core
	@trace
	def make_Not(self, d):
		#pprint("make_Not(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.Not()
	@out_of_core
	@trace
	def make_UAdd(self, d):
		#pprint("make_UAdd(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.UAdd()
	@out_of_core
	@trace
	def make_USub(self, d):
		#pprint("make_USub(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.USub()



	@out_of_core
	@trace
	def make_cmpop(self, d):
		#pprint("make_cmpop(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		choices = [
			self.make_Eq,
			self.make_NotEq,
			self.make_Lt,
			self.make_LtE,
			self.make_Gt,
			self.make_GtE,
			self.make_Is,
			self.make_IsNot,
			self.make_In,
			self.make_NotIn,
		]
		for c in self.choice(choices):
			assert not isinstance(c, GeneratorType)
			#pprint("make_cmpop c: %s" % (c,), indent=d)
			yield from c(d)
	@out_of_core
	@trace
	def make_Eq(self, d):
		#pprint("make_Eq(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.Eq()
	@out_of_core
	@trace
	def make_NotEq(self, d):
		#pprint("make_NotEq(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.NotEq()
	@out_of_core
	@trace
	def make_Lt(self, d):
		#pprint("make_Lt(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.Lt()
	@out_of_core
	@trace
	def make_LtE(self, d):
		#pprint("make_LtE(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.LtE()
	@out_of_core
	@trace
	def make_Gt(self, d):
		#pprint("make_Gt(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.Gt()
	@out_of_core
	@trace
	def make_GtE(self, d):
		#pprint("make_GtE(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.GtE()
	@out_of_core
	@trace
	def make_Is(self, d):
		#pprint("make_Is(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.Is()
	@out_of_core
	@trace
	def make_IsNot(self, d):
		#pprint("make_IsNot(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.IsNot()
	@out_of_core
	@trace
	def make_In(self, d):
		#pprint("make_In(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.In()
	@out_of_core
	@trace
	def make_NotIn(self, d):
		#pprint("make_NotIn(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		yield ast.NotIn()



	@out_of_core
	@trace
	def make_comprehension(self, d):
		#pprint("make_comprehension(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		target = self.make_expr(d)
		iter_  = self.make_expr(d)
		ifs    = self.make_star(self.make_expr, d)
		is_async = self.make_int(d)
		for t, it, i, a in product(target, iter_, ifs, is_async):
			assert not isinstance(t,  GeneratorType)
			assert not isinstance(it, GeneratorType)
			assert not isinstance(i,  GeneratorType)
			assert not isinstance(a,  GeneratorType)
			#pprint("make_comprehension t: %s, it: %s, i: %s, a: %s" % (t, it, i, a,), indent=d)
			yield ast.comprehension(target=t, iter=it, ifs=i, is_async=a)

	@out_of_core
	@trace
	def make_excepthandler(self, d):
		#pprint("make_excepthandler(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		type_ = self.make_optional(self.make_expr, d)
		name  = self.make_optional(self.make_identifier, d)
		body  = self.make_star(self.make_stmt, d)
		for t, n, b in product(type_, name, body):
			assert not isinstance(t, GeneratorType)
			assert not isinstance(n, GeneratorType)
			assert not isinstance(b, GeneratorType)
			#pprint("make_excepthandler t: %s, n: %s, b: %s" % (t, n, b,), indent=d)
			yield ast.excepthandler(type=t, name=n, body=list(b))

	@out_of_core
	@trace
	def make_arguments(self, d):
		#pprint("make_arguments(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		posonlyargs = self.make_star(self.make_arg, d)
		args        = self.make_star(self.make_arg, d)
		vararg      = self.make_optional(self.make_arg, d)
		kwonlyargs  = self.make_star(self.make_arg, d)
		kw_defaults = self.make_star(self.make_expr, d)
		kwarg       = self.make_optional(self.make_arg, d)
		defaults    = self.make_star(self.make_expr, d)
		for p, a, v, kwo, kwd, kwa, df in product(posonlyargs, args, vararg, kwonlyargs, kw_defaults, kwarg, defaults):
			assert not isinstance(p,   GeneratorType)
			assert not isinstance(a,   GeneratorType)
			assert not isinstance(v,   GeneratorType)
			assert not isinstance(kwo, GeneratorType)
			assert not isinstance(kwd, GeneratorType)
			assert not isinstance(kwa, GeneratorType)
			assert not isinstance(df,  GeneratorType)
			#pprint("make_arguments p: %s, a: %s, v: %s, kwo: %s, kwd: %s, kwa: %s, df: %s" % (p, a, v, kwo, kwd, kwa, df,), indent=d)
			yield ast.arguments(posonlyargs=p, args=a, vararg=v, kwonlyargs=kwo, kw_defaults=kwd, kwarg=kwa, defaults=df)
	
	@out_of_core
	@trace
	def make_arg(self, d):
		#pprint("make_arg(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		arg          = self.make_identifier(d)
		annotation   = self.make_optional(self.make_expr, d)
		type_comment = self.make_optional(self.make_string, d)
		for a, n, t in product(arg, annotation, type_comment):
			assert not isinstance(a, GeneratorType)
			assert not isinstance(n, GeneratorType)
			assert not isinstance(t, GeneratorType)
			#pprint("make_arg a: %s, n: %s, t: %s" % (a, n, t,), indent=d)
			yield ast.arg(arg=a, annotation=n, type_comment=t)

	@out_of_core
	@trace
	def make_keyword(self, d):
		#pprint("make_keyword(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		arg   = self.make_optional(self.make_identifier, d)
		value = self.make_expr(d)
		for a, v in product(arg, value):
			assert not isinstance(a, GeneratorType)
			assert not isinstance(v, GeneratorType)
			#pprint("make_keyword a: %s, v: %s" % (a, v,), indent=d)
			yield ast.keyword(arg=a, value=v)

	@out_of_core
	@trace
	def make_alias(self, d):
		#pprint("make_alias(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		name   = self.make_identifier(d)
		asname = self.make_optional(self.make_identifier, d)
		for n, a in product(name, asname):
			assert not isinstance(n, GeneratorType)
			assert not isinstance(a, GeneratorType)
			#pprint("make_alias n: %s, a: %s" % (n, a,), indent=d)
			yield ast.alias(name=n, asname=a)

	@out_of_core
	@trace
	def make_withitem(self, d):
		#pprint("make_withitem(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		context_expr  = self.make_expr(d)
		optional_vars = self.make_optional(self.make_expr, d)
		for c, o in product(context_expr, optional_vars):
			assert not isinstance(c, GeneratorType)
			assert not isinstance(o, GeneratorType)
			#pprint("make_withitem c: %s, o: %s" % (c, o,), indent=d)
			yield ast.withitem(context_expr=c, optional_vars=o)

	@out_of_core
	@trace
	def make_match_case(self, d):
		#pprint("make_match_case(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		pattern = self.make_pattern(d)
		guard   = self.make_optional(self.make_expr, d)
		body    = self.make_star(self.make_stmt, d)
		for p, g, b in product(pattern, guard, body):
			assert not isinstance(p, GeneratorType)
			assert not isinstance(g, GeneratorType)
			assert not isinstance(b, GeneratorType)
			#pprint("make_match_case p: %s, g: %s, b: %s" % (p, g, b,), indent=d)
			yield ast.match_case(pattern=p, guard=g, body=list(b))

	@out_of_core
	@trace
	def make_pattern(self, d):
		#pprint("make_pattern(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		choices = [
			self.make_MatchValue,
			self.make_MatchSingleton,
			self.make_MatchSequence,
			self.make_MatchMapping,
			self.make_MatchClass,
			self.make_MatchStar,
			self.make_MatchAs,
			self.make_MatchOr,
		]
		for c in self.choice(choices):
			assert not isinstance(c, GeneratorType)
			#pprint("make_pattern c: %s" % (c,), indent=d)
			yield from c(d)

	@out_of_core
	@trace
	def make_MatchValue(self, d):
		#pprint("make_MatchValue(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		value = self.make_expr(d)
		for v in value:
			assert not isinstance(v, GeneratorType)
			#pprint("make_MatchValue v: %s" % (v,), indent=d)
			yield ast.MatchValue(value=v)
	@out_of_core
	@trace
	def make_MatchSingleton(self, d):
		#pprint("make_MatchSingleton(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		value = self.make_constant(d)
		for v in value:
			assert not isinstance(v, GeneratorType)
			#pprint("make_MatchSingleton v: %s" % (v,), indent=d)
			yield ast.MatchSingleton(value=v)
	@out_of_core
	@trace
	def make_MatchSequence(self, d):
		#pprint("make_MatchSequence(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		patterns = self.make_star(self.make_pattern, d)
		for p in patterns:
			assert not isinstance(p, GeneratorType)
			#pprint("make_MatchSequence v: %s" % (p,), indent=d)
			yield ast.MatchSequence(patterns=p)
	@out_of_core
	@trace
	def make_MatchMapping(self, d):
		#pprint("make_MatchMapping(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		keys     = self.make_star(self.make_expr, d)
		patterns = self.make_star(self.make_pattern, d)
		rest     = self.make_optional(self.make_identifier, d)
		for k, p, r in product(keys, patterns, rest):
			assert not isinstance(k, GeneratorType)
			assert not isinstance(p, GeneratorType)
			assert not isinstance(r, GeneratorType)
			#pprint("make_MatchMapping k: %s, p: %s, r: %s" % (k, p, r,), indent=d)
			yield ast.MatchMapping(keys=k, patterns=p, rest=r)
	@out_of_core
	@trace
	def make_MatchClass(self, d):
		#pprint("make_MatchClass(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		cls          = self.make_expr(d)
		patterns     = self.make_star(self.make_pattern, d)
		kwd_attrs    = self.make_star(self.make_identifier, d)
		kwd_patterns = self.make_star(self.make_pattern, d)
		for c, p, ka, kp in product(cls, patterns, kwd_attrs, kwd_patterns):
			assert not isinstance(c,  GeneratorType)
			assert not isinstance(p,  GeneratorType)
			assert not isinstance(ka, GeneratorType)
			assert not isinstance(kp, GeneratorType)
			#pprint("make_MatchClass c: %s, p: %s, ka: %s, kp: %s" % (c, p, ka, kp,), indent=d)
			yield ast.MatchClass(cls=c, patterns=p, kwd_attrs=ka, kwd_patterns=kp)
	@out_of_core
	@trace
	def make_MatchStar(self, d):
		#pprint("make_MatchStar(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		name = self.make_optional(self.make_identifier, d)
		for n in name:
			assert not isinstance(n, GeneratorType)
			#pprint("make_MatchStar n: %s" % (n,), indent=d)
			yield ast.MatchStar(name=n)
	@out_of_core
	@trace
	def make_MatchAs(self, d):
		#pprint("make_MatchAs(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		pattern = self.make_optional(self.make_pattern, d)
		name    = self.make_optional(self.make_identifier, d)
		for p, n in product(pattern, name):
			assert not isinstance(p, GeneratorType)
			assert not isinstance(n, GeneratorType)
			#pprint("make_MatchAs p: %s, n: %s" % (p, n,), indent=d)
			yield ast.MatchAs(pattern=p, name=n)
	@out_of_core
	@trace
	def make_MatchOr(self, d):
		#pprint("make_MatchOr(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		patterns = self.make_star(self.make_pattern, d)
		for p in patterns:
			assert not isinstance(p, GeneratorType)
			#pprint("make_MatchOr p: %s" % (p,), indent=d)
			yield ast.MatchOr(patterns=p)



	@out_of_core
	@trace
	def make_type_ignore(self, d):
		#pprint("make_TypeIgnore(d=%s)" % (d,), indent=d)
		if d == self.max_rd: return # raise StopIteration()
		d += 1
		lineno = self.make_int(d)
		tag    = self.make_string(d)
		for l, t in product(lineno, tag):
			assert not isinstance(l, GeneratorType)
			assert not isinstance(t, GeneratorType)
			#pprint("make_TypeIgnore l: %s, t: %s" % (l, t,), indent=d)
			yield ast.TypeIgnore(lineno=l, tag=t)
	@out_of_core
	@trace
	def make_int(self, d):
		#pprint("make_int(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		# TODO
		i = randrange(-10, 10)
		yield i
	@out_of_core
	@trace
	def make_string(self, d):
		#pprint("make_string(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		# TODO
		n = randrange(10)
		c = ascii_letters + punctuation + digits
		f = lambda _: choice(c)
		s = map(f, range(n))
		r = ''.join(s)
		yield r
	@out_of_core
	@trace
	def make_identifier(self, d):
		# TODO identifier scope
		# TODO declare identifier, reference identifier

		#pprint("make_identifier(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		# TODO
		n = randrange(10)
		c = ascii_letters + digits
		f = lambda _: choice(c)
		s = map(f, range(n))

		c = ascii_letters
		s = (choice(c), *s)
		r = ''.join(s)
		yield r
	@out_of_core
	@trace
	def make_constant(self, d):
		#pprint("make_constant(d=%s)" % (d,), indent=d)
		#if d == self.max_rd: raise StopIteration()
		#d += 1
		# integer
		# float
		# complex
		# string
		# boolean
		choices = [
			self.make_int,
			self.make_float,
			#self.make_complex,
			self.make_string,
			self.make_boolean,
		]
		for c in self.choice(choices):
			assert not isinstance(c, GeneratorType)
			yield from c(d)
	@out_of_core
	@trace
	def make_float(self, d):
		r = random()
		yield r
	#def make_complex(self, d): pass
	@out_of_core
	@trace
	def make_boolean(self, d):
		b = bool(getrandbits(1))
		yield b



if __name__ == '__main__':
	for rd in range(10):
		print("rd: %s" % (rd,))
		A = CG(rd)
		for a, c, e in A.exec_module(): print("a: %s\nc: %s\ne: %s\n" % (ast.dump(a), c, e,))
		print()

