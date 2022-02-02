#! /usr/bin/env python3

import numpy as np
from tatsu.ast import AST
from tatsu.objectmodel import Node
from tatsu.semantics import ModelBuilderSemantics
import tatsu
from tatsu.walkers import NodeWalker
import ast

from itertools import product
from types import GeneratorType

from cg_abs  import CGAbs
from cg_type import CGType





from random import randrange, shuffle
from itertools import repeat

#def uniform_decision_engine(n, d=0, st=()): return randrange(n)

class CGRecursionException(Exception): pass
class CGNoChoicesException(Exception): pass

# TODO need to vectorize the code

@CGAbs.register
class CG():
#class CG(CGAbs):
	#def __init__(self, de=uniform_decision_engine): self.de = de
	#def __init__(self, de, *args, **kwargs):
	#def __init__(self, de, recursion_depth=200):
	def __init__(self, recursion_depth=200):
		#CGAbs.__init__(self, *args, **kwargs)
		#self.de = de
		self.recursion_depth = recursion_depth



	def make_mod(self, d=0, st=()):
		print("make_mod(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		# each of these functions maps to a number
		# => stack trace is-a vector of numbers
		d  = d + 1
		st = (*st, CGType.CG_mod,)

		choices = [
			self.make_Module,
			self.make_Interactive,
			self.make_Expression,
			self.make_FunctionType,
		]
		#return self.decision_helper(choices, d, st)
		for dh in self.decision_helper(choices, d, st):
			assert not isinstance(dh, GeneratorType)
			yield dh(d, st)

	def make_Module(self, d=0, st=()):
		print("make_Module(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Module,)

		body         = self.star(self.make_stmt, d, st)
		# TODO
		#type_ignores = self.star(self.make_type_ignore, d, st)
		#type_ignores = []
		type_ignores = [[],]

		for b, ti in product(body, type_ignores):
			assert not isinstance(b,  GeneratorType)
			assert not isinstance(ti, GeneratorType)
			assert     isinstance(b,  tuple)
			yield ast.Module(body=list(b), type_ignores=ti)
		#return ast.Module(body=body, type_ignores=type_ignores)




	def decide(self, n, tries=0, d=0, st=()):
		print("decide(n=%s, tries=%s, d=%s, st=%s)" % (n, tries, d, st,), flush=True)
		#return self.de(n, tries, d, st)
		#N = list(range(n+1))
		N = list(range(n))
		shuffle(N)
		for k in N: yield k
	def decision_helper(self, choices, d, st):
		print("decision_helper(choices=%s, d=%s, st=%s)" % (choices, d, st,), flush=True)
		nt = 0
		r = self.decide(len(choices), nt, d, st)
		#for R in r: yield choices[R](d, st)
		for R in r: yield choices[R]
		#while len(choices) != 0:
		#	r = self.decide(len(choices), nt, d, st)
		#	#try:                  return choices    [r](d, st)
		#	#except CGRecursionException: choices.pop(r)
		#	##nt = nt + 1
		#	#break # TODO this messes with the vectors ?
		#	try:                   yield choices    [r](d, st)
		#	except CGRecursionException: pass
		#    choices.pop(r)
		#	#nt = nt + 1
		#raise CGNoChoicesException()

	#def star_helper(self, k, N): yield from product(*k[:N])
	def star(self, f, d=0, st=()):
		print("star(f=%s, d=%s, st=%s)" % (f, d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_star,)

		# TODO how to give feedback on bad decision ?
		arbitrary = 10
		n = self.decide(arbitrary, d, st)
		# TODO yields list of generators, but needs to yield list of generated
		#for N in n:
		#	k = range(N)
		#	k = map(lambda g: f(d, st), k)
		#	k = list(k)
		#	print("star k: %s" % (k,))
		#	#yield k
		#	#for K in product(*k):
		#	#	assert not isinstance(K, GeneratorType)
		#	#	print("star K: %s" % (K,))
		#	#	yield K
		#	yield from product(*k)
		##n = range(n)
		##n = map(lambda g: f(d, st), n)
		##return list(n)
		k = range(arbitrary)
		k = map(lambda g: f(d, st), k)
		k = list(k)
		#for N in n: yield from self.star_helper(k, N)
		N = next(n)
		K = k[:N]
		print("star K: %s" % (K,))
		#print("star product(*K): %s" % (list(product(*K)),))
		for a in product(*K): yield a
		#yield from product(*K)
	def optional(self, f, d=0, st=()):
		print("optional(f=%s, d=%s, st=%s)" % (f, d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_optional,)

		b = self.decide(2, d, st)
		#if b: return f(d, st)
		#return None
		print("optional b: %s" % (b,))
		for B in b:
			if B: yield f(d, st)
			else: yield None




	def make_Interactive(self, d=0, st=()):
		print("make_Interactive(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Interactive,)

		body = self.star(self.make_stmt, d, st)
		for b in body:
			assert not isinstance(b, GeneratorType)
			yield ast.Interaction(b)
		#return ast.Interaction(body)
	def make_Expression(self, d=0, st=()):
		print("make_Expression(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Expression,)

		body = self.make_expr(d, st)
		for b in body:
			assert not isinstance(b, GeneratorType)
			yield ast.Expression(b)
		#return ast.Expression(body)
	def make_FunctionType(self, d=0, st=()):
		print("make_FunctionType(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_FunctionType,)

		argtypes = self.star(self.make_expr, d, st)
		returns  = self.make_expr(d, st)
		for a, r in product(argtypes, returns):
			assert not isinstance(a, GeneratorType)
			assert not isinstance(r, GeneratorType)
			yield ast.FunctionType(a, r)
		#return ast.FunctionType(argtypes, returns)



	def make_stmt(self, d=0, st=()):
		print("make_stmt(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_stmt,)

		# TODO
		choices = [
			self.make_FunctionDef,
#			self.make_AsyncFunctionDef,
			self.make_ClassDef,
			self.make_Return,
#			self.make_Delete,
#			self.make_Assign,
#			self.make_AugAssign,
#			self.make_AnnAssign,
#			self.make_For,
#			self.make_AsyncFor,
#			self.make_While,
#			self.make_If,
#			self.make_With,
#			self.make_AsyncWith,
#			self.make_Match,
#			self.make_Raise,
#			self.make_Try,
#			self.make_Assert,
#			self.make_Import,
#			self.make_ImportFrom,
#			self.make_Global,
#			self.make_Nonlocal,
#			self.make_Expr,
			self.make_Pass,
			self.make_Break,
			self.make_Continue,
		]
		for dh in self.decision_helper(choices, d, st):
			assert not isinstance(dh, GeneratorType)
			# TODO
			yield from dh(d, st)
		#return self.decision_helper(choices, d, st)
		#r = self.decide(26, d, st)
		# TODO bits should define the weights? something must define weights for ast nodes
		#if r ==  0: return self.make_FunctionDef     (d, st)
		#if r ==  1: return self.make_AsyncFunctionDef(d, st)
		#if r ==  2: return self.make_ClassDef        (d, st)
		#if r ==  3: return self.make_Return          (d, st)
		#if r ==  4: return self.make_Delete          (d, st)
		#if r ==  5: return self.make_Assign          (d, st)
		#if r ==  6: return self.make_AugAssign       (d, st)
		#if r ==  7: return self.make_AnnAssign       (d, st)
		#if r ==  8: return self.make_For             (d, st)
		#if r ==  9: return self.make_AsyncFor        (d, st)
		#if r == 10: return self.make_While           (d, st)
		#if r == 11: return self.make_If              (d, st)
		#if r == 12: return self.make_With            (d, st)
		#if r == 13: return self.make_AsyncWith       (d, st)
		#if r == 14: return self.make_Match           (d, st)
		#if r == 15: return self.make_Raise           (d, st)
		#if r == 16: return self.make_Try             (d, st)
		#if r == 17: return self.make_Assert          (d, st)
		#if r == 18: return self.make_Import          (d, st)
		#if r == 19: return self.make_ImportFrom      (d, st)
		#if r == 20: return self.make_Global          (d, st)
		#if r == 21: return self.make_Nonlocal        (d, st)
		#if r == 22: return self.make_Expr            (d, st)
		#if r == 23: return self.make_Pass            (d, st)
		#if r == 24: return self.make_Break           (d, st)
		#if r == 25: return self.make_Continue        (d, st)
		#raise Exception()
	def make_FunctionDef(self, d=0, st=()):
		print("make_FunctionDef(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_FunctionDef,)

		name           = self.make_identifier(d, st) # TODO add to scope
		args           = self.make_arguments (d, st)
		body           = self.star    (self.make_stmt,   d, st)
		decorator_list = self.star    (self.make_expr,   d, st)
		returns        = self.optional(self.make_expr,   d, st)
		type_comment   = self.optional(self.make_string, d, st)
		for n, a, b, d, r, t in product(name, args, body, decorator_list, returns, type_comment):
			assert not isinstance(n, GeneratorType)
			assert not isinstance(a, GeneratorType)
			assert not isinstance(b, GeneratorType)
			assert not isinstance(d, GeneratorType)
			assert not isinstance(r, GeneratorType)
			assert not isinstance(t, GeneratorType)
			yield ast.FunctionDef(n, a, b, d, r, t)
		#return ast.FunctionDef(name, args, body, decorator_list, returns, type_comment)
	def make_AsyncFunctionDef(self, d=0, st=()):
		print("make_AsyncFunctionDef(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_AsyncFunctionDef,)

		name           = self.make_identifier(d, st) #TODO add to scope
		args           = self.make_arguments(d, st)
		body           = self.star(self.make_stmt, d, st)
		decorator_list = self.star(self.make_expr, d, st)
		returns        = self.optional(self.make_expr, d, st)
		type_comment   = self.optional(self.make_string, d, st) # TODO non-optional
		for n, a, b, d, r, t in product(name, args, body, decorator_list, returns, type_comment):
			assert not isinstance(n, GeneratorType)
			assert not isinstance(a, GeneratorType)
			assert not isinstance(b, GeneratorType)
			assert not isinstance(d, GeneratorType)
			assert not isinstance(r, GeneratorType)
			assert not isinstance(t, GeneratorType)
			yield ast.AsyncFunctionDef(n, a, b, d, r, t)
		#return ast.AsyncFunctionDef(name, args, body, decorator_list, returns, type_comment)
	def make_ClassDef(self, d=0, st=()):
		print("make_ClassDef(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_ClassDef,)

		name           = self.make_identifier(d, st) # TODO add to scope
		bases          = self.star(self.make_expr, d, st)
		keywords       = self.star(self.make_expr, d, st)
		body           = self.star(self.make_stmt, d, st)
		decorator_list = self.star(self.make_expr, d, st)
		for n, b, k, bo, d in product(name, bases, keywords, body, decorator_list):
			assert not isinstance(n, GeneratorType)
			assert not isinstance(b, GeneratorType)
			assert not isinstance(k, GeneratorType)
			assert not isinstance(bo, GeneratorType)
			assert not isinstance(d, GeneratorType)
			yield ast.ClassDef(n, b, k, bo, d)
		#return ast.ClassDef(name, bases, keywords, body, decorator)
	def make_Return(self, d=0, st=()):
		print("make_Return(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Return,)

		value = self.optional(self.make_expr, d, st) # TODO match return type
		for v in value:
			assert not isinstance(v, GeneratorType)
			yield ast.Return(v)
		#return ast.Return(value)
	def make_Delete(self, d=0, st=()):
		print("make_Delete(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Delete,)

		targets = self.star(self.make_expr, d, st) # TODO what is the expression type ?
		for t in targets:
			assert not isinstance(t, GeneratorType)
			yield ast.Delete(t)
		#return ast.Delete(targets)
	def make_Assign(self, d=0, st=()):
		print("make_Assign(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Assign,)

		targets      = self.star(self.make_expr, d, st) # TODO match type
		value        = self.make_expr(d, st) # TODO match type
		type_comment = optional(self.make_string, d, st) # TODO non-optional
		for t, v, tc in product(t, v, tc):
			assert not isinstance(t, GeneratorType)
			assert not isinstance(v, GeneratorType)
			assert not isinstance(tc, GeneratorType)
			yield ast.Assign(t, v, tc)
		#return ast.Assign(targets, value, type_comment)
	def make_AugAssign(self, d=0, st=()):
		print("make_AugAssign(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_AugAssign,)

		target = self.make_expr(d, st) # 
		op     = self.make_operator(d, st) #
		value  = self.make_expr(d, st) #
		for t, o, v in product(target, op, value):
			assert not isinstance(t, GeneratorType)
			assert not isinstance(o, GeneratorType)
			assert not isinstance(v, GeneratorType)
			yield ast.AugAssign(t, o, v)
		#return ast.AugAssign(target, op, value)
	def make_AnnAssign(self, d=0, st=()):
		print("make_AnnAssign(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_AnnAssign,)

		target     = self.make_expr(d, st) #
		annotation = self.make_expr(d, st) #
		value      = self.optional(self.make_expr, d, st) #
		simple     = self.make_int(d, st) # bool
		for t, a, v, s in product(target, annotation, value, simple):
			assert not isinstance(t, GeneratorType)
			assert not isinstance(a, GeneratorType)
			assert not isinstance(v, GeneratorType)
			assert not isinstance(s, GeneratorType)
			yield ast.AnnAssign(t, a, v, s)
		#return ast.AnnAssign(target, annotation, value, simple)
	def make_For(self, d=0, st=()):
		print("make_For(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_For,)

		target       = self.make_expr(d, st)
		iter_        = self.make_expr(d, st)
		body         = self.star(self.make_stmt, d, st)
		orelse       = self.star(self.make_stmt, d, st)
		type_comment = self.optional(self.make_string, d, st)
		for t, i, b, o, tc in product(target, iter_, body, orelse, type_comment):
			assert not isinstance(t, GeneratorType)
			assert not isinstance(i, GeneratorType)
			assert not isinstance(b, GeneratorType)
			assert not isinstance(o, GeneratorType)
			assert not isinstance(tc, GeneratorType)
			yield ast.For(t, i, b, o, tc)
		#return ast.For(target, iter_, body, orelse, type_comment)
	def make_AsyncFor(self, d=0, st=()):
		print("make_AsyncFor(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_AsyncFor,)

		target       = self.make_expr(d, st)
		iter_        = self.make_expr(d, st)
		body         = self.star(self.make_stmt, d, st)
		orelse       = self.star(self.make_stmt, d, st)
		type_comment = self.optional(self.make_string, d, st)
		for t, i, b, o, tc in product(target, iter_, body, orelse, type_comment):
			assert not isinstance(t, GeneratorType)
			assert not isinstance(i, GeneratorType)
			assert not isinstance(b, GeneratorType)
			assert not isinstance(o, GeneratorType)
			assert not isinstance(tc, GeneratorType)
			yield ast.AsyncFor(t, i, b, o, tc)
		#return ast.AsyncFor(target, iter_, body, orelse, type_comment)
	def make_While(self, d=0, st=()):
		print("make_While(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_While,)

		test   = self.make_expr(d, st)
		body   = self.star(self.make_stmt, d, st)
		orelse = self.star(self.make_stmt, d, st)
		for t, b, o in product(test, body, orelse):
			assert not isinstance(t, GeneratorType)
			assert not isinstance(b, GeneratorType)
			assert not isinstance(o, GeneratorType)
			yield ast.While(t, b, o)
		#return ast.While(test, body, orelse)
	def make_If(self, d=0, st=()):
		print("make_If(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_If,)

		test   = self.make_expr(d, st)
		body   = self.star(self.make_stmt, d, st)
		orelse = self.star(self.make_stmt, d, st)
		for t, b, o in product(test, body, orelse):
			assert not isinstance(t, GeneratorType)
			assert not isinstance(b, GeneratorType)
			assert not isinstance(o, GeneratorType)
			yield ast.If(t, b, o)
		#return ast.If(test, body, orelse)
	def make_With(self, d=0, st=()):
		print("make_With(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_With,)

		items        = self.star    (self.make_withitem, d, st)
		body         = self.star    (self.make_stmt,      d, st)
		type_comment = self.optional(self.make_string,    d, st)
		for i, b, t in product(items, body, type_comment):
			assert not isinstance(i, GeneratorType)
			assert not isinstance(b, GeneratorType)
			assert not isinstance(t, GeneratorType)
			yield ast.With(i, b, t)
		#return ast.With(items, body, type_comment)
	def make_AsyncWith(self, d=0, st=()):
		print("make_AsyncWith(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_AsyncWith,)

		items        = self.star(self.make_withitem, d, st)
		body         = self.star(self.make_stmt, d, st)
		type_comment = self.optional(self.make_string, d, st)
		for i, b, t in product(items, body, type_comment):
			assert not isinstance(i, GeneratorType)
			assert not isinstance(b, GeneratorType)
			assert not isinstance(t, GeneratorType)
			yield ast.AsyncWith(i, b, t)
		#return ast.AsyncWith(items, body, type_comment)
	def make_Match(self, d=0, st=()):
		print("make_Match(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Match,)

		subject = self.make_expr(d, st)
		cases   = self.star(self.make_match_case, d, st)
		for s, c in product(subject, cases):
			assert not isinstance(s, GeneratorType)
			assert not isinstance(c, GeneratorType)
			yield ast.Match(s, c)
		#return ast.Match(subject, cases)
	def make_Raise(self, d=0, st=()):
		print("make_Raise(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Raise,)

		exc   = self.optional(self.make_expr, d, st)
		cause = self.optional(self.make_expr, d, st)
		for e, c in product(exc, cause):
			assert not isinstance(e, GeneratorType)
			assert not isinstance(c, GeneratorType)
			yield ast.Raise(e, c)
		#return ast.Raise(exc, cause)
	def make_Try(self, d=0, st=()):
		print("make_Try(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Try,)

		body      = self.star(self.make_stmt, d, st)
		handlers  = self.star(self.make_excepthandler, d, st)
		orelse    = self.star(self.make_stmt, d, st)
		finalbody = self.star(self.make_stmt, d, st)
		for b, h, o, f in product(body, handlers, orelse, finalbody):
			assert not isinstance(b, GeneratorType)
			assert not isinstance(h, GeneratorType)
			assert not isinstance(o, GeneratorType)
			assert not isinstance(f, GeneratorType)
			yield ast.Try(b, h, o, f)
		#return ast.Try(body, handlers, orelse, finalbody)
	def make_Assert(self, d=0, st=()):
		print("make_Assert(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Assert,)

		test = self.make_test(d, st) # expr
		msg  = self.optional(self.get_msg, d, st) # expr
		for t, m in product(test, msg):
			assert not isinstance(t, GeneratorType)
			assert not isinstance(m, GeneratorType)
			yield ast.Assert(t, m)
		#return ast.Assert(test, msg)
	def make_Import(self, d=0, st=()):
		print("make_Import(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Import,)

		names = self.star(self.get_name, d, st) # TODO reference scope
		for n in names:
			assert not isinstance(n, GeneratorType)
			yield ast.Import(n)
		#return ast.Import(names)
	def make_ImportFrom(self, d=0, st=()):
		print("make_ImportFrom(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_ImportFrom,)
		# TODO
		module = self.optional(self.get_module, d, st) # identifier
		names  = self.star(self.get_name, d, st) # alias
		level  = self.optional(self.get_level, d, st) # int # level of relative import
		for m, n, l in product(module, names, level):
			assert not isinstance(m, GeneratorType)
			assert not isinstance(n, GeneratorType)
			assert not isinstance(l, GeneratorType)
			yield ast.ImportFrom(m, n, l)
		#return ast.ImportFrom(module, names, level)
	def make_Global(self, d=0, st=()):
		print("make_Global(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Global,)

		names = self.star(self.get_name, d, st) # identifier
		for n in names:
			assert not isinstance(n, GeneratorType)
			yield ast.Global(n)
		#return ast.Global(names)
	def make_Nonlocal(self, d=0, st=()):
		print("make_Nonlocal(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Nonlocal,)

		names = self.star(self.get_name, d, st) # identifier
		for n in names:
			assert not isinstance(n, GeneratorType)
			yield ast.Nonlocal(n)
		#return ast.Nonlocal(names)
	def make_Expr(self, d=0, st=()):
		print("make_Expr(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Expr,)

		expr = self.make_expr(d, st) # type?
		for e in expr:
			assert not isinstance(e, GeneratorType)
			yield ast.Expr(e)
		#return ast.Expr(expr)
	def make_Pass    (self, d=0, st=()):
		print("make_Pass(d=%s, st=%s)" % (d, st,), flush=True)
		#return ast.Pass()
		yield ast.Pass()
	def make_Break   (self, d=0, st=()):
		print("make_Break(d=%s, st=%s)" % (d, st,), flush=True)
		#return ast.Break()
		yield ast.Break()
	def make_Continue(self, d=0, st=()):
		print("make_Continue(d=%s, st=%s)" % (d, st,), flush=True)
		#return ast.Continue()
		yield ast.Continue()

	def make_expr(self, d=0, st=()):
		print("make_expr(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_expr,)

		# TODO
		choices = [
			self.make_BoolOp,
			self.make_BinOp,
			self.make_UnaryOp,
#			self.make_Lambda,
#			self.make_IfExp,
#			self.make_Dict,
#			self.make_Set,
#			self.make_List,
#			self.make_SetComp,
#			self.make_DictComp,
#			self.make_GeneratorExp,
#			self.make_Await,
#			self.make_Yield,
#			self.make_YieldFrom,
			self.make_Compare,
			self.make_Call,
#			self.make_FormattedValue,
#			self.make_JoinedStr,
			self.make_Constant,
#			self.make_Attribute,
			self.make_Subscript,
			self.make_Starred,
			self.make_Name,
			self.make_List,
			self.make_Tuple,
			self.make_Slice,
		]
		for dh in self.decision_helper(choices, d, st):
			assert not isinstance(dh, GeneratorType)
			# TODO
			yield from dh(d, st)
		#return self.decision_helper(choices, d, st)
		#r = self.decide(26, d, st)
		#if r ==  0: return self.make_BoolOp(d, st)
		#if r ==  1: return self.make_BinOp(d, st)
		#if r ==  2: return self.make_UnaryOp(d, st)
		#if r ==  3: return self.make_Lambda(d, st)
		#if r ==  4: return self.make_IfExp(d, st)
		#if r ==  5: return self.make_Dict(d, st)
		#if r ==  6: return self.make_Set(d, st)
		#if r ==  7: return self.make_List(d, st)
		#if r ==  8: return self.make_SetComp(d, st)
		#if r ==  9: return self.make_DictComp(d, st)
		#if r == 10: return self.make_GeneratorExp(d, st)
		#if r == 11: return self.make_Await(d, st)
		#if r == 12: return self.make_Yield(d, st)
		#if r == 13: return self.make_YieldFrom(d, st)
		#if r == 14: return self.make_Compare(d, st)
		#if r == 15: return self.make_Call(d, st)
		#if r == 16: return self.make_FormattedValue(d, st)
		#if r == 17: return self.make_JoinedStr(d, st)
		#if r == 18: return self.make_Constant(d, st)
		#if r == 19: return self.make_Attribute(d, st)
		#if r == 20: return self.make_Subscript(d, st)
		#if r == 21: return self.make_Starred(d, st)
		#if r == 22: return self.make_Name(d, st) # TODO reference scope ?
		#if r == 23: return self.make_List(d, st)
		#if r == 24: return self.make_Tuple(d, st)
		#if r == 25: return self.make_Slice(d, st)
		#raise Exception()
	def make_BoolOp(self, d=0, st=()):
		print("make_BoolOp(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_BoolOp,)

		op     = self.make_boolop(d, st)
		values = self.optional(self.make_expr, d, st)
		for o, v in product(op, values):
			assert not isinstance(o, GeneratorType)
			assert not isinstance(v, GeneratorType)
			yield ast.BoolOp(o, v)
		#return ast.BoolOp(op, values)
	def make_NamedExpr(self, d=0, st=()):
		print("make_NamedExpr(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_NamedExpr,)

		target = self.make_expr(d, st)
		value  = self.make_expr(d, st)
		for t, v in product(target, value):
			assert not isinstance(t, GeneratorType)
			assert not isinstance(v, GeneratorType)
			yield ast.NamedExpr(t, v)
		#return ast.NamedExpr(target, value)
	def make_BinOp(self, d=0, st=()):
		print("make_BinOp(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_BinOp,)

		left  = self.make_expr(d, st)
		op    = self.make_operator(d, st)
		right = self.make_expr(d, st)
		for l, o, r in product(left, op, right):
			assert not isinstance(l, GeneratorType)
			assert not isinstance(o, GeneratorType)
			assert not isinstance(r, GeneratorType)
			yield ast.BinOp(l, o, r)
		#return ast.BinOp(left, op, right)
	def make_UnaryOp(self, d=0, st=()):
		print("make_UnaryOp(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_UnaryOp,)

		op      = self.make_unaryop(d, st)
		operand = self.make_expr(d, st)
		for o, O in product(op, operand):
			assert not isinstance(o, GeneratorType)
			assert not isinstance(O, GeneratorType)
			yield ast.UnaryOp(o, O)
		#return ast.UnaryOp(op, operand)
	def make_Lambda(self, d=0, st=()):
		print("make_Lambda(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Lambda,)

		args = self.make_arguments(d, st)
		body = self.make_expr(d, st)
		for a, b in product(args, body):
			assert not isinstance(a, GeneratorType)
			assert not isinstance(b, GeneratorType)
			yield ast.Lambda(a, b)
		#return ast.Lambda(args, body)
	def make_IfExp(self, d=0, st=()):
		print("make_IfExp(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_IfExp,)

		test   = self.make_expr(d, st)
		body   = self.make_expr(d, st)
		orelse = self.make_expr(d, st)
		for t, b, o in product(test, body, orelse):
			assert not isinstance(t, GeneratorType)
			assert not isinstance(b, GeneratorType)
			assert not isinstance(o, GeneratorType)
			yield ast.IfExp(t, b, o)
		#return ast.IfExp(test, body, orelse)
	def make_Dict(self, d=0, st=()):
		print("make_Dict(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Dict,)

		keys   = self.star(self.make_expr, d, st)
		values = self.star(self.make_expr, d, st)
		for k, v in product(keys, values):
			assert not isinstance(k, GeneratorType)
			assert not isinstance(v, GeneratorType)
			yield ast.Dict(k, v)
		#return ast.Dict(keys, values)
	def make_Set(self, d=0, st=()):
		print("make_Set(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Set,)

		elts = self.star(self.make_expr, d, st)
		for e in elts:
			assert not isinstance(e, GeneratorType)
			yield ast.Set(e)
		#return ast.Set(elts)
	def make_ListComp(self, d=0, st=()):
		print("make_ListComp(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_ListComp,)

		elt        = self.make_expr(d, st)
		generators = self.star(self.make_comprehension, d, st)
		for e, g in product(elt, generators):
			assert not isinstance(e, GeneratorType)
			assert not isinstance(g, GeneratorType)
			yield ast.ListComp(e, g)
		#return ast.ListComp(elt, generators)
	def make_SetComp(self, d=0, st=()):
		print("make_SetComp(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_SetComp,)

		elt        = self.make_expr(d, st)
		generators = self.star(self.make_comprehension, d, st)
		for e, g in product(elt, generators):
			assert not isinstance(e, GeneratorType)
			assert not isinstance(g, GeneratorType)
			yield ast.SetComp(e, g)
		#return ast.SetComp(elt, generators)
	def make_DictComp(self, d=0, st=()):
		print("make_DictComp(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_DictComp,)

		key        = self.make_expr(d, st)
		value      = self.make_expr(d, st)
		generators = self.star(self.make_comprehension, d, st)
		for k, v, g in product(key, value, generators):
			assert not isinstance(k, GeneratorType)
			assert not isinstance(v, GeneratorType)
			assert not isinstance(g, GeneratorType)
			yield ast.DictComp(k, v, g)
		#return ast.DictComp(key, value,generators)
	def make_GeneratorExp(self, d=0, st=()):
		print("make_GeneratorExp(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_GeneratorExp,)

		elt        = self.make_expr(d, st)
		generators = self.star(self.make_comprehension, d, st)
		for e, g in product(elt, generators):
			assert not isinstance(e, GeneratorType)
			assert not isinstance(g, GeneratorType)
			yield ast.GeneratorExp(e, g)
		#return ast.GeneratorExp(elt, generators)
	def make_Await(self, d=0, st=()):
		print("make_Await(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Await,)

		value = self.make_expr(d, st)
		for v in value:
			assert not isinstance(v, GeneratorType)
			yield ast.Await(v)
		#return ast.Await(value)
	def make_Yield(self, d=0, st=()):
		print("make_Yield(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Yield,)

		value = self.optional(self.make_expr, d, st)
		for v in value:
			assert not isinstance(v, GeneratorType)
			yield ast.Yield(v)
		#return ast.Yield(value)
	def make_YieldFrom(self, d=0, st=()):
		print("make_YieldFrom(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_YieldFrom,)

		value = self.make_expr(d, st)
		for v in value:
			assert not isinstance(v, GeneratorType)
			yield ast.YieldFrom(v)
		#return ast.YieldFrom(value)
	def make_Compare(self, d=0, st=()):
		print("make_Compare(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Compare,)

		left        = self.make_expr(d, st)
		ops         = self.star(self.make_cmpop, d, st)
		comparators = self.star(self.make_expr, d, st)
		for l, o, c in product(left, ops, comparators):
			assert not isinstance(l, GeneratorType)
			assert not isinstance(o, GeneratorType)
			assert not isinstance(c, GeneratorType)
			yield ast.Compare(l, o, c)
		#return ast.Compare(left, ops, comparators)
	def make_Call(self, d=0, st=()):
		print("make_Call(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Call,)

		left     = self.make_expr(d, st)
		args     = self.star(self.make_expr, d, st)
		keywords = self.star(self.make_keyword, d, st)
		for l, a, k in product(left, args, keywords):
			assert not isinstance(l, GeneratorType)
			assert not isinstance(a, GeneratorType)
			assert not isinstance(k, GeneratorType)
			yield ast.Call(l, a, k)
		#return ast.Call(left, args, keywords)
	def make_FormattedValue(self, d=0, st=()):
		print("make_FormattedValue(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_FormattedValue,)

		value       = self.make_expr(d, st)
		conversion  = self.optional(self.make_int, d, st) # 
		format_spec = self.optional(self.make_expr, d, st)
		for v, c, f in product(value, conversion, format_spec):
			assert not isinstance(v, GeneratorType)
			assert not isinstance(c, GeneratorType)
			assert not isinstance(f, GeneratorType)
			yield ast.FormattedValue(v, c, f)
		#return ast.FormattedValue(value, conversion, format_spec)
	def make_JoinedStr(self, d=0, st=()):
		print("make_JoinedStr(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_JoinedStr,)

		values = self.star(self.make_expr, d, st)
		for v in values:
			assert not isinstance(v, GeneratorType)
			yield ast.JoinedStr(v)
		#return ast.JoinedStr(values)
	def make_Constant(self, d=0, st=()):
		print("make_Constant(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Constant,)

		value = self.make_constant(d, st)
		kind  = self.optional(self.make_string, d, st)
		for v, k in product(value, kind):
			assert not isinstance(v, GeneratorType)
			assert not isinstance(k, GeneratorType)
			yield ast.Constant(v, k)
		#return ast.Constant(value, kind)
	#    -- the following expression can appear in assignment context
	def make_Attribute(self, d=0, st=()):
		print("make_Attribute(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Attribute,)

		value = self.make_expr(d, st)
		attr  = self.make_identifier(d, st)
		ctx   = self.make_expr_context(d, st)
		for v, a, c in product(value, attr, ctx):
			assert not isinstance(v, GeneratorType)
			assert not isinstance(a, GeneratorType)
			assert not isinstance(c, GeneratorType)
			yield ast.Attribute(v, a, c)
		#return ast.Attribute(value, attr, ctx)
	def make_Subscript(self, d=0, st=()):
		print("make_Subscript(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Subscript,)

		value  = self.make_expr(d, st)
		slice_ = self.make_expr(d, st)
		ctx    = self.make_expr_context(d, st)
		for v, s, c in product(value, slice_, ctx):
			assert not isinstance(v, GeneratorType)
			assert not isinstance(s, GeneratorType)
			assert not isinstance(c, GeneratorType)
			yield ast.Subscript(v, s, c)
		#return ast.Subscript(value, slice_, ctx)
	def make_Starred(self, d=0, st=()):
		print("make_Starred(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Starred,)

		value = self.make_expr(d, st)
		ctx   = self.make_expr_context(d, st)
		for v, c in product(value, ctx):
			assert not isinstance(v, GeneratorType)
			assert not isinstance(c, GeneratorType)
			yield ast.Starred(v, c)
		#return ast.Starred(value, ctx)
	def make_Name(self, d=0, st=()):
		print("make_Name(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Name,)

		id_ = self.make_identifier(d, st) # or get?
		ctx = self.make_expr_context(d, st)
		for i, c in product(id_, ctx):
			assert not isinstance(i, GeneratorType)
			assert not isinstance(c, GeneratorType)
			yield ast.Name(i, c)
		#return ast.Name(id_, ctx)
	def make_List(self, d=0, st=()):
		print("make_List(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_List,)

		elts = self.star(self.make_expr, d, st)
		ctx = self.make_expr_context(d, st)
		for e, c in product(elts, ctx):
			assert not isinstance(e, GeneratorType)
			assert not isinstance(c, GeneratorType)
			yield ast.List(e, c)
		#return ast.List(elts, ctx)
	def make_Tuple(self, d=0, st=()):
		print("make_Tuple(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Tuple,)

		elts = self.star(self.make_expr, d, st)
		ctx  = self.make_expr_context(d, st)
		for e, c in product(elts, ctx):
			assert not isinstance(e, GeneratorType)
			assert not isinstance(c, GeneratorType)
			yield ast.Tuple(e, c)
		#return ast.Tuple(elts, ctx)
	#    -- can appear only in Subscript
	def make_Slice(self, d=0, st=()):
		print("make_Slice(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_Slice,)

		lower = self.optional(self.make_expr, d, st)
		upper = self.optional(self.make_expr, d, st)
		step  = self.optional(self.make_expr, d, st)
		for l, u, s in product(lower, upper, step):
			assert not isinstance(l, GeneratorType)
			assert not isinstance(u, GeneratorType)
			assert not isinstance(s, GeneratorType)
			yield ast.Slice(l, u, s)
		#return ast.Slice(lower, upper, step)




	def make_expr_context(self, d=0, st=()):
		print("make_expr_context(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_expr_context,)

		#r = self.decide(3, d, st) # TODO how to give feedback
		#if r == 0: return ast.Load()
		#if r == 1: return ast.Store()
		#if r == 2: return ast.Del()
		#raise Exception()

		choices = [
				ast.Load,
				ast.Store,
				ast.Del,
		]
		for dh in self.decision_helper(choices, d, st):
			assert not isinstance(dh, GeneratorType)
			yield dh()


	def make_boolop(self, d=0, st=()):
		print("make_boolop(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_boolop,)

		#r = self.decide(2, d, st)# TODO how to give feedback
		#if r == 0: return ast.And()
		#if r == 1: return ast.Or()
		#raise Exception()

		choices = [
			ast.And,
			ast.Or,
		]
		for dh in self.decision_helper(choices, d, st):
			assert not isinstance(dh, GeneratorType)
			yield dh()

	def make_operator(self, d=0, st=()):
		print("make_operator(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_operator,)

		#r = self.decide(12, d, st) # TODO how to give feedback
		#if r ==  0: return ast.Add()
		#if r ==  1: return ast.Sub()
		#if r ==  2: return ast.Mult()
		#if r ==  3: return ast.MatMult()
		#if r ==  4: return ast.Div()
		#if r ==  5: return ast.Mod()
		#if r ==  6: return ast.Pow()
		#if r ==  7: return ast.LShift()
		#if r ==  8: return ast.RShift()
		#if r ==  9: return ast.BitOr()
		#if r == 10: return ast.BitXor()
		#if r == 11: return ast.BitAnd()
		#if r == 12: return ast.FloorDiv()
		#raise Exception()

		choices = [
			ast.Add,
			ast.Sub,
			ast.Mult,
			ast.MatMult,
			ast.Div,
			ast.Mod,
			ast.Pow,
			ast.LShift,
			ast.RShift,
			ast.BitOr,
			ast.BitXor,
			ast.BitAnd,
			ast.FloorDiv,
		]
		for dh in self.decision_helper(choices, d, st):
			assert not isinstance(dh, GeneratorType)
			yield dh()

	def make_unaryop(self, d=0, st=()):
		print("make_unaryop(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_unaryop,)

		#r = self.decide(4, d, st) # TODO feedback
		#if r == 0: return ast.Invert()
		#if r == 1: return ast.Not()
		#if r == 2: return ast.UAdd()
		#if r == 3: return ast.USub()
		#raise Exception()

		choices = [
			ast.Invert,
			ast.Not,
			ast.UAdd,
			ast.USub,
		]
		for dh in self.decision_helper(choices, d, st):
			assert not isinstance(dh, GeneratorType)
			yield dh()

	def make_cmpop(self, d=0, st=()):
		print("make_cmpop(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_cmpop,)

		#r = self.decide(10, d, st) # TODO feedback
		#if r == 0: return ast.Eq()
		#if r == 1: return ast.NotEq()
		#if r == 2: return ast.Lt()
		#if r == 3: return ast.LtE()
		#if r == 4: return ast.Gt()
		#if r == 5: return ast.GtE()
		#if r == 6: return ast.Is()
		#if r == 7: return ast.IsNot()
		#if r == 8: return ast.In()
		#if r == 9: return ast.NotIn()
		#raise Exception()

		choices = [
			ast.Eq,
			ast.NotEq,
			ast.Lt,
			ast.LtE,
			ast.Gt,
			ast.GtE,
			ast.Is,
			ast.IsNot,
			ast.In,
			ast.NotIn,
		]
		for dh in self.decision_helper(choices, d, st):
			assert not isinstance(dh, GeneratorType)
			yield dh()



	def make_comprehension(self, d=0, st=()):
		print("make_comprehension(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_comprehension,)

		target   = self.make_expr(d, st)
		iter_    = self.make_expr(d, st)
		ifs      = self.star(self.make_expr, d, st)
		is_async = self.make_int(d, st) # bool?
		for t, i, f, a in product(target, iter_, ifs, is_async):
			assert not isinstance(t, GeneratorType)
			assert not isinstance(i, GeneratorType)
			assert not isinstance(f, GeneratorType)
			assert not isinstance(a, GeneratorType)
			yield ast.comprehension(t, i, f, a)
		#return ast.comprehension(target, iter_, ifs, is_async)

	def make_excepthandler(self, d=0, st=()):
		print("make_excepthandler(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_excepthandler,)

		type_ = self.optional(self.make_expr, d, st)
		name  = self.optional(self.make_identifier, d, st) # get?
		body  = self.star(self.make_stmt, d, st)
		for t, n, b in product(type_, name, body):
			assert not isinstance(t, GeneratorType)
			assert not isinstance(n, GeneratorType)
			assert not isinstance(b, GeneratorType)
			yield ast.ExceptHandler(t, n, b)
		#return ast.ExceptHandler(type_, name, body)

	def make_arguments(self, d=0, st=()):
		print("make_arguments(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_arguments,)

		posonlyargs = self.star(self.make_arg, d, st)
		args        = self.star(self.make_arg, d, st)
		vararg      = self.optional(self.make_arg, d, st)
		kwonlyargs  = self.star(self.make_arg, d, st)
		kw_defaults = self.star(self.make_expr, d, st)
		kwarg       = self.optional(self.make_arg, d, st)
		defaults    = self.star(self.make_expr, d, st)
		for p, a, v, kwo, kwd, kwa, d in product(posonlyargs, args, vararg, kwonlyargs, kw_defaults, kwarg, defaults):
			assert not isinstance(p, GeneratorType)
			assert not isinstance(a, GeneratorType)
			assert not isinstance(v, GeneratorType)
			assert not isinstance(kwo, GeneratorType)
			assert not isinstance(kwd, GeneratorType)
			assert not isinstance(kwa, GeneratorType)
			assert not isinstance(d, GeneratorType)
			yield ast.arguments(p, a, v, kwo, kwd, kwa, d)
		#return ast.arguments(posonlyargs, args, vararg, kwonlyargs, kw_defaults, kwarg, defaults)

	def make_arg(self, d=0, st=()):
		print("make_arg(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_arg,)

		arg          = self.make_identifier(d, st) # get?
		annotation   = self.optional(self.make_expr, d, st)
		type_comment = self.optional(self.make_string, d, st)
		for a, n, t in product(arg, annotation, type_comment):
			assert not isinstance(a, GeneratorType)
			assert not isinstance(n, GeneratorType)
			assert not isinstance(t, GeneratorType)
			yield ast.arg(a, n, t)
		#return ast.arg(arg, annotation, type_comment)

	def make_keyword(self, d=0, st=()):
		print("make_keyword(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_keyword,)

		arg = self.optional(self.make_identifier, d, st) # get?
		value = self.make_expr(d, st)
		for a, v in product(arg, value):
			assert not isinstance(a, GeneratorType)
			assert not isinstance(v, GeneratorType)
			yield ast.keyword(a, v)
		#return ast.keyword(arg, value)

	def make_alias(self, d=0, st=()):
		print("make_alias(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_alias,)

		name   = self.get_identifier(d, st)
		asname = self.optional(self.make_identifier, d, st)
		for n, a in product(name, asname):
			assert not isinstance(n, GeneratorType)
			assert not isinstance(a, GeneratorType)
			yield ast.alias(n, a)
		#return ast.alias(name, asname)

	def make_withitem(self, d=0, st=()):
		print("make_withitem(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_withitem,)

		context_expr  = self.make_expr(d, st)
		optional_vars = self.optional(self.make_expr, d, st)
		for c, o in product(context_expr, optional_vars):
			assert not isinstance(c, GeneratorType)
			assert not isinstance(o, GeneratorType)
			yield ast.withitem(c, o)
		#return ast.withitem(context_expr, optional_vars)

	def make_match_case(self, d=0, st=()):
		print("make_match_case(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_match_case,)
		
		pattern = self.make_pattern(d, st)
		guard   = self.optional(self.make_expr, d, st)
		body    = self.star(self.make_stmt, d, st)
		for p, g, b in product(pattern, guard, body):
			assert not isinstance(p, GeneratorType)
			assert not isinstance(g, GeneratorType)
			assert not isinstance(b, GeneratorType)
			yield ast.match_case(p, g, b)
		#return ast.match_case(pattern, guard, body)

	def make_pattern(self, d=0, st=()):
		print("make_pattern(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_pattern,)

		choices = [
			self.make_MatchValue,#(d, st),
			self.make_MatchSingleton,#(d, st),
			self.make_MatchSequence,#(d, st),
			self.make_MatchMapping,#(d, st),
			self.make_MatchClass,#(d, st),
			self.make_MatchStar,#(d, st),
			self.make_MatchAs,#(d, st),
			self.make_MatchOr,#(d, st),
		]
		for dh in self.decision_helper(choices, d, st):
			assert not isinstance(dh, GeneratorType)
			yield dh(d, st)
		#return self.decision_helper(choices, d, st)
		#r = self.decide(8, d, st)
		#if r == 0: return self.make_MatchValue(d, st)
		#if r == 1: return self.make_MatchSingleton(d, st)
		#if r == 2: return self.make_MatchSequence(d, st)
		#if r == 3: return self.make_MatchMapping(d, st)
		#if r == 4: return self.make_MatchClass(d, st)
		#if r == 5: return self.make_MatchStar(d, st)
		#if r == 6: return self.make_MatchAs(d, st)
		#if r == 7: return self.make_MatchOr(d, st)
		#raise Exception()

	def make_MatchValue(self, d=0, st=()):
		print("make_MatchValue(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_MatchValue,)

		value = self.make_expr(d, st)
		for v in value:
			assert not isinstance(v, GeneratorType)
			yield ast.MatchValue(v)
		#return ast.MatchValue(value)
	def make_MatchSingleton(self, d=0, st=()):
		print("make_MatchSingleton(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_MatchSingleton,)

		value = self.make_constant(d, st)
		for v in value:
			assert not isinstance(v, GeneratorType)
			yield ast.MatchSingleton(v)
		#return ast.MatchSingleton(value)
	def make_MatchSequence(self, d=0, st=()):
		print("make_MatchSequence(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_MatchSequence,)

		patterns = self.star(self.make_pattern, d, st)
		for p in patterns:
			assert not isinstance(p, GeneratorType)
			yield ast.MatchSequence(p)
		#return ast.MatchSequence(patterns)
	def make_MatchMapping(self, d=0, st=()):
		print("make_MatchMapping(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_MatchMapping,)

		keys     = self.star(self.make_expr, d, st)
		patterns = self.star(self.make_pattern, d, st)
		rest     = self.optional(self.make_identifier, d, st) # get?
		for k, p, r in product(keys, patterns, rest):
			assert not isinstance(k, GeneratorType)
			assert not isinstance(p, GeneratorType)
			assert not isinstance(r, GeneratorType)
			yield ast.MatchMapping(k, p, r)
		#return ast.MatchMapping(keys, patterns, rest)
	def make_MatchClass(self, d=0, st=()):
		print("make_MatchClass(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_MatchClass,)

		cls          = self.make_expr(d, st)
		patterns     = self.star(self.make_pattern, d, st)
		kwd_attrs    = self.star(self.make_identifier, d, st) # get?
		kwd_patterns = self.star(self.make_pattern, d, st)
		for c, p, ka, kp in product(cls, patterns, kwd_attrs, kwd_patterns):
			assert not isinstance(c, GeneratorType)
			assert not isinstance(p, GeneratorType)
			assert not isinstance(ka, GeneratorType)
			assert not isinstance(kp, GeneratorType)
			yield ast.MatchClass(c, p, ka, kp)
		#return ast.MatchClass(cls, patterns, kwd_attrs, kwd_patterns)
	def make_MatchStar(self, d=0, st=()):
		print("make_MatchStar(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_MatchClass,)

		name = self.optional(self.make_identifier, d, st)
		for n in name:
			assert not isinstance(n, GeneratorType)
			yield ast.MatchStar(n)
		#return ast.MatchStar(name)
	def make_MatchAs(self, d=0, st=()):
		print("make_MatchAs(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_MatchAs,)

		pattern = self.optional(self.make_pattern, d,st)
		name    = self.optional(self.identifier, d, st)
		for p, n in product(pattern, name):
			assert not isinstance(p, GeneratorType)
			assert not isinstance(n, GeneratorType)
			yield ast.MatchAs(p, n)
		#return ast.MatchAs(pattern, name)
	def make_MatchOr(self, d=0, st=()):
		print("make_MatchOr(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_MatchOr,)

		patterns = self.star(self.make_pattern, d, st)
		for p in patterns:
			assert not isinstance(p, GeneratorType)
			yield ast.MatchOr(p)
		#return ast.MatchOr(patterns)

	def make_type_ignore(self, d=0, st=()):
		print("make_type_ignore(d=%s, st=%s)" % (d, st,), flush=True)
		# TODO
		return ast.type_ignore()

	def make_int(self, d=0, st=()):
		print("make_int(d=%s, st=%s)" % (d, st,), flush=True)
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_int,)

		return self.decide(10, d, st)
	def get_level(self, d=0, st=()):
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_level,)

		return self.make_int(d, st)
	def make_char(self, d=0, st=()):
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_char,)

		#return self.make_int(d, st) % 128 # 
		for D in self.make_int(d, st): yield D % 128
	def make_string(self, d=0, st=()):
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_string,)

		s = self.star(self.make_char, d, st)
		print("make_string s: %s" % (s,))
		#s = map(lambda c: chr(c), s)
		##return ast.Str(s)
		#return ''.join(s)
		for S in s:
			print("make_string S: %s" % (S,))
			S = map(chr, S)
			S = ''.join(S)
			print("make_string S: %s" % (S,))
			yield S
	def get_module(self, d=0, st=()):
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_module,)

		return self.make_string(d, st)
	def make_identifier(self, d=0, st=()):
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_identifier,)

		return self.make_string(d, st)
	def get_name(self, d=0, st=()):
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_name,) # make_ vs get_

		return self.make_string(d, st)
	def make_constant(self, d=0, st=()):
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_constant,)

		#r = self.decide(2, d, st) # TODO loop ?
		#if r == 0: return self.make_int(d, st)
		#if r == 1: return self.make_string(d, st)
		#raise Exception()
		choices = [
				self.make_int,
				self.make_string,
		]
		for dh in self.decision_helper(choices, d, st):
			#assert not isinstance(dh, GeneratorType)
            # TODO
			yield from dh(d, st)
	def make_test(self, d=0, st=()):
		if d == self.recursion_depth: raise CGRecursionException()
		d  = d + 1
		st = (*st, CGType.CG_test,)
		#return self.make_expr(d, st)
		for e in self.make_expr(d, st): yield e

