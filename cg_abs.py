#! /usr/bin/env python3

from abc import ABCMeta, abstractmethod
#from abc import ABC, abstractmethod

class CGAbs(metaclass=ABCMeta):
#class CGAbs:#(metaclass=ABCMeta):
#class CGAbs(ABC):
	@abstractmethod
	def make_mod             (self): raise Exception()
	@abstractmethod
	def make_Module          (self): raise Exception()
	@abstractmethod
	def make_Interactive     (self): raise Exception()
	@abstractmethod
	def make_Expression      (self): raise Exception()
	@abstractmethod
	def make_FunctionType    (self): raise Exception()

	@abstractmethod
	def make_stmt            (self): raise Exception()
	@abstractmethod
	def make_FunctionDef     (self): raise Exception()
	@abstractmethod
	def make_AsyncFunctionDef(self): raise Exception()
	@abstractmethod
	def make_ClassDef        (self): raise Exception()
	@abstractmethod
	def make_Return          (self): raise Exception()
	@abstractmethod
	def make_Delete          (self): raise Exception()
	@abstractmethod
	def make_Assign          (self): raise Exception()
	@abstractmethod
	def make_AugAssign       (self): raise Exception()
	@abstractmethod
	def make_AnnAssign       (self): raise Exception()
	@abstractmethod
	def make_For             (self): raise Exception()
	@abstractmethod
	def make_AsyncFor        (self): raise Exception()
	@abstractmethod
	def make_While           (self): raise Exception()
	@abstractmethod
	def make_If              (self): raise Exception()
	@abstractmethod
	def make_With            (self): raise Exception()
	@abstractmethod
	def make_AsyncWith       (self): raise Exception()
	@abstractmethod
	def make_Match           (self): raise Exception()
	@abstractmethod
	def make_Raise           (self): raise Exception()
	@abstractmethod
	def make_Try             (self): raise Exception()
	@abstractmethod
	def make_Assert          (self): raise Exception()
	@abstractmethod
	def make_Import          (self): raise Exception()
	@abstractmethod
	def make_ImportFrom      (self): raise Exception()
	@abstractmethod
	def make_Global          (self): raise Exception()
	@abstractmethod
	def make_Nonlocal        (self): raise Exception()
	@abstractmethod
	def make_Expr            (self): raise Exception()
	@abstractmethod
	def make_Pass            (self): raise Exception()
	@abstractmethod
	def make_Break           (self): raise Exception()
	@abstractmethod
	def make_Continue        (self): raise Exception()

	@abstractmethod
	def make_expr            (self): raise Exception()
	@abstractmethod
	def make_BoolOp          (self): raise Exception()
	@abstractmethod
	def make_NamedExpr       (self): raise Exception()
	@abstractmethod
	def make_BinOp           (self): raise Exception()
	@abstractmethod
	def make_UnaryOp         (self): raise Exception()
	@abstractmethod
	def make_Lambda          (self): raise Exception()
	@abstractmethod
	def make_IfExp           (self): raise Exception()
	@abstractmethod
	def make_Dict            (self): raise Exception()
	@abstractmethod
	def make_Set             (self): raise Exception()
	@abstractmethod
	def make_ListComp        (self): raise Exception()
	@abstractmethod
	def make_SetComp         (self): raise Exception()
	@abstractmethod
	def make_DictComp        (self): raise Exception()
	@abstractmethod
	def make_GeneratorExp    (self): raise Exception()
	@abstractmethod
	def make_Await           (self): raise Exception()
	@abstractmethod
	def make_Yield           (self): raise Exception()
	@abstractmethod
	def make_YieldFrom       (self): raise Exception()
	@abstractmethod
	def make_Compare         (self): raise Exception()
	@abstractmethod
	def make_Call            (self): raise Exception()
	@abstractmethod
	def make_FormattedValue  (self): raise Exception()
	@abstractmethod
	def make_JoinedStr       (self): raise Exception()
	@abstractmethod
	def make_Constant        (self): raise Exception()
	@abstractmethod
	def make_Attribute       (self): raise Exception()
	@abstractmethod
	def make_Subscript       (self): raise Exception()
	@abstractmethod
	def make_Starred         (self): raise Exception()
	@abstractmethod
	def make_Name            (self): raise Exception()
	@abstractmethod
	def make_List            (self): raise Exception()
	@abstractmethod
	def make_Tuple           (self): raise Exception()
	@abstractmethod
	def make_Slice           (self): raise Exception()

	@abstractmethod
	def make_expr_context    (self): raise Exception()
	@abstractmethod
	def make_Load            (self): raise Exception()
	@abstractmethod
	def make_Store           (self): raise Exception()
	@abstractmethod
	def make_Del             (self): raise Exception()

	@abstractmethod
	def make_boolop          (self): raise Exception()
	@abstractmethod
	def make_And             (self): raise Exception()
	@abstractmethod
	def make_Or              (self): raise Exception()

	@abstractmethod
	def make_operator        (self): raise Exception()
	@abstractmethod
	def make_Add             (self): raise Exception()
	@abstractmethod
	def make_Sub             (self): raise Exception()
	@abstractmethod
	def make_Mult            (self): raise Exception()
	@abstractmethod
	def make_MatMult         (self): raise Exception()
	@abstractmethod
	def make_Div             (self): raise Exception()
	@abstractmethod
	def make_Mod             (self): raise Exception()
	@abstractmethod
	def make_Pow             (self): raise Exception()
	@abstractmethod
	def make_LShift          (self): raise Exception()
	@abstractmethod
	def make_RShift          (self): raise Exception()
	@abstractmethod
	def make_BitOr           (self): raise Exception()
	@abstractmethod
	def make_BitXor          (self): raise Exception()
	@abstractmethod
	def make_BitAnd          (self): raise Exception()
	@abstractmethod
	def make_FloorDiv        (self): raise Exception()

	@abstractmethod
	def make_unaryop         (self): raise Exception()
	@abstractmethod
	def make_Invert          (self): raise Exception()
	@abstractmethod
	def make_Not             (self): raise Exception()
	@abstractmethod
	def make_UAdd            (self): raise Exception()
	@abstractmethod
	def make_USub            (self): raise Exception()

	@abstractmethod
	def make_cmpop           (self): raise Exception()
	@abstractmethod
	def make_Eq              (self): raise Exception()
	@abstractmethod
	def make_NotEq           (self): raise Exception()
	@abstractmethod
	def make_Lt              (self): raise Exception()
	@abstractmethod
	def make_LtE             (self): raise Exception()
	@abstractmethod
	def make_Gt              (self): raise Exception()
	@abstractmethod
	def make_GtE             (self): raise Exception()
	@abstractmethod
	def make_Is              (self): raise Exception()
	@abstractmethod
	def make_IsNot           (self): raise Exception()
	@abstractmethod
	def make_In              (self): raise Exception()
	@abstractmethod
	def make_NotIn           (self): raise Exception()

	@abstractmethod
	def make_comprehension   (self): raise Exception()

	@abstractmethod
	def make_excepthandler   (self): raise Exception()

	@abstractmethod
	def make_arguments       (self): raise Exception()

	@abstractmethod
	def make_arg             (self): raise Exception()

	@abstractmethod
	def make_keyword         (self): raise Exception()

	@abstractmethod
	def make_alias           (self): raise Exception()

	@abstractmethod
	def make_withitem        (self): raise Exception()

	@abstractmethod
	def make_match_case      (self): raise Exception()

	@abstractmethod
	def make_pattern         (self): raise Exception()
	@abstractmethod
	def make_MatchValue      (self): raise Exception()
	@abstractmethod
	def make_MatchSingleton  (self): raise Exception()
	@abstractmethod
	def make_MatchSequence   (self): raise Exception()
	@abstractmethod
	def make_MatchMapping    (self): raise Exception()
	@abstractmethod
	def make_MatchClass      (self): raise Exception()
	@abstractmethod
	def make_MatchStar       (self): raise Exception()
	@abstractmethod
	def make_MatchAs         (self): raise Exception()
	@abstractmethod
	def make_MatchOr         (self): raise Exception()
	@abstractmethod
	def make_attributes      (self): raise Exception()

	@abstractmethod
	def make_type_ignore     (self): raise Exception()

