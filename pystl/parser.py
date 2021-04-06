from copy import deepcopy
from collections import defaultdict
from functools import reduce
import numpy as np
import operator as op
from parsimonious import Grammar, NodeVisitor
import re
import pystl.variable

from pystl.variable import M, EPS
# Define AST classes
class Expression():
    """
    A class for defining an expression objects.

    :np.array data
    """
    __slots__ = ('deter_data', 'nondeter_data')
    def __init__(self, term = None):
        """ Constructor method """
        if isinstance(term, (int, float)):
            self.deter_data = np.array([term], dtype = float)
            self.nondeter_data = np.empty(0)
        elif isinstance(term, pystl.variable.DeterVar):
            self.deter_data = np.zeros(term.idx + 1)
            self.deter_data[term.idx] = 1
            self.nondeter_data = np.empty(0)
        elif isinstance(term, pystl.variable.NondeterVar):
            self.deter_data = np.zeros(1)
            self.nondeter_data = np.zeros(term.idx + 1)
            self.nondeter_data[term.idx] = 1
        elif isinstance(term, Expression):
            self.deter_data = term.deter_data
            self.nondeter_data = term.nondeter_data
        else:
            self.deter_data = np.zeros(1)
            self.nondeter_data = np.empty(0)
            assert(term == None)
    def __str__(self):
        """ Prints information of the contract """  
        res = ["{}".format(self.deter_data[0])]
        res += ["{} deter_var_{}".format(multiplier, i) for (i, multiplier) in enumerate(self.deter_data[1:])]
        res += ["{} nondeter_var_{}".format(multiplier, i) for (i, multiplier) in enumerate(self.nondeter_data)]
        return " + ".join(res)
    def __repr__(self):
        """ Prints information of the contract """  
        return "Expression: {}".format(str(self))
    def __add__(self, other):
        other = Expression(other)
        res = Expression()
        #  print(other)
        #  print(res)
        if len(self.deter_data) > len(other.deter_data):
            res.deter_data = deepcopy(self.deter_data)
            if len(other.deter_data) > 0:
                res.deter_data[:len(other.deter_data)] += other.deter_data
        else:
            res.deter_data = deepcopy(other.deter_data)
            if len(self.deter_data) > 0:
                res.deter_data[:len(self.deter_data)] += self.deter_data
        if len(self.nondeter_data) > len(other.nondeter_data):
            res.nondeter_data = deepcopy(self.nondeter_data)
            if len(other.nondeter_data) > 0:
                res.nondeter_data[:len(other.nondeter_data)] += other.nondeter_data
        else:
            res.nondeter_data = deepcopy(other.nondeter_data)
            if len(self.nondeter_data) > 0:
                res.nondeter_data[:len(self.nondeter_data)] += self.nondeter_data
        return res
    def __radd__(self, other):
        return self.__add__(other)
    def __sub__(self, other):
        return self + (other * -1)
    def __rsub__(self, other):
        return (-1 * self) + other
    def __mul__(self, other):
        assert(isinstance(other, (int, float)))
        res = deepcopy(self)
        res.deter_data *= other
        res.nondeter_data *= other
        return res
    def __rmul__(self, other):
        return self.__mul__(other)
    def __truediv__(self, other):
        res = deepcopy(self)
        res.deter_data /= other
        res.nondeter_data /= other
        return res
    def __lt__(self, other):
        return AtomicPredicate(self - other + EPS)
    def __le__(self, other):
        return AtomicPredicate(self - other)
    def __gt__(self, other):
        return AtomicPredicate(other - self + EPS)
    def __ge__(self, other):
        return AtomicPredicate(other - self)
    def __eq__(self, other):
        return AtomicPredicate(self - other) & AtomicPredicate(other - self)
    def transform(self, deter_id_map, nondeter_id_map):
        if (len(self.deter_data) > 0):
            mask = deter_id_map[:len(self.deter_data)]
            deter_data = np.zeros(np.max(mask) + 1)
            deter_data[mask] = self.deter_data
            self.deter_data = deter_data

        if (len(self.nondeter_data) > 0):
            mask = nondeter_id_map[:len(self.nondeter_data)]
            nondeter_data = np.zeros(np.max(mask) + 1)
            nondeter_data[mask] = self.nondeter_data
            self.nondeter_data = nondeter_data

class ASTObject():
    """
    An abstract syntax tree (AST) class from which all AST objects are derived.

    :param  idx          : An unique index of an AST object.
    :type   idx          : int
    :param  ast_type     : Name of the object class which is shared by all objects of the same class.
    :type   ast_type     : str
    :param  formula_list : children of the object.
    :type   formula_list : list of ASTObject
    """
    __slots__ = ('idx', 'ast_type', 'formula_list')

    def __init__(self, ast_type, formula_list = []):
        """ Constructor method """
        self.idx          = -1
        self.ast_type     = ast_type
        self.formula_list = formula_list
    def __str__(self):
        if self.ast_type == "True":
            return "TRUE"
        elif self.ast_type == "False":
            return "FALSE"
        else: assert(False)
    def __repr__(self):
        if self.ast_type == "True":
            return "{} -> TRUE".format(self.idx)
        elif self.ast_type == "False":
            return "{} -> FALSE".format(self.idx)
        else: assert(False)
    def U(self, interval, other):
        return TemporalFormula("U", [self, other], interval)
    def R(self, interval, other):
        return TemporalFormula("R", [self, other], interval)
    def invert_ast_type(self):
        assert(self.ast_type in ("AP", "StAP", "G", "F", "U", "R", "And", "Or", "Not"))
        if self.ast_type in ("AP", "StAP"):
            pass
        elif self.ast_type == "G": self.ast_type = "F"
        elif self.ast_type == "F": self.ast_type = "G"
        elif self.ast_type == "U": self.ast_type = "R"
        elif self.ast_type == "R": self.ast_type = "U"
        elif self.ast_type == "And": self.ast_type = "Or"
        elif self.ast_type == "Or": self.ast_type = "And"
        elif self.ast_type == "Not": assert(False)
        else: assert(False)
    def transform(self, deter_id_map, nondeter_id_map):
        if self.ast_type in ('AP', 'StAP'):
            self.expr.transform(deter_id_map, nondeter_id_map)
        else:
            for f in self.formula_list:
                f.transform(deter_id_map, nondeter_id_map)
    @staticmethod
    def nontemporal_formula_construction(operator, formula_list, ignore_ap, drop_ap):
        formula_list = [f for f in formula_list if f.ast_type != ignore_ap.ast_type]
        if any(f.ast_type == drop_ap.ast_type for f in formula_list):
            return drop_ap
        elif len(formula_list) == 0:
            return ignore_ap
        elif len(formula_list) == 1:
            return formula_list[0]
        else:
            res = []
            for f in formula_list:
                if f.ast_type != operator:
                    res.append(f)
                else:
                    res.extend(f.formula_list)
            return NontemporalFormula(operator, res)
    def __and__(self, other):
        return ASTObject.nontemporal_formula_construction("And", [self, other], true, false)
    def __or__(self, other):
        return ASTObject.nontemporal_formula_construction("Or", [self, other], false, true)
    def implies(self, other):
        return ~self | other

class AtomicPredicate(ASTObject):
    """
    A class for defining an atomic predicate objects.
    
    : param expr : An inequality expression
    : type expr  : Expression
    """
    __slots__ = ('expr')

    def __init__(self, expr):
        """ Constructor method """
        super().__init__('AP')
        self.expr = expr
    def __str__(self):
        return "({} <= 0)".format(str(self.expr))
    def __repr__(self):
        return "{} -> ({} <= 0)".format(self.idx, str(self.expr))

class StochasticAtomicPredicate(ASTObject):
    """
    A class for defining a stochastic atomic predicate objects.
    
    : param expr : An inequality expression
    : type expr  : Expression
    : param prob : An minimum limit for the probability of the object
    : type prob  : float
    """

    __slots__ = ('expr', 'prob')
    def __init__(self, expr, prob):
        """ Constructor method """
        super().__init__('StAP')
        assert(isinstance(expr, Expression))
        self.expr = expr
        self.prob = Expression(prob)
    def probability(self):
        assert(not np.any(self.prob.deter_data[1:]) and not np.any(self.prob.nondeter_data))
        return self.prob.deter_data[0]
    def __str__(self):
        return "P[{}] ({} <= 0)".format(str(self.prob), str(self.expr))
    def __repr__(self):
        return "{} -> P[{}] ({} <= 0)".format(self.idx, str(self.prob), str(self.expr))

class TemporalFormula(ASTObject):
    """
    A class for defining a formula objects.
    
    :param name: An unique name of a formula object.
    :type name: str
    """

    __slots__ = ('interval')
    def __init__(self, operator, formula_list, interval):
        """ Constructor method """
        assert(operator in ("G", "F", "U", "R"))
        assert(operator not in ("G", "F") or len(formula_list) == 1)
        assert(operator not in ("U", "R") or len(formula_list) == 2)
        assert(isinstance(interval, list) and len(interval) ==2)
        super().__init__(operator, formula_list)
        self.interval = interval
    def __str__(self):
        if (self.ast_type in ("G", "F")):
            return "({}[{}, {}] {})".format(self.ast_type, self.interval[0], self.interval[1], str(self.formula_list[0]))
        elif (self.ast_type in ("U", "R")):
            return "({} {}[{}, {}] {})".format(str(self.formula_list[0]), self.ast_type, self.interval[0], self.interval[1], str(self.formula_list[1]))
        else:
            assert(False)
    def __repr__(self):
        if (self.ast_type in ("G", "F")):
            res = "{} -> ({}[{}, {}] {})".format(self.idx, self.ast_type, self.interval[0], self.interval[1], self.formula_list[0].idx)
        elif (self.ast_type in ("U", "R")):
            res = "{} -> ({} {}[{}, {}] {})".format(self.idx, self.formula_list[0].idx, self.ast_type, self.interval[0], self.interval[1], self.formula_list[1].idx)
        else:
            assert(False)

        for f in self.formula_list:
            res += '\n  '
            res += '\n  '.join(repr(f).splitlines())
        return res

class NontemporalFormula(ASTObject):
    """
    A class for defining a formula objects.
    
    :param name: An unique name of a formula object.
    :type name: str
    """

    def __init__(self, operator, formula_list):
        """ Constructor method """
        assert(operator in ("And", "Or", "Not"))
        assert(operator != "Not" or len(formula_list) == 1)
        assert(operator == "Not" or len(formula_list) >= 2)
        super().__init__(operator, formula_list)
    def __str__(self):
        if self.ast_type == "And":
            res = "(" + " & ".join([str(f) for f in self.formula_list]) + ")"
        elif self.ast_type == "Or":
            res = "(" + " | ".join([str(f) for f in self.formula_list]) + ")"
        elif self.ast_type == "Not":
            res = "(!{})".format(str(self.formula_list[0]))
        else: assert(False)
        return res
    def __repr__(self):
        res = "{} -> ".format(self.idx)
        if self.ast_type == "And":
            res += "(" + " & ".join([str(f.idx) for f in self.formula_list]) + ")"
        elif self.ast_type == "Or":
            res += "(" + " | ".join([str(f.idx) for f in self.formula_list]) + ")"
        elif self.ast_type == "Not":
            res += "(!{})".format(self.formula_list[0].idx)
        else: assert(False)

        for f in self.formula_list:
            res += '\n  '
            res += '\n  '.join(repr(f).splitlines())
        return res

def P(prob, ap):
    return StochasticAtomicPredicate(ap.expr, prob)
def G(interval, ap):
    return TemporalFormula("G", [ap], interval)
def F(interval, ap):
    return TemporalFormula("F", [ap], interval)

true = ASTObject("True")
false = ASTObject("False")

def Neg(self):
    if self.ast_type == "Not":
        return self.formula_list[0]
    elif self.ast_type == "True":
        return false
    elif self.ast_type == "False":
        return true
    else:
        return NontemporalFormula("Not", [self])
ASTObject.__invert__ = Neg

# Define Parser
ststl_grammar = Grammar('''
phi = (neg / paren_phi / true / false
     / and_outer / or_outer / implies_outer
     / u / r / g / f / AP / stAP)

neg = ("~" / "!" / "¬") __ phi
paren_phi = "(" __ phi __ ")"

true = "TRUE"/ "True"
false = "FALSE"/ "False"

and_outer = "(" __ and_inner __ ")"
and_inner = (phi __ ("∧" / "and" / "&") __ and_inner) / phi

or_outer = "(" __ or_inner __ ")"
or_inner = (phi __ ("∨" / "or" / "|") __ or_inner) / phi

implies_outer = "(" __ implies_inner __ ")"
implies_inner = (phi __ ("→" / "->") __ implies_inner) / phi

u = "(" __ phi _ "U" interval _ phi __ ")"
r = "(" __ phi _ "R" interval _ phi __ ")"
g = "(" __ "G" interval __ phi __ ")"
f = "(" __ "F" interval __ phi __ ")"

interval = "[" __ const __ "," __ const __ "]"

stAP =  "(" __ "P[" __ expression_outer __ "]" __ AP __ ")"
AP =  "(" __ expression_outer __ comparison __ expression_outer __ ")"

expression_outer =  __ (expression_inner) __
expression_inner = (term __ operator __ expression_inner) / term

term = (const_variable/ const)
const_variable = variable / (const __ variable)

comparison = "<=" / ">=" / "=>" / "=<" / "<" / ">" / "=" / "=="
operator = "+" / "-"
variable = ~r"[a-z_][a-z_\\d]*"
const = ~r"[-+]?(\\d*\\.\\d+|\\d+)"

_ = ~r"\\s"+
__ = ~r"\\s"*
''')

class Parser(NodeVisitor):
    def __init__(self, contract):
        super().__init__()
        self.contract = contract
    def __call__(self, formula: str, rule: str = "phi"):
        return self.visit(ststl_grammar[rule].parse(formula))
    def generic_visit(self, _, children):
        return children
    def visit_const(self, node, _):
        return float(node.text)
    def visit_variable(self, node, _):
        var = node.text
        if var in self.contract.deter_var_name2id:
            return self.contract.deter_var_list[self.contract.deter_var_name2id[var]]
        elif var in self.contract.nondeter_var_name2id:
            return self.contract.nondeter_var_list[self.contract.nondeter_var_name2id[var]]
        else:
            raise ValueError("Undefined variable name {}.".format(var))
    def visit_operator(self, node, children):
        return node.text
    def visit_comparison(self, node, _):
        return node.text
    def visit_const_variable(self, node, children):
        if (isinstance(children[0], pystl.variable.Var)):
            return Expression(children[0])
        elif (len(children[0]) == 3):
            return (children[0][0] * children[0][2])
        else: assert(False)
    def visit_term(self, node, children):
        return Expression(children[0])
    def visit_expression_inner(self, node, children):
        #  print("expression_inner")
        #  print(children)
        #  input()
        if isinstance(children[0], (Expression)):
            return children
        elif len(children[0]) == 5:
            ((left, _, op, _, right),) = children
            if op == '-':
                right[0] *= -1
            else: assert(op == '+')
            return [left] + right
        else: assert(False)
    def visit_expression_outer(self, node, children):
        #  print("expression_outer")
        #  print(children)
        #  input()
        return reduce(op.add, children[1])
    def visit_AP(self, node, children):
        #  print(children)
        #  input()
        (_, _, left, _, operator, _, right, _, _) = children
        #  print(left, right)
        #  input()
        if operator == '<=' or operator == '=<':
            return left <= right
        elif operator == '>=' or operator == '=>':
            return left >= right
        elif operator == '>':
            return left > right
        elif operator == '<':
            return left < right
        elif operator == '=' or operator == '==':
            return left == right
        else: assert(False)
    def visit_stAP(self, node, children):
        #  print(children)
        #  input()
        (_, _, _, _, prob, _, _, _, ap, _, _) = children
        return P(prob, ap)
    def visit_interval(self, node, children):
        #  print(children)
        #  input()
        (_, _, left, _, _, _, right, _, _) = children
        return [int(left), int(right)]
    def visit_f(self, node, children):
        #  print(children)
        #  input()
        (_, _, _, interval, _, phi, _, _) = children
        return F(interval, phi)
    def visit_g(self, node, children):
        #  print(children)
        #  input()
        (_, _, _, interval, _, phi, _, _) = children
        return G(interval, phi)
    def visit_u(self, node, children):
        #  print(children)
        #  input()
        (_, _, phi1, _, _, interval, _, phi2, _, _) = children
        return phi1.U(interval, phi2)
    def visit_r(self, node, children):
        #  print(children)
        #  input()
        (_, _, phi1, _, _, interval, _, phi2, _, _) = children
        return phi1.R(interval, phi2)
    def visit_phi(self, node, children):
        #  print("phi:{}".format(children))
        #  input()
        return children[0]

    def nontemporal_op_inner(self, _, children):
        #  print("inner:{}".format(children))
        #  input()
        if isinstance(children[0], ASTObject):
            return children
        elif (len(children[0]) == 5):
            ((left, _, _, _, right),) = children
            return [left] + right
        else: assert(False)

    visit_or_inner = nontemporal_op_inner
    visit_and_inner = nontemporal_op_inner
    visit_implies_inner = nontemporal_op_inner
    def visit_or_outer(self, node, children):
        #  print("or outer:{}".format(children))
        #  input()
        return reduce(op.or_, children[2])
    def visit_and_outer(self, node, children):
        #  print("or outer:{}".format(children))
        #  input()
        return reduce(op.and_, children[2])
    def visit_implies_outer(self, node, children):
        #  print("or outer:{}".format(children))
        #  input()
        def implies(x, y):
            return x.implies(y)
        return reduce(implies, children[2])
    def visit_true(self, node, children):
        return true
    def visit_false(self, node, children):
        return false
    def visit_phi(self, node, children):
        return children[0]
    def visit_neg(self, _, children):
        #  print("neg:{}".format(children))
        #  input()
        return ~children[2]
    def visit_paren_phi(self, node, children):
        return children[2]

