import re
from collections import defaultdict

M = 10**4
EPS = 10**-4
# Define AST classes
class ASTObject():
    """
    An abstract syntax tree (AST) class from which all AST objects are derived.

    :param name: An unique name of an AST object.
    :type name: str
    :param class_id: Name of the object class which is shared by all objects of the same class. Determines the name space of the object.
    :type class_id: str
    """

    def __init__(self, name, class_id):
        """ Constructor method """
        self.name      =  name
        self.class_id  =  class_id
    
    def __str__(self):
        return str(self.__dict__)
    def __repr__(self):
        return repr(self.__dict__)

class Expression():
    """
    A class for defining an expression objects.
    
    :param name: An unique name of an expression object.
    :type name: str
    """

    def __init__(self, term = None):
        """ Constructor method """
        self.terms = defaultdict(int)
        if (term != None):
            self.__construct(term)
        # print(self)

    def __construct(self, term):
        """ Add terms. """
        data = re.findall("([+-])?\s*([0-9]*)?\s*([a-zA-Z][a-zA-Z0-9]*)?", term)
        for t in data:
            if t == ('', '', ''):
                continue
            #  print(data)

            multiplier = 0
            if (t[0] == "" or t[0] == "+"):
                multiplier = 1
            elif t[0] == "-":
                multiplier = -1
            else:
                assert(False)

            if t[1] != "":
                multiplier *= float(t[1])

            self.add(t[2], multiplier)
    def add(self, term, multiplier):
        term = None if term == "" else term
        self.terms[term] += multiplier
    def __str__(self):
        return repr(self)
    def __repr__(self):
        return ' + '.join(['{}{}'.format(multiplier, term) if term != None else '{}'.format(multiplier) for (term, multiplier) in self.terms.items()])
    def __add__(self, other):
        res = Expression()
        res.terms.update(self.terms)
        for (term, multiplier) in other.terms.items():
            res.terms[term] += multiplier
        return res
    def __sub__(self, other):
        res = Expression()
        res.terms.update(self.terms)
        for (term, multiplier) in other.terms.items():
            res.terms[term] -= multiplier
        return res
    def __mul__(self, other):
        res = Expression()
        res.terms = defaultdict(int, {t: other * m for (t, m) in self.terms.items()})
        return res

class AtomicPredicate(ASTObject):
    """
    A class for defining an atomic predicate objects.
    
    :param name: An unique name of an atomic predicate object.
    :type name: str
    """

    def __init__(self, name, operator, expr):
        """ Constructor method """
        super().__init__(name, 'AP')
        if (operator == ">"):
            self.operator  = "<="
            self.expr      = expr * -1 + EPS
        elif (operator == "=>"):
            self.operator  = "<="
            self.expr      = expr * -1
        elif (operator == "<"):
            self.operator  = "<="
            self.expr      = expr + EPS
        else:
            assert(operator in ("==", "<="))
            self.operator  = operator
            self.expr      = expr
    def __str__(self):
        return '({} {} 0)'.format(self.expr, self.operator)
    def __repr__(self):
        return '{} -> {} {} 0'.format(self.name, self.expr, self.operator)

class StochasticAtomicPredicate(ASTObject):
    """
    A class for defining a stochastic atomic predicate objects.
    
    :param name: An unique name of a stochastic atomic predicate object.
    :type name: str
    """

    def __init__(self, name, ap, prob, negation):
        """ Constructor method """
        super().__init__(name, 'StAP')
        self.ap = ap
        self.prob = prob
        self.negation = negation
    def __str__(self):
        return "({}P[{}] {})".format("!" if self.negation else "", self.prob, str(self.ap))
    def __repr__(self):
        return "{} -> {}P[{}] {}".format(self.name, "!" if self.negation else "", self.prob, self.ap.name)

class TemporalFormula(ASTObject):
    """
    A class for defining a formula objects.
    
    :param name: An unique name of a formula object.
    :type name: str
    """

    def __init__(self, name, operator, start_time, end_time, formula_list):
        """ Constructor method """
        assert(operator not in ("G", "F") or len(formula_list) == 1)
        assert(operator not in ("U", "R") or len(formula_list) == 2)
        super().__init__(name, 'TEMFORMULA')
        self.operator = operator
        self.start_time = start_time
        self.end_time = end_time
        self.formula_list = formula_list
    def __str__(self):
        return "({}[{}, {}] {})".format(self.operator, self.start_time, self.end_time, str(self.formula_list[0]))
    def __repr__(self):
        res = "{} -> {}[{}, {}] ".format(self.name, self.operator, self.start_time, self.end_time)
        res += " ".join(f.name for f in self.formula_list)
        for f in self.formula_list:
            res += "\n  " + "\n  ".join(repr(f).splitlines())
        return res

class NontemporalFormula(ASTObject):
    """
    A class for defining a formula objects.
    
    :param name: An unique name of a formula object.
    :type name: str
    """

    def __init__(self, name, operator, formula_list):
        """ Constructor method """
        assert(len(formula_list) >= 2)
        super().__init__(name, 'NONTEMFORMULA')
        self.operator = operator
        self.formula_list = formula_list
    def __str__(self):
        res = "({}".format(str(self.formula_list[0]))
        for f in self.formula_list[1:]:
            res += " {} {}".format(self.operator, str(f))
        res += ")"
        return res
    def __repr__(self):
        res = "{} -> {}".format(self.name, self.formula_list[0].name)
        for f in self.formula_list[1:]:
            res += " {} {}".format(self.operator, f.name)
        res += "\n"
        for f in self.formula_list:
            res += "  " + repr(f) + "\n"
        return res

# Define utility functions

class Parser():
    node_id = 0
    def str2substr(self, str):
        """
        Returns a list of substrings.
        """
        # TODO: True and False handling

        # Unnest the string, if applicable
        str_len = len(str)
        if str[0] == '(' and str[str_len-1] == ')':
            tmp_str = str[1:str_len-1]
        else:
            tmp_str = str

        # Split the string into two parts
        part1, part2 = tmp_str.split(' ', 1)
        prob = None

        # Find the start and end time
        if part1[0] in ('G', 'F', 'U', 'R'):
            operator = part1[0]
            start_time, end_time = part1[2:len(part1)-1].split(',')
            start_time = int(start_time)
            end_time = int(end_time)
        elif part1[0] == 'P':
            operator = part1[0]
            start_time = 0
            end_time = 0
            prob = float(part1[2:len(part1)-1])
        else:
            operator = part1
            start_time = 0
            end_time = 0

        # Find the list of substrings
        substr_list = []
        paren_count = 0
        start = 0
        if operator in ('=>', '<=', '>', '<', '=='): # Inequality and equality operator
            substr_list = part2.split(' ')
        elif operator == 'P': # Probability operator
            substr_list.append(part2)
        else: # Non-temporal or Temporal operator
            for idx, char in enumerate(part2):
                # Update paren_count
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1

                # Add a substr to the substr_list
                if idx != 0 and char != ' ' and paren_count == 0:
                    substr_list.append(part2[start:idx+1])
                    start = idx+2

        return operator, start_time, end_time, prob, substr_list
    def negate(self, operator):
        if operator == '=>':
            return '<'
        elif operator == '<=':
            return '>'
        elif operator == '>':
            return '<='
        elif operator == '<':
            return '=>'
        elif operator == '|':
            return '&'
        elif operator == '&':
            return '|'
        elif operator == 'G':
            return 'F'
        elif operator == 'F':
            return 'G'
        elif operator == 'U':
            return 'R'
        elif operator == 'R':
            return 'U'
        else:
            assert(False)
    def name(self):
        Parser.node_id += 1
        return "node_{}".format(Parser.node_id)
    # Define ststl parser
    def __call__(self, logic_formula, negation = False):
        """
        Parse a string in AST format representing a logic formula to a parse tree.

        :param logic_formula: A string in AST format representing a logic formula.
        :type logic_formula: str
        """
        # Split the logic formula to sub-level logic formulas
        operator, start_time, end_time, prob, substr_list = self.str2substr(logic_formula)
        print(self.str2substr(logic_formula))

        # Create appropriate object of each sub-level logic formulas
        if operator in ('=>', '<=', '>', '<', '=='): # Inequality and equality operator
            if negation:
                operator = self.negate(operator)

            expr = Expression(substr_list[0]) - Expression(substr_list[1])
            AP = AtomicPredicate(self.name(), operator, expr)
            # print(AP)
            return AP

        elif operator == 'P': # Probability unary operator
            AP = self.__call__(substr_list[0], negation=False)
            stAP = StochasticAtomicPredicate(self.name(), AP, prob, negation)
            # print(stAP)
            return stAP

        elif operator == '!': # Non-temporal unary operator
            formula = self.__call__(substr_list[0], negation = not negation)
            # print(formula)
            return formula

        elif operator == '->': # Non-temporal binary operator
            operator = "|"
            if negation:
                operator = self.negate(operator)

            subformula1 = self.__call__(substr_list[0], negation= not negation)
            subformula2 = self.__call__(substr_list[1], negation= negation)

            formula = NontemporalFormula(self.name(), operator, [subformula1, subformula2])
            # print(formula)
            return formula

        elif operator in ('&', '|'): # Non-temporal multinary operator
            if negation:
                operator = self.negate(operator)

            subformula_list = []
            for i in range(len(substr_list)):
                subformula_list.append(self.__call__(substr_list[i], negation= negation))

            formula = NontemporalFormula(self.name(), operator, subformula_list)

            return formula

        elif operator in ('G', 'F', 'U', 'R'): # Temporal operators
            if negation:
                operator = self.negate(operator)

            subformula_list = []
            for i in range(len(substr_list)):
                subformula_list.append(self.__call__(substr_list[i], negation= negation))

            formula = TemporalFormula(self.name(), operator, start_time, end_time, subformula_list)
            return formula
        else:
            assert(False)
