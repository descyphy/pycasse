from copy import deepcopy
from parsimonious import Grammar, NodeVisitor

EPS = 10**-4
M = 10**4

class ASTObject():
    """
    An abstract syntax tree (AST) class from which all AST objects are derived.

    :param  formula          : The formula of an AST object
    :type   formula          : str
    """
    __slots__ = ('formula')

    def __init__(self, formula):
        """ Constructor method """
        self.formula = formula

    def find_horizon(self, start, end):
        """ Find the horizon of the AST Object Node. """
        if type(self) in (AP, stAP, boolean):
            return [start, end]
        elif type(self) == nontemporal_unary:
            if self.operator == 'X':
                return self.children_list[0].find_horizon(start, end+1)
            else:
                return self.children_list[0].find_horizon(start, end)
        elif type(self) in (nontemporal_binary, nontemporal_multinary):
            tmp_start = 0
            tmp_end = 0
            for children in self.children_list:
                [tmp_tmp_start, tmp_tmp_end] = children.find_horizon(start, end)
                if tmp_tmp_start < tmp_start:
                    tmp_start = tmp_tmp_start
                if tmp_tmp_end > tmp_end:
                    tmp_end = tmp_tmp_end
            return [start+tmp_start, end+tmp_end]
        elif type(self) == temporal_unary:
            [curr_start, curr_end] = self.interval
            return self.children_list[0].find_horizon(start+curr_start, end+curr_end)
        elif type(self) == temporal_binary:
            [_, curr_end] = self.interval
            [tmp_tmp_start1, tmp_tmp_end1] = self.children_list[0].find_horizon(start, end+curr_end)
            [tmp_tmp_start2, tmp_tmp_end2] = self.children_list[1].find_horizon(start, end+curr_end)
            return [min(tmp_tmp_start1, tmp_tmp_start2), max(tmp_tmp_end1, tmp_tmp_end2)]

    def push_negation(self, neg=False):
        """ Remove all the negation nodes and push all the negations to the leaf node. """
        def invert_sign(formula):
            """ Invert the sign of the inequalities. """
            if ">=" in formula:
                formula = formula.replace(">=", "<")
            elif "=>" in formula:
                formula = formula.replace("=>", "<")
            elif "<=" in formula:
                formula = formula.replace("<=", ">")
            elif "=<" in formula:
                formula = formula.replace("=<", ">")
            elif ">" in formula:
                formula = formula.replace(">", "<=")
            elif "<" in formula:
                formula = formula.replace("<", "=>")
            return formula

        if type(self) == boolean:
            if neg:
                if self.formula == 'True':
                    self.formula = 'False'
                else:
                    self.formula = 'True'
            return self

        elif type(self) in (AP, stAP):
            if neg:
                if type(self) == stAP:
                    # Add [1] to prob_list_list
                    if [1] not in self.prob_var_list_list:
                        self.prob_multipliers.append(0)
                        self.prob_var_list_list.append([1])
                        self.prob_power_list_list.append([1])

                    # Find the original prob formula
                    tmp_prob_orig_formula = ""
                    count = 0
                    for char in self.formula:
                        if char == "]":
                            break
                        elif count == 1:
                            tmp_prob_orig_formula += char
                        
                        if char == "[":
                            count += 1

                    # Find the negated prob formula
                    tmp_prob_neg_formula = ""
                    for i, multiplier in enumerate(self.prob_multipliers):
                        self.prob_multipliers[i] = -multiplier

                        if self.prob_var_list_list[i] == [1]:
                            self.prob_multipliers[i] += 1

                        if self.prob_multipliers[i] < 0:
                            tmp_prob_neg_formula += "{}".format(self.prob_multipliers[i])
                        else:
                            if tmp_prob_neg_formula == "":
                                tmp_prob_neg_formula += "{}".format(self.prob_multipliers[i])
                            else:
                                tmp_prob_neg_formula += "+{}".format(self.prob_multipliers[i])

                        for var, power in zip(self.prob_var_list_list[i], self.prob_power_list_list[i]):
                            if var != 1:
                                if power == 1:
                                    tmp_prob_neg_formula += "*{}".format(var)
                                else:
                                    tmp_prob_neg_formula += "*{}^{}".format(var, power)
                    self.formula = self.formula.replace("P[{}]".format(tmp_prob_orig_formula), "P[{}]".format(tmp_prob_neg_formula))
                            
                self.formula = invert_sign(self.formula)
                self.multipliers = [-x for x in self.multipliers]
                if [1] in self.var_list_list:
                    const_index = self.var_list_list.index([1])
                    self.multipliers[const_index] += EPS
                else:
                    self.multipliers.append(EPS)
                    self.var_list_list.append([1])
                    self.power_list_list.append([1])

            return self

        elif type(self) == nontemporal_unary:
            if self.operator == '!':
                self.children_list[0] = self.children_list[0].push_negation(neg=not neg)
                return self.children_list[0]
            else:
                self.children_list[0] = self.children_list[0].push_negation(neg=neg)
                return self

        elif type(self) == nontemporal_binary:
            new_self = nontemporal_multinary(None)
            new_self.formula = self.formula
            new_self.variables = self.variables

            if neg:
                new_self.operator = "&"
                new_self.children_list.append(self.children_list[0].push_negation(neg=not neg))
                new_self.children_list.append(self.children_list[1].push_negation(neg=neg))
                new_self.formula = "({}) & ({})".format(new_self.children_list[0].formula, new_self.children_list[1].formula)
            else:
                new_self.operator = "|"
                new_self.children_list.append(self.children_list[0].push_negation(neg=not neg))
                new_self.children_list.append(self.children_list[1].push_negation(neg=neg))
                new_self.formula = "({}) | ({})".format(new_self.children_list[0].formula, new_self.children_list[1].formula)
            return new_self
        
        elif type(self) == nontemporal_multinary:
            # Define a new node
            new_self = nontemporal_multinary(None)
            new_self.operator = self.operator
            new_self.variables = self.variables

            # Push the negation
            for children in self.children_list:
                tmp_children = children.push_negation(neg=neg)
                new_self.children_list.append(tmp_children)

            # Modify the current node if negated
            new_self.formula = ""
            if neg:
                if self.operator == "|":
                    new_self.operator = "&"
                elif self.operator == "&":
                    new_self.operator = "|"

            # Find a new formula
            for children in new_self.children_list:
                new_self.formula += "({}) {} ".format(children.formula, new_self.operator)
            new_self.formula = new_self.formula[0:len(new_self.formula)-3]

            return new_self

        elif type(self) == temporal_unary:
            self.children_list[0] = self.children_list[0].push_negation(neg=neg)
            if neg:
                if self.operator == "G":
                    self.operator = "F"
                    self.formula = "F[{},{}] ({})".format(int(self.interval[0]), int(self.interval[1]), self.children_list[0].formula)
                elif self.operator == 'F':
                    self.operator = "G"
                    self.formula = "G[{},{}] ({})".format(int(self.interval[0]), int(self.interval[1]), self.children_list[0].formula)
            else:
                self.formula = "{}[{},{}] ({})".format(self.operator, int(self.interval[0]), int(self.interval[1]), self.children_list[0].formula)
            return self
        
        elif type(self) == temporal_binary:
            self.children_list[0] = self.children_list[0].push_negation(neg=neg)
            self.children_list[1] = self.children_list[1].push_negation(neg=neg)
            if neg:
                if self.operator == "U":
                    self.operator = "R"
                    self.formula = "({}) F[{},{}] ({})".format(self.children_list[0].formula, int(self.interval[0]), int(self.interval[1]), self.children_list[1].formula)
                    self.formula = self.formula.replace("U[", "R[")
                elif self.operator == "R":
                    self.operator = "U"
                    self.formula = "({}) U[{},{}] ({})".format(self.children_list[0].formula, int(self.interval[0]), int(self.interval[1]), self.children_list[1].formula)
                    self.formula = self.formula.replace("R[", "U[")
            return self
    
    def printInfo(self, layer=0):
        """ Print the AST. """
        print("    "*layer+ "{}".format(self.formula))
        if type(self) in (nontemporal_unary, temporal_unary):
            self.children_list[0].printInfo(layer=layer+1)
        elif type(self) in (nontemporal_binary, temporal_binary):
            self.children_list[0].printInfo(layer=layer+1)
            self.children_list[1].printInfo(layer=layer+1)
        elif type(self) == nontemporal_multinary:
            for children in self.children_list:
                children.printInfo(layer=layer+1)
        if layer == 0:
            print()

class term():
    __slots__ = ('multiplier', 'var', 'power')

    def __init__(self, data):
        """ Constructor method """
        if isinstance(data[0], float):
            self.multiplier = data[0]
            self.var = 1
        else:
            self.multiplier = 1
            self.var = data[0]

        if len(data) == 3:
            self.power = data[2]
        else:
            self.power = 1

    def __str__(self):
        res = ""
        res += "Multiplier: {}\n".format(self.multiplier)
        res += "Variable: {}\n".format(self.var)
        res += "Power: {}".format(self.power)
        return res

class multiterm():
    __slots__ = ('multiplier', 'var_list', 'power_list')

    def __init__(self, data):
        """ Constructor method """
        if len(data) == 3:
            self.multiplier = data[0].multiplier*data[2].multiplier
            self.var_list = deepcopy(data[2].var_list)
            self.power_list= deepcopy(data[2].power_list)
            if data[0].var != 1:
                self.var_list.append(data[0].var)
                self.power_list.append(data[0].power)
        else:
            self.multiplier = data[0].multiplier
            self.var_list = [data[0].var]
            self.power_list = [data[0].power]

    def __str__(self):
        res = ""
        res += "Multiplier: {}\n".format(self.multiplier)
        res += "Variables: {}\n".format(self.var_list)
        res += "Powers: {}".format(self.power_list)
        return res
        
class expression():
    __slots__ = ('variables', 'multipliers', 'var_list_list', 'power_list_list')

    def __init__(self, data):
        """ Constructor method """
        if len(data) == 5:
            self.multipliers = deepcopy(data[4].multipliers)
            if data[2] == '-':
                self.multipliers[-1] = -self.multipliers[-1]
            self.var_list_list = deepcopy(data[4].var_list_list)
            self.power_list_list = deepcopy(data[4].power_list_list)
            
            if data[0].var_list != ['']:
                self.multipliers.append(data[0].multiplier)
                self.var_list_list.append(data[0].var_list)
                self.power_list_list.append(data[0].power_list)
        else:
            self.multipliers = [data[0].multiplier]
            self.var_list_list = [data[0].var_list]
            self.power_list_list = [data[0].power_list]
        
        self.variables = set()
        for var_list in self.var_list_list:
            self.variables = self.variables.union(set(var_list))

        self.variables = list(self.variables)

    def __str__(self):
        res = ""
        res += "Variables: {}\n".format(self.variables)
        res += "Multiplier list: {}\n".format(self.multipliers)
        res += "Variables lists: {}\n".format(self.var_list_list)
        res += "Powers lists: {}".format(self.power_list_list)
        return res

class AP(ASTObject):
    __slots__ = ('variables', 'multipliers', 'var_list_list', 'power_list_list', 'equal')

    def __init__(self, data):
        """ Constructor method """
        super().__init__(data[1])
        data = data[0]
        self.multipliers = data[0].multipliers
        self.var_list_list = data[0].var_list_list
        self.power_list_list = data[0].power_list_list
        self.equal = False

        if data[2] in ("<=", "=<"):
            self.multipliers = self.multipliers + [-elem for elem in data[4].multipliers]
            self.var_list_list += data[4].var_list_list
            self.power_list_list += data[4].power_list_list
        elif data[2] == "<":
            self.multipliers = self.multipliers + [-elem for elem in data[4].multipliers]
            self.var_list_list += data[4].var_list_list
            self.power_list_list += data[4].power_list_list
            if [1] in self.var_list_list:
                idx = self.var_list_list.index([1])
                self.multipliers[idx] += EPS
            else:
                self.multipliers.append(EPS)
                self.var_list_list.append([1])
                self.power_list_list.append([1])
        elif data[2] in (">=", "=>"):
            self.multipliers = [-elem for elem in self.multipliers] + data[4].multipliers
            self.var_list_list += data[4].var_list_list
            self.power_list_list += data[4].power_list_list
        elif data[2] == ">":
            self.multipliers = [-elem for elem in self.multipliers] + data[4].multipliers
            self.var_list_list += data[4].var_list_list
            self.power_list_list += data[4].power_list_list
            if [1] in self.var_list_list:
                idx = self.var_list_list.index([1])
                self.multipliers[idx] += EPS
            else:
                self.multipliers.append(EPS)
                self.var_list_list.append([1])
                self.power_list_list.append([1])
        else:
            self.multipliers = self.multipliers + [-elem for elem in data[4].multipliers]
            self.var_list_list += data[4].var_list_list
            self.power_list_list += data[4].power_list_list
            self.equal = True
        
        self.variables = set()
        for var_list in self.var_list_list:
            self.variables = self.variables.union(set(var_list))

        self.variables = list(self.variables)

    def __str__(self):
        res = ""
        res += "Formula: {}\n".format(self.formula)
        res += "ID: {}\n".format(hex(id(self)))
        res += "Variables: {}\n".format(self.variables)
        res += "Multiplier list: {}\n".format(self.multipliers)
        res += "Variables lists: {}\n".format(self.var_list_list)
        res += "Powers lists: {}\n".format(self.power_list_list)
        res += "Equal?: {}\n".format(self.equal)
        return res

class stAP(ASTObject):
    __slots__ = ('variables', 'prob_multipliers', 'prob_var_list_list', 'prob_power_list_list', 'multipliers', 'var_list_list', 'power_list_list')

    def __init__(self, data):
        """ Constructor method """
        super().__init__(data[1])
        data = data[0]
        self.prob_multipliers = data[2].multipliers
        self.prob_var_list_list = data[2].var_list_list
        self.prob_power_list_list = data[2].power_list_list
        self.multipliers = data[8].multipliers
        self.var_list_list = data[8].var_list_list
        self.power_list_list = data[8].power_list_list

        self.variables = set()
        for var_list in self.var_list_list:
            self.variables = self.variables.union(set(var_list))

        self.variables = list(self.variables)

    def __str__(self):
        res = ""
        res += "Formula: {}\n".format(self.formula)
        res += "ID: {}\n".format(hex(id(self)))
        res += "Variables: {}\n".format(self.variables)
        res += "Prob multiplier list: {}\n".format(self.prob_multipliers)
        res += "Prob variables lists: {}\n".format(self.prob_var_list_list)
        res += "Prob powers lists: {}\n".format(self.prob_power_list_list)
        res += "Multiplier list: {}\n".format(self.multipliers)
        res += "Variables lists: {}\n".format(self.var_list_list)
        res += "Powers lists: {}\n".format(self.power_list_list)
        return res

class temporal_unary(ASTObject):
    __slots__ = ('operator', 'interval', 'variables', 'children_list')

    def __init__(self, data):
        super().__init__(data[1])
        data = data[0]
        """ Constructor method """
        self.operator = data[0]
        self.interval = data[2]
        self.interval = [int(entry) for entry in self.interval]
        self.children_list = data[6][0]
        
        self.variables = set()
        for children in self.children_list:
            self.variables = self.variables.union(set(children.variables))

        self.variables = list(self.variables)

    def __str__(self):
        res = ""
        res += "Formula: {}\n".format(self.formula)
        res += "ID: {}\n".format(hex(id(self)))
        res += "Variables: {}\n".format(self.variables)
        res += "Operator: {}\n".format(self.operator)
        res += "Interval: {}\n".format(self.interval)
        res += "Children: {}\n".format(self.children_list)
        return res

class temporal_binary(ASTObject):
    __slots__ = ('operator', 'interval', 'variables', 'children_list')

    def __init__(self, data):
        """ Constructor method """
        super().__init__(data[1])
        data = data[0]
        self.operator = data[6]
        self.interval = data[7]
        self.interval = [int(entry) for entry in self.interval]
        self.children_list = data[2][0] + data[11][0]
        
        self.variables = set()
        for children in self.children_list:
            self.variables = self.variables.union(set(children.variables))

        self.variables = list(self.variables)

    def __str__(self):
        res = ""
        res += "Formula: {}\n".format(self.formula)
        res += "ID: {}\n".format(hex(id(self)))
        res += "Variables: {}\n".format(self.variables)
        res += "Operator: {}\n".format(self.operator)
        res += "Interval: {}\n".format(self.interval)
        res += "Children: {}\n".format(self.children_list)
        return res

class nontemporal_unary(ASTObject):
    __slots__ = ('operator', 'variables', 'children_list')

    def __init__(self, data):
        """ Constructor method """
        super().__init__(data[1])
        data = data[0]
        self.operator = data[0]
        self.children_list = data[4][0]
        # for children in self.children_list:

        self.variables = set()
        for children in self.children_list:
            self.variables = self.variables.union(set(children.variables))

        self.variables = list(self.variables)

    def __str__(self):
        res = ""
        res += "Formula: {}\n".format(self.formula)
        res += "ID: {}\n".format(hex(id(self)))
        res += "Operator: {}\n".format(self.operator)
        res += "Children: {}\n".format(self.children_list)
        return res

class nontemporal_binary(ASTObject):
    __slots__ = ('operator', 'variables', 'children_list')

    def __init__(self, data):
        """ Constructor method """
        super().__init__(data[1])
        data = data[0]
        self.operator = data[6]
        self.children_list = data[2][0] + data[10][0]
        self.variables = set()
        for children in self.children_list:
            self.variables = self.variables.union(set(children.variables))

        self.variables = list(self.variables)

    def __str__(self):
        res = ""
        res += "Formula: {}\n".format(self.formula)
        res += "ID: {}\n".format(hex(id(self)))
        res += "Variables: {}\n".format(self.variables)
        res += "Operator: {}\n".format(self.operator)
        res += "Children: {}\n".format(self.children_list)
        return res

class nontemporal_multinary(ASTObject):
    __slots__ = ('operator', 'variables', 'children_list')

    def __init__(self, data):
        """ Constructor method """
        if data is None:
            self.operator = ""
            self.variables = []
            self.children_list = []
        else:
            super().__init__(data[1])
            data = data[0]
            self.operator = data[6]
            if isinstance(data[8], nontemporal_multinary):
                if self.operator == data[8].operator:
                    self.children_list = data[2][0] + data[8].children_list
                else:
                    raise ValueError("Cannot have a list of predicates connected via both AND and OR.")
            else:
                self.children_list = data[2][0] + data[10][0]
            self.variables = set()
            for children in self.children_list:
                self.variables = self.variables.union(set(children.variables))

            self.variables = list(self.variables)

    def __str__(self):
        res = ""
        res += "Formula: {}\n".format(self.formula)
        res += "ID: {}\n".format(hex(id(self)))
        res += "Variables: {}\n".format(self.variables)
        res += "Operator: {}\n".format(self.operator)
        res += "Children: {}\n".format(self.children_list)
        return res

class boolean(ASTObject):
    __slots__ = ('formula', 'variables')
    def __init__(self, data):
        """ Constructor method """
        super().__init__(data[1])
        self.variables = []

    def __str__(self):
        res = ""
        res += "Formula: {}\n".format(self.formula)
        res += "ID: {}\n".format(hex(id(self)))
        return res

# StSTL Grammar
ststl_grammar = Grammar('''
phi = true / false / nontemporal_unary / nontemporal_binary / nontemporal_multinary / temporal_unary / temporal_binary / stAP / AP

true = "True"
false = "False"

nontemporal_unary = nontemporal_unary_operator __ "(" __ phi __ ")"
nontemporal_binary = ("(" __ phi __ ")" __ nontemporal_binary_operator __ "(" __ phi __ ")")
nontemporal_multinary = ("(" __ phi __ ")" __ nontemporal_multinary_operator __ nontemporal_multinary) / ("(" __ phi __ ")" __ nontemporal_multinary_operator __ "(" __ phi __ ")")

nontemporal_unary_operator = "!" / "X"
nontemporal_binary_operator = "->"
nontemporal_multinary_operator = "and" / "&" / "or" / "|"

temporal_unary = temporal_unary_operator __ interval __ "(" __ phi __ ")"
temporal_binary = "(" __ phi __ ")" __ temporal_binary_operator interval __ "(" __ phi __ ")"

temporal_unary_operator = "G" / "F"
temporal_binary_operator = "U" / "R"

interval = "[" __ number __ "," __ number __ "]"

AP =  (expression __ comparison __ expression)
stAP =  "P[" __ expression __ "]" __ "(" __ AP __ ")" 

expression = (multiterm __ operator __ expression) / multiterm
multiterm = (term product multiterm) / term
term = number / variable_power / variable
variable_power = variable power number
variable = ~r"[a-z_\\d]"*
number = ~"[0-9.]+"

comparison = "<=" / ">=" / "=>" / "=<" / "<" / ">" / "=="
operator = "+" / "-"
product = "*"
power = "^"
__ = ~r"\\s"*
''')

class Parser(NodeVisitor):
    def __init__(self):
        super().__init__()

    def __call__(self, formula: str, rule: str = "phi"):
        return self.visit(ststl_grammar[rule].parse(formula))

    def visit_phi(self, node, children):
        # print(children[0])
        # input()
        return [children, node.text]

    def visit_true(self, node, children):
        return boolean([True, 'True'])

    def visit_false(self, node, children):
        return boolean([False, 'False'])

    def visit_nontemporal_unary(self, node, children):
        return nontemporal_unary([children, node.text])

    def visit_nontemporal_binary(self, node, children):
        return nontemporal_binary([children, node.text])

    def visit_nontemporal_multinary(self, node, children):
        return nontemporal_multinary([children[0], node.text])

    def visit_nontemporal_unary_operator(self, node, children):
        return node.text

    def visit_nontemporal_binary_operator(self, node, children):
        return node.text

    def visit_nontemporal_multinary_operator(self, node, children):
        return node.text

    def visit_temporal_unary(self, node, children):
        return temporal_unary([children, node.text])

    def visit_temporal_binary(self, node, children):
        return temporal_binary([children, node.text])

    def visit_temporal_unary_operator(self, node, children):
        return node.text

    def visit_temporal_binary_operator(self, node, children):
        return node.text

    def visit_interval(self, node, children):
        (_, _, left, _, _, _, right, _, _) = children
        return [left, right]

    def visit_AP(self, node, children):
        return AP([children, node.text])

    def visit_stAP(self, node, children):
        return stAP([children, node.text])

    def visit_expression(self, node, children):
        if isinstance(children[0], list):
            return expression(children[0])
        else:
            return expression(children)

    def visit_multiterm(self, node, children):
        if isinstance(children[0], list):
            return multiterm(children[0])
        else:
            return multiterm(children)

    def visit_term(self, node, children):
        if isinstance(children[0], list):
            return term(children[0])
        else:
            return term(children)

    def visit_variable(self, node, children):
        return node.text

    def visit_variable_power(self, node, children):
        return children

    def visit_number(self, node, children):
        return float(node.text)

    def visit_power(self, node, children):
        return node.text

    def visit_product(self, node, children):
        return node.text

    def visit_operator(self, node, children):
        return node.text

    def visit_comparison(self, node, children):
        return node.text

    def generic_visit(self, node, children):
        return children