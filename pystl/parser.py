import re

# Define global constants
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
		self.name = name
		self.class_id = class_id
	
	def __str__(self):
		return str(self.__dict__)

class Term(ASTObject):
	"""
	A class for defining a term objects.
	
	:param name: An unique name of an term object.
	:type name: str
	"""

	def __init__(self, name, str):
		""" Constructor method """
		super().__init__(name, 'TERM')
		self.multiplier = 1
		self.variable = None
		self.__add_var_n_mul(str)

	def __add_var_n_mul(self, str):
		""" Add a variable and a multiplier. """
		str_len = len(str)
		for idx, char in enumerate(str):
			if char.isalpha():
				if idx == 1:
					if str[0] == "-":
						self.multiplier = -1
					elif str[0] == "+":
						self.multiplier = 1
					else:
						self.multiplier = float(str[0])
				elif idx != 0:
					self.multiplier = float(str[0:idx])
				self.variable = str[idx:str_len]
				break
			if idx == str_len-1:
				self.multiplier = float(str)

class Expression(ASTObject):
	"""
	A class for defining an expression objects.
	
	:param name: An unique name of an expression object.
	:type name: str
	"""

	def __init__(self, name, str0):
		""" Constructor method """
		super().__init__(name, 'EXPR')
		self.terms = []
		self.__add_terms(str0)
		# print(self)

	def __add_terms(self, str0):
		""" Add terms. """
		num_term = 0
		prev = 0
		for idx, char in enumerate(str0):
			if char in ('+', '-') and idx != prev:
				num_term += 1
				self.terms.append(Term(self.name+'_'+str(num_term), str0[prev:idx]))
				prev = idx
			if idx == len(str0)-1:
				num_term += 1
				self.terms.append(Term(self.name+'_'+str(num_term), str0[prev:idx+1]))

class AtomicPredicate(ASTObject):
	"""
	A class for defining an atomic predicate objects.
	
	:param name: An unique name of an atomic predicate object.
	:type name: str
	"""

	def __init__(self, name, operator, start_time, end_time, expr_left, expr_right):
		""" Constructor method """
		super().__init__(name, 'AP')
		self.operator = operator
		self.start_time = start_time
		self.end_time = end_time
		self.expr_left = expr_left
		self.expr_right = expr_right

class StochasticAtomicPredicate(ASTObject):
	"""
	A class for defining a stochastic atomic predicate objects.
	
	:param name: An unique name of a stochastic atomic predicate object.
	:type name: str
	"""

	def __init__(self, name, start_time, end_time, ap, prob, negation):
		""" Constructor method """
		super().__init__(name, 'StAP')
		self.start_time = start_time
		self.end_time = end_time
		self.ap = ap
		self.prob = prob
		self.negation = negation

class Formula(ASTObject):
	"""
	A class for defining a formula objects.
	
	:param name: An unique name of a formula object.
	:type name: str
	"""

	def __init__(self, name, operator, start_time, end_time, formula_list):
		""" Constructor method """
		super().__init__(name, 'FORMULA')
		self.operator = operator
		self.start_time = start_time
		self.end_time = end_time
		self.formula_list = formula_list

# Define utility functions
def str2substr(str):
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

# Define ststl parser
def parse_ststl(name, logic_formula, logic_start_time = 0, logic_end_time = 0, negation = False):
	"""
	Parse a string in AST format representing a logic formula to a parse tree.

	:param logic_formula: A string in AST format representing a logic formula.
	:type logic_formula: str
	"""
	# Split the logic formula to sub-level logic formulas
	operator, start_time, end_time, prob, substr_list = str2substr(logic_formula)
	# print(str2substr(logic_formula))

	# Create appropriate object of each sub-level logic formulas
	if operator in ('=>', '<=', '>', '<', '=='): # Inequality and equality operator
		expr1 = Expression(name+'_1', substr_list[0])
		expr2 = Expression(name+'_2', substr_list[1])
		if negation:
			if operator == '=>':
				AP = AtomicPredicate(name, '<', logic_start_time, logic_end_time, expr1, expr2)
			elif operator == '<=':
				AP = AtomicPredicate(name, '>', logic_start_time, logic_end_time, expr1, expr2)
			elif operator == '>':
				AP = AtomicPredicate(name, '<=', logic_start_time, logic_end_time, expr1, expr2)
			elif operator == '<':
				AP = AtomicPredicate(name, '=>', logic_start_time, logic_end_time, expr1, expr2)
			else: # '=='
				AP = AtomicPredicate(name, '!=', logic_start_time, logic_end_time, expr1, expr2)
		else:
			AP = AtomicPredicate(name, operator, logic_start_time, logic_end_time, expr1, expr2)
		# print(AP)
		return AP, logic_start_time, logic_end_time

	elif operator == 'P': # Probability unary operator
		AP, t1, t2 = parse_ststl(name+'_1', substr_list[0], logic_start_time=logic_start_time, logic_end_time=logic_end_time, negation=False)
		stAP = StochasticAtomicPredicate(name, logic_start_time, logic_end_time, AP, prob, negation)
		# print(stAP)
		return stAP, t1, t2

	elif operator == '!': # Non-temporal unary operator
		if negation:
			formula, t1, t2 = parse_ststl(name, substr_list[0], logic_start_time=logic_start_time, logic_end_time=logic_end_time)
		else:
			formula, t1, t2 = parse_ststl(name, substr_list[0], logic_start_time=logic_start_time, logic_end_time=logic_end_time, negation=True)
		# print(formula)
		return formula, t1, t2

	elif operator == '->': # Non-temporal binary operator
		if negation:
			subformula1, t1_1, t2_1 = parse_ststl(name+'_1', substr_list[0], logic_start_time=logic_start_time, logic_end_time=logic_end_time, negation=False)
			subformula2, t1_2, t2_2 = parse_ststl(name+'_2', substr_list[1], logic_start_time=logic_start_time, logic_end_time=logic_end_time, negation=True)
			formula = Formula(name, '&', start_time, end_time, [subformula1, subformula2])
		else:
			subformula1, t1_1, t2_1 = parse_ststl(name+'_1', substr_list[0], logic_start_time=logic_start_time, logic_end_time=logic_end_time, negation=True)
			subformula2, t1_2, t2_2 = parse_ststl(name+'_2', substr_list[1], logic_start_time=logic_start_time, logic_end_time=logic_end_time, negation=False)
			formula = Formula(name, '|', start_time, end_time, [subformula1, subformula2])
		# print(formula)
		return formula, min(t1_1, t1_2), max(t2_1, t2_2)

	elif operator in ('&', '|'): # Non-temporal multinary operator
		subformula_list = []
		for i in range(len(substr_list)):
			tmp_subformula, t1, t2  = parse_ststl(name+'_'+str(i+1), substr_list[i], logic_start_time=logic_start_time, logic_end_time=logic_end_time, negation=negation)
			subformula_list.append(tmp_subformula)
		if negation:
			if operator == '&':
				formula = Formula(name, '|', start_time, end_time, subformula_list)
			else:
				formula = Formula(name, '&', start_time, end_time, subformula_list)
		else:
			formula = Formula(name, operator, start_time, end_time, subformula_list)
		# print(formula)
		return formula, t1, t2 

	elif operator in ('G', 'F'): # Temporal unary operators
		subformula, t1, t2 = parse_ststl(name+'_1', substr_list[0], logic_start_time=logic_start_time+start_time, logic_end_time=logic_end_time+end_time, negation=negation)
		if negation:
			if operator == 'G':
				formula = Formula(name, 'F', start_time, end_time, [subformula])
			else:
				formula = Formula(name, 'G', start_time, end_time, [subformula])
		else:
			formula = Formula(name, operator, start_time, end_time, [subformula])
		# print(formula)
		return formula, t1, t2 

	elif operator in ('U', 'R'): # Temporal binary operator
		subformula1, t1_1, t2_1  = parse_ststl(name+'_1', substr_list[0], logic_start_time=logic_start_time, logic_end_time=logic_end_time+end_time, negation=negation)
		subformula2, t1_2, t2_2  = parse_ststl(name+'_2', substr_list[1], logic_start_time=logic_start_time+start_time, logic_end_time=logic_end_time+end_time, negation=negation)
		if negation:
			if operator == 'U':
				formula = Formula(name, 'R', start_time, end_time, [subformula1, subformula2])
			else:
				formula = Formula(name, 'U', start_time, end_time, [subformula1, subformula2])
		else:
			formula = Formula(name, operator, start_time, end_time, [subformula1, subformula2])
		# print(formula)
		return formula, min(t1_1, t1_2), max(t2_1, t2_2)