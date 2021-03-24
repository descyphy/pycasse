from pystl.core import SMCSolver, MILPSolver
from copy import deepcopy
import numpy as np
import random
import matplotlib.pyplot as plt

M = 10**4
EPS = 10**-4

class contract:
	"""
	A contract class for defining a contract object.

	:param id: An id of the contract
	:type id: str
	"""

	__slots__ = ('id', 'controlled_vars', 'deter_uncontrolled_vars', 'nondeter_uncontrolled_vars', 'parameters', 'assumption', 'guarantee', 'sat_guarantee', 'isSat')

	def __init__(self, id = ''):
		""" Constructor method """
		self.id = id
		self.controlled_vars            = {'var_names': [], 'bounds': np.empty((0,2), int), 'vtypes': []}
		self.deter_uncontrolled_vars    = {'var_names': [], 'bounds': np.empty((0,2), int), 'vtypes': []}
		self.nondeter_uncontrolled_vars = {'var_names': [], 'mean': np.array([]), 'cov': np.array([]), 'dtype': 'GAUSSIAN'}
		self.parameters                 = {'param_names': [], 'bounds': np.empty((0,2), int), 'ptypes': []}
		self.assumption                 = 'True'
		self.guarantee                  = 'False'
		self.sat_guarantee              = 'False'
		self.isSat                      = False

	def set_controlled_vars(self, var_names, bounds = np.array([]), vtypes = []):
		""" 
		Adds controlled variables and their information to the contract 

		:param var_names: A list of names for controlled variables
		:type var_names: list
		:param bounds: An numpy array of lower and upper bounds for controlled variables, defaults to :math:`[-10^4,10^4]`
		:type bounds: :class:`numpy.ndarray`, optional
		:param vtypes: A list of variable types for controlled variables, each entry can be either "BINARY", "INTEGER", or "CONTINUOUS", defaults to "CONTINUOUS"
		:type vtypes: list, optional
		"""
		self.controlled_vars['var_names'] = self.controlled_vars['var_names'] + var_names
		var_len = len(var_names)
		
		if var_len > len(vtypes):
			self.controlled_vars['vtypes'] = self.controlled_vars['vtypes'] + var_len*['CONTINUOUS']
		else:
			self.controlled_vars['vtypes'] = self.controlled_vars['vtypes'] + vtypes
		
		if var_len > len(bounds):
			for vtype in self.controlled_vars['vtypes']:
				if vtype == 'CONTINUOUS' or vtype == 'INTEGER':
					self.controlled_vars['bounds'] = np.vstack(([self.controlled_vars['bounds'], [-M,M]]))
				else: 
					self.controlled_vars['bounds'] = np.vstack(([self.controlled_vars['bounds'], [0,1]]))
		else:
			self.controlled_vars['bounds'] = np.vstack(([self.controlled_vars['bounds'], bounds]))

	def set_deter_uncontrolled_vars(self, var_names, bounds = np.array([]), vtypes = []):
		"""
		Adds deterministic uncontrolled variables and their information to the contract

		:param var_names: A list of names for deterministic uncontrolled variables
		:type var_names: list
		:param bounds: An numpy array of lower and upper bounds for deterministic uncontrolled variables, defaults to :math:`[-10^4,10^4]`
		:type bounds: :class:`numpy.ndarray`, optional
		:param vtypes: A list of variable types for deterministic uncontrolled variables, each entry can be either "BINARY", "INTEGER", or "CONTINUOUS", defaults to "CONTINUOUS"
		:type vtypes: list, optional
		"""
		self.deter_uncontrolled_vars['var_names'] = self.deter_uncontrolled_vars['var_names'] + var_names
		var_len = len(var_names)
		
		if var_len > len(vtypes):
			self.deter_uncontrolled_vars['vtypes'] = self.deter_uncontrolled_vars['vtypes'] + var_len*['CONTINUOUS']
		else:
			self.deter_uncontrolled_vars['vtypes'] = self.deter_uncontrolled_vars['vtypes'] + vtypes

		if var_len > len(bounds):
			for vtype in self.deter_uncontrolled_vars['vtypes']:
				if vtype == 'CONTINUOUS' or vtype == 'INTEGER':
					self.deter_uncontrolled_vars['bounds'] = np.vstack(([self.deter_uncontrolled_vars['bounds'], [-M,M]]))
				else: 
					self.deter_uncontrolled_vars['bounds'] = np.vstack(([self.deter_uncontrolled_vars['bounds'], [0,1]]))
		else:
			self.deter_uncontrolled_vars['bounds'] = np.vstack(([self.deter_uncontrolled_vars['bounds'], bounds]))

	def set_nondeter_uncontrolled_vars(self, var_names, mean = np.array([]), cov = np.array([]), dtype = 'GAUSSIAN'):
		"""
		Adds uncontrolled variables and their information to the contract. We plan to add more distribution such as `UNIFORM`, `TRUNCATED_GAUSSIAN`, or even a distribution from `DATA`

		:param var_names: A list of names for uncontrolled variables
		:type var_names: list
		:param mean: A mean vector of uncontrolled variables
		:type mean: :class:`numpy.ndarray`
		:param cov: A covariance matrix of uncontrolled variables
		:type cov: :class:`numpy.ndarray`
		:param dtype: A distribution type for uncontrolled variables, can only be `GAUSSIAN` for now, defaults to `GAUSSIAN`
		:type dtype: str, optional
		"""
		self.nondeter_uncontrolled_vars['var_names'] = var_names
		self.nondeter_uncontrolled_vars['mean'] = mean
		self.nondeter_uncontrolled_vars['cov'] = cov

		if dtype != 'GAUSSIAN':
			raise ValueError('dtype should be GAUSSIAN for now.')
		else:
			self.nondeter_uncontrolled_vars['dtype'] = dtype

	def set_params(self, param_names, bounds = np.array([]), ptypes = []):
		"""
		Adds parameterss and their information to the contract.

		:param param_names: A list of names for parameters
		:type param_names: list
		:param bounds: An numpy array of lower and upper bounds for parameters, defaults to :math:`[-10^4,10^4]`
		:type bounds: :class:`numpy.ndarray`, optional
		:param ptypes: A list of variable types for parameters, each entry can be either "BINARY", "INTEGER", or "CONTINUOUS", defaults to "CONTINUOUS"
		:type ptypes: list, optional
		"""
		self.parameters['param_names'] = self.parameters['param_names'] + param_names
		param_len = len(param_names)
		
		if param_len > len(ptypes):
			self.parameters['ptypes'] = self.parameters['ptypes'] + param_len*['CONTINUOUS']
		else:
			self.parameters['ptypes'] = self.parameters['ptypes'] + ptypes

		if param_len > len(bounds):
			for vtype in self.parameters['ptypes']:
				if vtype == 'CONTINUOUS' or vtype == 'INTEGER':
					self.parameters['bounds'] = np.vstack(([self.parameters['bounds'], [-M,M]]))
				else: 
					self.parameters['bounds'] = np.vstack(([self.parameters['bounds'], [0,1]]))
		else:
			self.parameters['bounds'] = np.vstack(([self.parameters['bounds'], bounds]))

	def set_assume(self, assumption):
		"""
		Sets the assumption of the contract

		:param assumption: An STL or StSTL formula which characterizes the assumption set of the contract
		:type assumption: str
		"""
		self.assumption = assumption

	def set_guaran(self, guarantee):
		"""
		Sets the guarantee of the contract

		:param guarantee: An STL or StSTL formula which characterizes the guarantee set of the contract
		:type guarantee: str
		"""
		self.guarantee = guarantee

	def saturate(self):
		""" Saturates the contract """
		if self.isSat:
			return
		else:
			self.isSat = True
			if self.guarantee == 'True':
				self.sat_guarantee = 'True'
			elif self.assumption == 'False':
				self.sat_guarantee = 'True'
			elif self.assumption == 'True':
				self.sat_guarantee = self.guarantee
			elif self.guarantee == 'False':
				self.sat_guarantee = '(! ' + self.assumption + ')'
			else:
				self.sat_guarantee = '(-> ' + self.assumption + ' ' + self.guarantee + ')'

	def checkCompat(self, print_sol=False):
		""" Checks compatibility of the contract """
		# Build a MILP Solver
		solver = MILPSolver()
		print("Checking compatibility of the contract {}...".format(self.id))

		# Add a contract
		c = deepcopy(self)
		if self.assumption == 'True':
			print("Contract {} is compatible.\n".format(self.id))
			return
		elif self.assumption == 'False':
			print("Contract {} is not compatible.\n".format(self.id))
		else:
			c.set_guaran(self.assumption)
		c.set_assume('True')
		c.saturate()
		solver.add_contract(c)

		# Solve the problem 
		solved = solver.solve()

		# Print the solution
		if solved:
			print("Contract {} is compatible.\n".format(self.id))
			if print_sol:
				print("Printing a behavior that satisfies the assumption of the contract {}...".format(self.id))
				for v in solver.model.getVars():
					if 'node' not in v.varName:
						print("{} {}".format(v.varName, v.x))
				print("")
		else:
			print("Contract {} is not compatible.\n".format(self.id))

	def checkConsis(self, print_sol=False):
		""" Checks consistency of the contract """
		# Build a MILP Solver
		solver = MILPSolver()
		print("Checking consistency of the contract {}...".format(self.id))

		# Add a contract
		c = deepcopy(self)
		if self.sat_guarantee == 'True':
			print("Contract {} is compatible.\n".format(self.id))
			return
		elif self.sat_guarantee == 'False':
			print("Contract {} is not compatible.\n".format(self.id))
		else:
			c.set_guaran(self.sat_guarantee)
		c.set_assume('True')
		c.saturate()
		solver.add_contract(c)

		# Solve the problem 
		solved = solver.solve()

		# Print the solution
		if solved:
			print("Contract {} is consistent.\n".format(self.id))
			if print_sol:
				print("Printing a behavior that satisfies the saturated guarantee of the contract {}...".format(self.id))
				for v in solver.model.getVars():
					if 'node' not in v.varName:
						print("{} {}".format(v.varName, v.x))
				print("")
		else:
			print("Contract {} is not consistent.\n".format(self.id))

	def checkFeas(self, print_sol=False):
		""" Checks feasibility of the contract """
		# Build a MILP Solver
		solver = MILPSolver()
		print("Checking feasibility of the contract {}...".format(self.id))

		# Add a contract
		c = deepcopy(self)
		if self.assumption == 'False' or self.guarantee == 'False':
			print("Contract {} is not feasible.\n".format(self.id))
			return
		elif self.assumption == 'True':
			c.set_guaran(self.guarantee)
		elif self.guarantee == 'True':
			c.set_guaran(self.assumption)
		else:
			c.set_guaran("(& {} {})".format(self.assumption, self.guarantee))
		c.set_assume('True')
		c.saturate()
		solver.add_contract(c)

		# Solve the problem 
		solved = solver.solve()

		# Print the solution
		if solved:
			print("Contract {} is feasible.\n".format(self.id))
			if print_sol:
				print("Printing a behavior that satisfies both the assumption and guarantee of the contract {}...".format(self.id))
				for v in solver.model.getVars():
					if 'node' not in v.varName:
						print("{} {}".format(v.varName, v.x))
				print("")
		else:
			print("Contract {} is not feasible.\n".format(self.id))

	def checkRefine(self, contract2refine, print_sol=False):
		""" Checks whether contract2refine refines the contract """
		# Build a MILP Solver
		solver = MILPSolver()
		print("Checking whether contract {} refines contract {}...".format(self.id, contract2refine.id))

		# Add a contract
		c1 = contract('Condition1')
		c1.set_assume('True')
		c1.set_guaran("(! (-> {} {}))".format(contract2refine.assumption, self.assumption))
		merge_variables(c1, contract2refine, self)
		solver.add_contract(c1)

		# Solve the problem 
		print("Checking condition 1 for refinement...")
		solved = solver.solve()

		# Print the counterexample
		if solved:
			print("Condition 1 for refinement violated. Contract {} does not refine contract {}.\n".format(self.id, contract2refine.id))
			if print_sol:
				print("Printing a counterexample which violates condition 1 for refinement...")
				for v in solver.model.getVars():
					if 'node' not in v.varName:
						print("{} {}".format(v.varName, v.x))
				print("")
			return

		# Resets a MILP Solver
		solver.reset()

		# Add a contract
		c2 = contract('Condition2')
		c2.set_assume('True')
		c2.set_guaran("(! (-> {} {}))".format(self.sat_guarantee, contract2refine.sat_guarantee))
		merge_variables(c2, contract2refine, self)
		solver.add_contract(c2)

		# Solve the problem 
		print("Checking condition 2 for refinement...")
		solved = solver.solve()

		# Print the counterexample
		if solved:
			print("Condition 2 for refinement violated. Contract {} does not refine contract {}.\n".format(self.id, contract2refine.id))
			if print_sol:
				print("Printing a counterexample which violates condition 2 for refinement...")
				for v in solver.model.getVars():
					if 'node' not in v.varName:
						print("{} {}".format(v.varName, v.x))
				print("")
			return
		
		print("Contract {} refines {}.\n".format(self.id, contract2refine.id))

	def find_opt_param(self, objective, N=100):
		""" Find an optimal set of parameters for a contract given an objective function. """
		# Build a MILP Solver
		solver = MILPSolver()
		print("Finding an optimal set of parameters for contract {}...".format(self.id))
		
		# Build a deepcopy of the contract
		c = deepcopy(self)

		# Sample the parameters N times
		sampled_param = np.random.rand(N, len(self.parameters['param_names']))
		for i in range(len(self.parameters['param_names'])):
			sampled_param[:,1] *= self.parameters['bounds'][i,1] - self.parameters['bounds'][i,0]
			sampled_param[:,1] += self.parameters['bounds'][i,0]
		
		# Initialize the figure
		fig = plt.figure()
		plt.xlabel(self.parameters['param_names'][0])
		plt.ylabel(self.parameters['param_names'][1])
		plt.xlim([self.parameters['bounds'][0,0], self.parameters['bounds'][0,1]])
		plt.ylim([self.parameters['bounds'][1,0], self.parameters['bounds'][1,1]])
		
		# Replace the parameters with parameter samples and solve an optimization probelem
		for i in range(N):
			# Replace the parameters with parameter samples
			tmp_assumption = self.assumption
			tmp_guarantee = self.guarantee
			for j in range(len(self.parameters['param_names'])):
				tmp_assumption = tmp_assumption.replace(self.parameters['param_names'][j], str(sampled_param[i,j]))
				tmp_guarantee = tmp_guarantee.replace(self.parameters['param_names'][j], str(sampled_param[i,j]))
			
			# Resets a MILP Solver
			solver.reset()
			
			# Add a contract
			c.set_assume(tmp_assumption)
			c.set_guaran(tmp_guarantee)
			solver.add_contract(c)

			# Solve the problem 
			# print("Checking the set of parameters {}.".format(sampled_param[i,:]))
			solved = solver.solve(objective='min')
			
			# Print the counterexample
			if solved:
				plt.plot(sampled_param[i, 0], sampled_param[i, 1], 'go')
			else:
				plt.plot(sampled_param[i, 0], sampled_param[i, 1], 'ro')
		
		plt.savefig('test.jpg')

	def printInfo(self):
		""" Prints information of the contract """	
		print("Contract ID: " + self.id)
		if not not self.controlled_vars['var_names']:
			print("controlled Variables: " + str(list(self.controlled_vars['var_names'])))
		if not not self.deter_uncontrolled_vars['var_names']:
			print("Deterministic Uncontrolled Variables: " + str(list(self.deter_uncontrolled_vars['var_names'])))
		if not not self.nondeter_uncontrolled_vars['var_names']:
			print("Nondeterministic Uncontrolled Variables: " + str(list(self.nondeter_uncontrolled_vars['var_names'])))
		if not not self.parameters['param_names']:
			print("Parameters: " + str(list(self.parameters['param_names'])))
		print("Assumption: " + self.assumption)
		print("Guarantee: " + self.guarantee)
		print("Saturated Guarantee: " + self.sat_guarantee)
		print("isSat: " + str(self.isSat) + "\n")

def conjunction(c1, c2):
	""" Returns the conjunction of two contracts
	
	:param c1: A contract c1
	:type c1: :class:`pystl.contracts.contract.contract`
	:param c2: A contract c2
	:type c2: :class:`pystl.contracts.contract.contract`
	:return: A conjoined contract c1^c2
	:rtype: :class:`pystl.contracts.contract.contract`
	"""
	# Initialize a conjoined contract object
	conjoined = contract(c1.id + '^' + c2.id)

	# Merge controlled and uncontrolled variables
	merge_variables(conjoined, c1, c2)

	# Check saturation of c1 and c2, saturate them if not saturated
	if not c1.isSat:
		c1.saturate()
	if not c2.isSat:
		c2.saturate()

	# Find conjoined guarantee, G': G1 and G2
	if c1.sat_guarantee == 'False' or c2.sat_guarantee == 'False':
		conjoined.set_guaran('False')
	elif c1.sat_guarantee == 'True' and c2.sat_guarantee == 'True':
		conjoined.set_guaran('True')
	elif c1.sat_guarantee == 'True':
		conjoined.set_guaran(c2.sat_guarantee)
	elif c2.sat_guarantee == 'True':
		conjoined.set_guaran(c1.sat_guarantee)
	else:
		conjoined.set_guaran('(& {} {})'.format(c1.sat_guarantee, c2.sat_guarantee))

	# Find conjoined assumption, A': A1 or A2
	if c1.assumption == 'True' or c2.assumption == 'True':
		conjoined.set_assume('True')
	elif c1.assumption == 'False' and c2.assumption == 'False':
		conjoined.set_assume('False')
	elif c1.assumption == 'False':
		conjoined.set_assume(c2.assumption)
	elif c2.assumption == 'False':
		conjoined.set_assume(c1.assumption)
	else:
		conjoined.set_assume('(| {} {})'.format(c1.assumption, c2.assumption))

	conjoined.sat_guarantee = conjoined.guarantee
	conjoined.isSat = True
	return conjoined

def composition(c1, c2):
	""" Returns the composition of two contracts
	
	:param c1: A contract c1
	:type c1: :class:`pystl.contracts.contract.contract`
	:param c2: A contract c2
	:type c2: :class:`pystl.contracts.contract.contract`
	:return: A composed contract c1*c2
	:rtype: :class:`pystl.contracts.contract.contract`
	"""
	# Initialize a composed contract object
	composed = contract(c1.id + '*' + c2.id)

	# Merge controlled and uncontrolled variables
	merge_variables(composed, c1, c2)

	# Check saturation of c1 and c2, saturate them if not saturated
	if not c1.isSat:
		c1.saturate()
	if not c2.isSat:
		c2.saturate()

	# Find A': A1 and A2
	if c1.assumption == 'False' or c2.assumption == 'False':
		composed.set_assume('False')
	elif c1.assumption == 'True' and c2.assumption == 'True':
		composed.set_assume('True')
	elif c1.assumption == 'True':
		composed.set_assume(c2.assumption)
	elif c2.assumption == 'True':
		composed.set_assume(c1.assumption)
	else:
		composed.set_assume('(& {} {})'.format(c1.assumption, c2.assumption))

	# Find G': G1 and G2
	if c1.sat_guarantee == 'False' or c2.sat_guarantee == 'False':
		composed.set_guaran('False')
	elif c1.sat_guarantee == 'True' and c2.sat_guarantee == 'True':
		composed.set_guaran('True')
	elif c1.sat_guarantee == 'True':
		composed.set_guaran(c2.sat_guarantee)
	elif c2.sat_guarantee == 'True':
		composed.set_guaran(c1.sat_guarantee)
	else:
		composed.set_guaran('(& {} {})'.format(c1.sat_guarantee, c2.sat_guarantee))

	# Find composed assumption: A' or not G' 
	if composed.guarantee == 'False':
		composed.set_assume('True')
	elif composed.assumption == 'True':
		composed.set_assume('True')
	elif composed.assumption == 'False':
		composed.set_assume('(! {})'.format(composed.guarantee))
	elif composed.guarantee == 'True':
		composed.set_assume(composed.assumption)
	else:
		composed.set_assume('(| (! {}) {})'.format(composed.guarantee, composed.assumption))
	
	composed.sat_guarantee = composed.guarantee
	composed.isSat = True
	return composed

def merge(c1, c2):
	""" Returns the merge of two contracts
	
	:param c1: A contract c1
	:type c1: :class:`pystl.contracts.contract.contract`
	:param c2: A contract c2
	:type c2: :class:`pystl.contracts.contract.contract`
	:return: A merged contract c1+c2
	:rtype: :class:`pystl.contracts.contract.contract`
	"""
	# Initialize a merged contract object
	merged = contract(c1.id + '+' + c2.id)

	# Merge controlled and uncontrolled variables
	merge_variables(merged, c1, c2)

	# Check saturation of c1 and c2, saturate them if not saturated
	if not c1.isSat:
		c1.saturate()
	if not c2.isSat:
		c2.saturate()

	# Find assumption, A': A1 and A2
	if c1.assumption == 'False' or c2.assumption == 'False':
		merged.set_assume('False')
	elif c1.assumption == 'True' and c2.assumption == 'True':
		merged.set_assume('True')
	elif c1.assumption == 'True':
		merged.set_assume(c2.assumption)
	elif c2.assumption == 'True':
		merged.set_assume(c1.assumption)
	else:
		merged.set_assume('(& {} {})'.format(c1.assumption, c2.assumption))

	# Find guarantee, G': G1 and G2
	if c1.sat_guarantee == 'False' or c2.sat_guarantee == 'False':
		merged.set_guaran('False')
	elif c1.sat_guarantee == 'True' and c2.sat_guarantee == 'True':
		merged.set_guaran('True')
	elif c1.sat_guarantee == 'True':
		merged.set_guaran(c2.sat_guarantee)
	elif c2.sat_guarantee == 'True':
		merged.set_guaran(c1.sat_guarantee)
	else:
		merged.set_guaran('(& {} {})'.format(c1.sat_guarantee, c2.sat_guarantee))

	# Find merged guarantee, G' or not A'
	if merged.assumption == 'False':
		merged.set_guaran('True')
	elif merged.guarantee == 'True':
		merged.set_guaran('True')
	elif merged.guarantee == 'False':
		merged.set_guaran('(! {})'.format(merged.assumption))
	elif merged.assumption == 'True':
		merged.set_guaran(merged.guarantee)
	else:
		merged.set_guaran('(| (! {}) {})'.format(merged.assumption, merged.guarantee))

	merged.sat_guarantee = merged.guarantee
	merged.isSat = True
	return merged

def quotient(c, c2):
	""" Returns the quotient c/c2
	
	:param c: A contract c
	:type c: :class:`pystl.contracts.contract.contract`
	:param c2: A contract c2
	:type c2: :class:`pystl.contracts.contract.contract`
	:return: A quotient contract c/c2
	:rtype: :class:`pystl.contracts.contract.contract`
	"""

	# Initialize a quotient contract object
	quotient = contract(c.id + '/' + c2.id)

	# Merge controlled and uncontrolled variables
	merge_variables(quotient, c, c2)

	# Check saturation of c and c2, saturate them if not saturated
	if not c.isSat:
		c.saturate()
	if not c2.isSat:
		c2.saturate()

	# Find quotient assumption, A': A and G2
	if c.assumption == 'False' or c2.sat_guarantee == 'False':
		quotient.set_assume('False')
	elif c.assumption == 'True' and c2.sat_guarantee == 'True':
		quotient.set_assume('True')
	elif c.assumption == 'True':
		quotient.set_assume(c2.sat_guarantee)
	elif c2.sat_guarantee == 'True':
		quotient.set_assume(c.assumption)
	else:
		quotient.set_assume('(& {} {})'.format(c.assumption, c2.sat_guarantee))

	# Find quotient guarantee, G': G and A2
	if c.sat_guarantee == 'False' or c2.assumption == 'False':
		quotient.set_guaran('False')
	elif c.sat_guarantee == 'True' and c2.assumption == 'True':
		quotient.set_guaran('True')
	elif c.sat_guarantee == 'True':
		quotient.set_guaran(c2.assumption)
	elif c2.assumption == 'True':
		quotient.set_guaran(c.sat_guarantee)
	else:
		quotient.set_guaran('(& {} {})'.format(c.sat_guarantee, c2.assumption))

	# Find quotient guarantee, G' or not A'
	if quotient.assumption == 'False':
		quotient.set_guaran('True')
	elif quotient.guarantee == 'True':
		quotient.set_guaran('True')
	elif quotient.guarantee == 'False':
		quotient.set_guaran('(! {})'.format(quotient.assumption))
	elif quotient.assumption == 'True':
		quotient.set_guaran(quotient.guarantee)
	else:
		quotient.set_guaran('(| (! {}) {})'.format(quotient.guarantee, quotient.assumption))

	quotient.sat_guarantee = quotient.guarantee
	quotient.isSat = True
	return quotient

def separation(c, c2):
	""" Returns the separation c%c2
	
	:param c: A contract c
	:type c: :class:`pystl.contracts.contract.contract`
	:param c2: A contract c2
	:type c2: :class:`pystl.contracts.contract.contract`
	:return: A separated contract c%c2
	:rtype: :class:`pystl.contracts.contract.contract`
	"""

	# Initialize a quotient contract object
	separation = contract(c.id + '%' + c2.id)

	# Merge controlled and uncontrolled variables
	merge_variables(separation, c, c2)

	# Check saturation of c and c2, saturate them if not saturated
	if not c.isSat:
		c.saturate()
	if not c2.isSat:
		c2.saturate()

	# Find separation assumption, A': A and G2
	if c.assumption == 'False' or c2.sat_guarantee == 'False':
		separation.set_assume('False')
	elif c.assumption == 'True' and c2.sat_guarantee == 'True':
		separation.set_assume('True')
	elif c.assumption == 'True':
		separation.set_assume(c2.sat_guarantee)
	elif c2.sat_guarantee == 'True':
		separation.set_assume(c.assumption)
	else:
		separation.set_assume('(& {} {})'.format(c.assumption, c2.sat_guarantee))

	# Find separation guarantee, G': G and A2
	if c.sat_guarantee == 'False' or c2.assumption == 'False':
		separation.set_guaran('False')
	elif c.sat_guarantee == 'True' and c2.assumption == 'True':
		separation.set_guaran('True')
	elif c.sat_guarantee == 'True':
		separation.set_guaran(c2.assumption)
	elif c2.assumption == 'True':
		separation.set_guaran(c.sat_guarantee)
	else:
		separation.set_guaran('(& {} {})'.format(c.sat_guarantee, c2.assumption))

	# Find separation guarantee, G' or not A'
	if separation.assumption == 'False':
		separation.set_assume('True')
	elif separation.guarantee == 'True':
		separation.set_assume('True')
	elif separation.guarantee == 'False':
		separation.set_assume('(! {})'.format(separation.assumption))
	elif separation.assumption == 'True':
		separation.set_assume(separation.guarantee)
	else:
		separation.set_assume('(| (! {}) {})'.format(separation.assumption, separation.guarantee))
		
	separation.sat_guarantee = separation.guarantee
	separation.isSat = True
	return separation

def find_opt_param_refinement(c1, c2, objective):
	""" Find an optimal set of parameters for a refinement relationship. """

def merge_variables(c, c1, c2):
	""" Merges the controlled and uncontrolled variables on c1 and c2 """

	# Merge controlled_vars
	# Merge 'var_names'
	c.controlled_vars = c1.controlled_vars
	diff_controlled = set(c2.controlled_vars['var_names']) - set(c1.controlled_vars['var_names'])
	c.controlled_vars['var_names'] = c1.controlled_vars['var_names'] + list(diff_controlled) 

	# Merge 'vtypes' and 'bounds'
	for var in list(diff_controlled): 
		index = c2.controlled_vars['var_names'].index(var)
		temp = c2.controlled_vars['bounds'][index,:]
		if temp.ndim == 1:
			temp = [temp]

		# Merge 'vtypes' for a variable
		c.controlled_vars['vtypes'] = c1.controlled_vars['vtypes']
		c.controlled_vars['vtypes'] = c.controlled_vars['vtypes'] + [c2.controlled_vars['vtypes'][index]]

		# Merge 'bounds' for a variable
		c.controlled_vars['bounds'] = np.concatenate((c.controlled_vars['bounds'], temp), axis = 0)

	# Merge deter_uncontrolled_vars
	for var in set(c2.deter_uncontrolled_vars['var_names']).union(set(c1.deter_uncontrolled_vars['var_names'])):
		if var not in c.controlled_vars['var_names']:
			# Merge 'var_names'
			if not c.deter_uncontrolled_vars['var_names']:
				c.deter_uncontrolled_vars['var_names'] = [var]
			else:
				c.deter_uncontrolled_vars['var_names'] = c.deter_uncontrolled_vars['var_names'] + [var]
			
			if var in c2.deter_uncontrolled_vars['var_names']:
				index = c2.deter_uncontrolled_vars['var_names'].index(var)
				temp = c2.deter_uncontrolled_vars['bounds'][index,:]
				if temp.ndim == 1:
					temp = [temp]

				# Merge 'vtypes' for a variable
				c.deter_uncontrolled_vars['vtypes'] = c.deter_uncontrolled_vars['vtypes'] + [c2.deter_uncontrolled_vars['vtypes'][index]]

				# Merge 'bounds' for a variable
				c.deter_uncontrolled_vars['bounds'] = np.concatenate((c.deter_uncontrolled_vars['bounds'], temp), axis = 0)
			else:
				index = c1.deter_uncontrolled_vars['var_names'].index(var)
				temp = c1.deter_uncontrolled_vars['bounds'][index,:]
				if temp.ndim == 1:
					temp = [temp]

				# Merge 'vtypes' for a variable
				c.deter_uncontrolled_vars['vtypes'] = c.deter_uncontrolled_vars['vtypes'] + [c1.deter_uncontrolled_vars['vtypes'][index]]

				# Merge 'bounds' for a variable
				c.deter_uncontrolled_vars['bounds'] = np.concatenate((c.deter_uncontrolled_vars['bounds'], temp), axis = 0)

	# Merge nondeter_uncontrolled_vars
	if not c1.nondeter_uncontrolled_vars['var_names'] and not c2.nondeter_uncontrolled_vars['var_names']:
		# Merge 'var_names'
		c.nondeter_uncontrolled_vars = c1.nondeter_uncontrolled_vars
		diff_uncontrolled = set(c2.nondeter_uncontrolled_vars['var_names']) - set(c1.nondeter_uncontrolled_vars['var_names'])		
		c.nondeter_uncontrolled_vars['var_names'] = c1.nondeter_uncontrolled_vars['var_names'] + list(diff_uncontrolled)
		
		# Merge 'dtype'
		if c.nondeter_uncontrolled_vars['dtype'] == 'GAUSSIAN' and c2.nondeter_uncontrolled_vars['dtype'] == 'GAUSSIAN':
			c.nondeter_uncontrolled_vars['dtype'] == 'GAUSSIAN'
			# Merge 'bounds', 'mean', and 'cov'
			for var in list(diff_uncontrolled): 
				index = c2.nondeter_uncontrolled_vars['var_names'].index(var)
				if temp.ndim == 1:
					temp = [temp]

				# Merge 'mean'
				temp_mean = [c2.nondeter_uncontrolled_vars['mean'][index]]
				c.nondeter_uncontrolled_vars['mean'] = np.concatenate((c.nondeter_uncontrolled_vars['mean'], temp_mean), axis = 0)
				
				# Merge 'cov'
				var_len = len(c1.nondeter_uncontrolled_vars['var_names'])
				temp_cov1 = [[c2.nondeter_uncontrolled_vars['cov'][index,index]]]
				temp_cov2 = np.concatenate((np.zeros((var_len-1,1)), temp_cov1), axis = 0)
				c.nondeter_uncontrolled_vars['cov'] = np.concatenate((c.nondeter_uncontrolled_vars['cov'], np.zeros((1, var_len-1))), axis = 0)
				c.nondeter_uncontrolled_vars['cov'] = np.concatenate((c.nondeter_uncontrolled_vars['cov'], temp_cov2), axis = 1)
		else:
			c.nondeter_uncontrolled_vars['dtype'] == 'OTHERS'

	elif not c1.nondeter_uncontrolled_vars['var_names']:
		c.nondeter_uncontrolled_vars = c1.nondeter_uncontrolled_vars
	else:
		c.nondeter_uncontrolled_vars = c2.nondeter_uncontrolled_vars
