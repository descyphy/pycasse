import numpy as np
import pystl.parser

M = float(10**4)
EPS = float(10**-4)

class Var:
    __slots__ = ('name', 'idx')
    
    def __init__(self, name, idx):
        self.name = name
        self.idx = idx

    def __add__(self, other):
        return pystl.parser.Expression(self) + pystl.parser.Expression(other)

    def __radd__(self, other):
        return pystl.parser.Expression(self) + pystl.parser.Expression(other)

    def __sub__(self, other):
        return pystl.parser.Expression(self) - pystl.parser.Expression(other)

    def __rsub__(self, other):
        return pystl.parser.Expression(other) - pystl.parser.Expression(self)

    def __mul__(self, other):
        assert(isinstance(other, (int, float)))
        return pystl.parser.Expression(self) * other

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        assert(isinstance(other, (int, float)))
        return pystl.parser.Expression(self) / other

    def __pow__(self, other):
        return pystl.parser.Expression((self, other))

    def __lt__(self, other):
        return pystl.parser.Expression(self) < pystl.parser.Expression(other)

    def __le__(self, other):
        return pystl.parser.Expression(self) <= pystl.parser.Expression(other)

    def __gt__(self, other):
        return pystl.parser.Expression(self) > pystl.parser.Expression(other)

    def __ge__(self, other):
        return pystl.parser.Expression(self) >= pystl.parser.Expression(other)

    def __eq__(self, other):
        return pystl.parser.Expression(self) == pystl.parser.Expression(other)

class DeterVar(Var):
    """ 
    Constructs controlled variables/ uncontrolled variables/ parameters and their information

    :str name: A name for the variable
    :str var_type: A variable type for the variable, each entry can be either "controlled", "uncontrolled", "parameter"
    :str data_type: A data type for the variable, each entry can be either "BINARY", "INTEGER", or "CONTINUOUS", defaults to "CONTINUOUS"
    :np.array bounds: An numpy array of lower and upper bounds for the variable, defaults to `[-10^4,10^4]` for "CONTINUOUS" and "INTEGER" variable and `[0,1]` for "BINARY"
    """
    __slots__ = ('var_type', 'data_type', 'bound')
    
    def __init__(self, name, idx, var_type, data_type = 'CONTINUOUS', bound = None):
        """ Constructor. """
        # Assertions
        assert(var_type in ('controlled', 'uncontrolled', 'parameter'))
        assert(data_type in ('BINARY', 'INTEGER', 'CONTINUOUS'))

        # Inherit from Var class
        super().__init__(name, idx)
        
        # Set variable type and data_type
        self.var_type = var_type
        self.data_type = data_type

        # Set bounds
        if bound is None:
            if data_type in ('INTEGER', 'CONTINUOUS'):
                self.bound = np.array([-M, M])
            elif self.data_type == 'BINARY':
                self.bound = np.array([0,1])
            else: assert(False)
        else:
            assert(bound.shape == (2,))
            self.bound = bound
        self.bound = self.bound.astype(float)

    def __str__(self):
        """ Prints information of the contract """  
        res = "Deterministic Variable name: {}\n".format(self.name)
        res += "  var_type: {}, dtype: {}, bound: {}".format(self.var_type, self.data_type, self.bound)
        return res

    def __repr__(self):
        """ Prints information of the contract """  
        res = "Deterministic Variable name: {}\n".format(self.name)
        res += "  idx: {}, var_type: {}, dtype: {}, bound: {}".format(self.idx, self.var_type, self.data_type, self.bound)
        return res

class NondeterVar(Var):
    """ 
    Constructs nondeterministic uncontrolled variables and their information

    :str name: A name for the variable
    :str data_type: A data type for the variable, each entry can only be `GAUSSIAN` for now, defaults to `GAUSSIAN`
    """
    __slots__ = ('data_type')
    
    def __init__(self, name, idx, data_type = 'GAUSSIAN'):
        # Assertions
        assert(data_type in ('GAUSSIAN'))

        # Inherit from Var class
        super().__init__(name, idx)
        
        # Set data_type
        self.data_type = data_type

    def __str__(self):
        """ Prints information of the contract """  
        res = "Nondeterministic Variable name: {}\n".format(self.name)
        res += "  dtype: {}".format(self.data_type)
        return res

    def __repr__(self):
        """ Prints information of the contract """  
        res = "Nondeterministic Variable name: {}\n".format(self.name)
        res += "  idx: {}, dtype: {}".format(self.idx, self.data_type)
        return res
