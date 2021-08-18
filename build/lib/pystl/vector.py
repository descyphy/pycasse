import numpy as np
import copy
from pystl.variable import Var

MAGIC_NUMBER = 15.0

class Vector:
    __array_priority__ = MAGIC_NUMBER

    def __init__(self, var_list, time = 0):
        """ Constructor. """
        self.max_time = time
        len_var = len([v.idx for v in var_list if isinstance(v, Var)])
        self.constant = np.zeros(len(var_list))
        self.data = np.zeros((len(var_list), len_var))
        self.var2id = {}
        for i, var in enumerate(var_list):
            assert(isinstance(var, (int, float, Var)))
            if isinstance(var, (int, float)):
                self.constant[i] += var
            elif isinstance(var, Var):
                key = (var.idx, time)
                if key not in self.var2id:
                    self.var2id[key] = len(self.var2id)
                self.data[i, self.var2id[key]] += 1
            else: assert(False)
    
    def __len__(self):
        assert(len(self.constant) == len(self.data))
        return len(self.constant)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return self._add(other)
        else:
            return self._merge(other, 1)

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return self._add(other)
        else:
            return self._merge(other, 1)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return self._add(-1 * other)
        else:
            return self._merge(other, -1)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return self._add(-1 * other)
        else:
            return self._merge(other, -1)
            
    def __eq__(self, other):
        if isinstance(other, list):
            other = Vector(other)
        return Equation(self - other)

    def __mul__(self, other):
        # TODO: address mul and rmul such that both mul and rmul work correctly
        res = copy.deepcopy(self)
        # print(res.data)
        # print(other)
        if isinstance(other, (int, float)):
            res.constant *= other
            res.data *= other
        else:
            res.constant = res.constant @ other
            res.data = res.data @ other
        # print(res)
        return res
    
    def __rmul__(self, other):
        res = copy.deepcopy(self)
        # print(other)
        # print(res.data)
        if isinstance(other, (int, float)):
            res.constant *= other
            res.data *= other
        else:
            res.constant = other @ res.constant
            res.data = other @ res.data
        # print(res)
        return res

    def __truediv__(self, other):
        res = copy.deepcopy(self)
        assert(isinstance(other, (int, float)))
        res.data /= other
        return res

    def __str__(self):
        return repr(self)

    def __repr__(self):
        data = []
        for row in range(len(self.data)):
            expr = ["{}".format(self.constant[row])]
            for key, idx in self.var2id.items():
                if self.data[row, idx] != 0:
                    expr.append("{} {}_{}".format(self.data[row, idx], key[0], key[1]))
            data.append(" + ".join(expr))
        res = "time max: {}\n".format(self.max_time)
        res += "[ " + "\n  ".join(data) + " ]\n"
        return res

    def _add(self, multiplier):
        res = copy.deepcopy(self)
        res.constant += multiplier
        return res

    def _merge(self, other, multiplier):
        assert(len(self) == len(other))
        res = copy.deepcopy(self)

        res.constant += other.constant

        for var, value in other.var2id.items():
            if var in res.var2id:
                res.data[:, res.var2id[var]] += multiplier * other.data[:, value]
            else:
                res.max_time = max(res.max_time, var[1])
                res.data = np.hstack((res.data, multiplier * other.data[:, value, np.newaxis]))
                res.var2id[var] = len(res.var2id)
        return res

class Equation:
    def __init__(self, vector):
        self.vector = vector
    def __repr__(self):
        return repr(self.vector)

def Next(other):
    res = Vector([])
    if other.data.shape[1] >= 1:
        res.max_time = other.max_time + 1
    else:
        res.max_time = other.max_time
    res.constant = other.constant.copy()
    res.data = other.data.copy()
    for key, value in other.var2id.items():
        res.var2id[(key[0], key[1] + 1)] = value
    return res
