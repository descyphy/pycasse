import numpy as np
import copy
from pystl.variable import Var

class Dynamics:
    def __init__(self, var_list, time = 0):
        self.max_time = time

        len_var = len(set([None] + [v.idx for v in var_list]))
        self.data = np.zeros((len(var_list), len_var))
        self.var2id = {(0,0): 0}
        for i, var in enumerate(var_list):
            assert(var is None, isinstance(var, (Var)))
            if var is not None:
                key = (var.idx, time)
                if key not in self.var2id:
                    self.var2id[key] = len(self.var2id)

                self.data[i, self.var2id[key]] += 1
        #  print(self.var2id)
        #  print(self.data)
    def __add__(self, other):
        if isinstance(other, (int, float)):
            return self.add(other)
        else:
            return self.merge(other, 1)
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return self.add(-1 * other)
        else:
            return self.merge(other, -1)
    def __eq__(self, other):
        return Equation(self.__sub__(other))
    def __mul__(self, other):
        #  print(self)
        #  print(other)
        #  input()
        res = copy.deepcopy(self)
        if isinstance(other, (int, float)):
            res.data *= other
        else:
            res.data = other @ res.data
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
            expr = []
            for key, idx in self.var2id.items():
                if self.data[row, idx] != 0:
                    if key[0] == 0:
                        expr.append("{}".format(self.data[row, idx]))
                    else:
                        expr.append("{} {}_{}".format(self.data[row, idx], key[0], key[1]))
            #  print(expr)
            data.append(" + ".join(expr))
        res = "time max: {}\n".format(self.max_time)
        res += "[ " + "\n  ".join(data) + " ]\n"
        return res
    def add(self, multiplier):
        res = copy.deepcopy(self)
        res.data[:, res.var2id[(0, 0)]] += multiplier
        return res
    def merge(self, other, multiplier):
        #  print(self.var2id)
        #  print(self.data)
        #  print(other.var2id)
        #  print(other.data)
        #  input()
        res = copy.deepcopy(self)
        for var, value in other.var2id.items():
            #  print(var, value)
            if var in res.var2id:
                res.data[:, res.var2id[var]] += multiplier * other.data[:, value]
            else:
                res.max_time = max(res.max_time, var[1])
                res.data = np.hstack((res.data, multiplier * other.data[:, value, np.newaxis]))
                res.var2id[var] = len(res.var2id)
        return res

class Equation:
    def __init__(self, dynamic):
        self.dynamic = dynamic
    def __repr__(self):
        return repr(self.dynamic)

def Next(other):
    res = Dynamics([])
    res.max_time = other.max_time

    res.data = other.data.copy()
    for key, value in other.var2id.items():
        if key[0] == 0:
            res.var2id[key] = value
        else:
            res.max_time = other.max_time + 1
            res.var2id[(key[0], key[1] + 1)] = value
    return res
