import numpy as np

from base.field import Field

def func_identity(x): return x
def func_const_0(x): return 0
def func_const_1(x): return 1
def func_const_2(x): return 2
def func_const_3(x): return 3
def func_const_4(x): return 4
def func_const_5(x): return 5
def func_const_6(x): return 6
def func_const_7(x): return 7
def func_const_8(x): return 8
def func_const_9(x): return 9

candidate_functions = [
    func_identity,
    func_const_0,
    func_const_1,
    func_const_2,
    func_const_3,
    func_const_4,
    func_const_5,
    func_const_6,
    func_const_7,
    func_const_8,
    func_const_9
]

class Operation:
    def __call__(self, data):
        pass


class Replace(Operation):
    def __init__(self, replacements):
        # replacements is an array with 10 elements - some permutation of numbers 0..9
        self.replacements = replacements
        self.repl_func = np.vectorize(lambda x: self.replacements[x])

    def __call__(self, data):
        c = data.copy()
        c = self.repl_func(c)
        return c  # Field(c)
        
        
class Repaint(Operation):
    def __init__(self, input_data):
        data = np.unique(input_data, return_counts=True)
        s = sorted(zip(*data), key=lambda x: x[1], reverse=True)
        self.replacements = [x for x, y in s]

    def build_replacements_dict(self, data, filter_zero=False):
        replacements = self.replacements
        if filter_zero:
            replacements = [k for k in self.replacements if k != 0]
            data = [d for d in data if d != 0]
        repl_dict = dict(list(zip(data, replacements)))
        return repl_dict    

    def __call__(self, input_data):
        data = np.unique(input_data, return_counts=True)
        data = sorted(zip(*data), key=lambda x: x[1], reverse=True)
        data = [x for x, y in data]
        replacements = self.build_replacements_dict(data, filter_zero=True)
        #print(replacements)
        if len(replacements) < 1:
            return input_data
        repl_coords = {k: np.where(input_data==k) for k in replacements}
        result = input_data.copy()
        for k, (x, y) in repl_coords.items():
            c = replacements[k]
            result[x, y] = c
        return result