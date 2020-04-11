import numpy as np

from base.field import Field

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