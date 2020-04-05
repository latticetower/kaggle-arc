import os
import numpy as np
import matplotlib
try:
    matplotlib.use("svg")
except:
    pass
import matplotlib.pyplot as plt

def binary_dice(a, b):
    s = (np.sum(a) + np.sum(b))
    if s != 0:
        return 2*np.sum(a*b)/s
    return None


def multiclass_dice(a, b, c):
    return binary_dice(1*(a == c), 1*(b==c))


class Field:
    def __init__(self, data):
        self.data = np.asarray([[ x for x in line ] for line in data], dtype=np.uint8)
        self.multiplier = 0.5

    @property
    def height():
        return self.data.shape[0]

    @property
    def width():
        return self.data.shape[1]

    @property
    def shape():
        return self.data.shape

    @property
    def dtype():
        return self.data.dtype
    
    def __eq__(self, b):
        if not isinstance(b, Field):
            return self.data == b #this does all conversion magic
        if not (self.height == b.height and self.width==b.width):
            return False
        return np.all(self.data == b.data)

    def __ne__(self, b):
        #if not isinstance(b, Field):
        #    return self.data != b
        return ~(self==b)

    def __repr__(self):
        return repr(self.data)

    def show(self, ax=None, label=None):
        if ax is None:
            plt.figure(figsize=(self.width*self.multiplier, self.height*self.multiplier))
            ax = plt.gca()
        ax.imshow(self.data)
        if label is not None:
            ax.set_title(label)
        plt.axis("off")

    @staticmethod
    def compare_length(a, b):
        return a.width == b.width and a.height == b.height

    @staticmethod
    def dice(a, b):
        dist = [
            multiclass_dice(a, b, i)
            for i in range(10)]
        dist = [d for d in dist if d is not None]
        return np.mean(dist)

    @staticmethod
    def sized_dice(a, b):
        if Field.compare_length(a, b):
            return Field.dice(a, b)
        h = min(a.height, b.height)
        w = min(a.width, b.width)
        a_ = Field(a.data[:h, :w])
        b_ = Field(b.data[:h, :w])
        d = Field.dice(a_, b_)
        size_coef = 2 * w * h / (a.width * a.height + b.width * b.height)
        return size_coef * d

    @classmethod
    def distance(cls, a, b):
        return 1 - cls.dice(a, b)

    @classmethod
    def score(cls, a, b):
        return cls.sized_dice(a, b)

    def str_iter(self):
        yield "|"
        for line in self.data:
            for x in line:
                yield str(x)
            yield "|"

    def __repr__(self):
        return "".join(self.str_iter())

    def zeros(self, multiplier=1):
        return self.consts(value=0, multiplier=multiplier)

    def consts(self, value=1, multiplier=1):
        new_shape = tuple([ x * multiplier for x in self.data.shape])
        return Field(
            value*np.ones(new_shape, dtype=self.data.dtype))

    @staticmethod
    def fromstring(s):
        assert s[0] == "|"
        data = [
            [ int(x) for x in line ]
            for line in s[1:-1].split("|") ]
        return Field(data)
