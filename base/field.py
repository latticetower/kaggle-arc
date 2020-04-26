import os
import numpy as np
import matplotlib
try:
    if not matplotlib.is_interactive():
        matplotlib.use("svg")
except:
    pass

import matplotlib.pyplot as plt

import torch
from itertools import product
from collections import OrderedDict
import networkx as nx



def binary_dice(a, b):
    s = (np.sum(a) + np.sum(b))
    if s != 0:
        return 2*np.sum(a*b)/s
    return None


def multiclass_dice(a, b, c):
    return binary_dice(1*(a == c), 1*(b==c))


class Field:
    def __init__(self, data):
        self.data = np.asarray([[ (x if x >= 0 else 10 - x) for x in line ] for line in data], dtype=np.uint8)
        self.multiplier = 0.5
    def get(self, i, j, default_color=0):
        if i < 0 or j < 0:
            return default_color
        if i >= self.data.shape[0] or j >= self.data.shape[1]:
            return default_color
        return self.data[i, j]
        
    @property
    def height(self):
        return self.data.shape[0]

    @property
    def width(self):
        return self.data.shape[1]

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype
    @property
    def data_splitted(self):
        return np.stack([1.0*(self.data == i) for i in range(10)])

    def t(self):
        return torch.tensor(self.data)

    def t_splitted(self):
        return torch.tensor(self.data_splitted)

    @staticmethod
    def from_splitted(data):
        return Field(np.argmax(data, 0))

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
        
    def build_nxgraph(self, connectivity={0: 4}):
        def get_features(data):
            return np.stack([(data==i)*1.0 for i in range(10)], 0)

        graph_nx = nx.Graph()
        graph_nx.graph['global_features'] = np.asarray(
            [[
                (np.sum(self.data == i) > 0)*1.0
                for i in range(10)
            ]]).astype(np.float64)
        all_features = get_features(self.data)
        node_ids = OrderedDict() # node id -> (i, j) pair
        node_coords = OrderedDict() # node (i, j) pair -> id

        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                new_id = len(node_ids)
                node_ids[new_id] = (i, j)
                node_coords[(i, j)] = new_id
                graph_nx.add_node(new_id,
                    left=i==0,
                    top=j==0,
                    right=i==self.data.shape[0],
                    bottom=j==self.data.shape[1],
                    color=self.data[i, j],
                    x=all_features[:, i, j].astype(np.float64),
                    pos=(i, j))

        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                neighbours = [
                    (i1, j1)
                    for i1, j1 in product([i - 1, i, i + 1], [j - 1, j, j + 1])
                    if (i1 != i or j1 != j) and i1 >= 0 and j1 >= 0
                    and i1 < self.data.shape[0] and j1 < self.data.shape[1]
                ]
                id0 = node_coords[(i, j)]
                color0 = self.data[i, j]
                
                if connectivity.get(color0, 4) == 4:
                    neighbours = [(i1, j1) for i1, j1 in neighbours if (i1 == i or j1 == j)]
                
                for i1, j1 in neighbours:
                    id1 = node_coords[(i1, j1)]
                    color1 = self.data[i1, j1]
                    if color0 == color1:
                        graph_nx.add_edge(id0, id1,
                            features=np.asarray(
                                [
                                    (color0==x)*1.0
                                    for x in range(10)
                                ]).astype(np.float64))
                        #graph_nx.add_edge(id1, id0, features=[color0])

                #graph_nx.add_node()
        return graph_nx