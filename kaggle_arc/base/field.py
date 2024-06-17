import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
import numpy as np
import matplotlib
# try:
#     if not matplotlib.is_interactive():
#         matplotlib.use("svg")
# except:
#     pass

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from itertools import product
from collections import OrderedDict
import networkx as nx

from typing import NamedTuple

from constants import *
from base.utils import *


def binary_dice(a, b):
    s = (np.sum(a) + np.sum(b))
    if s != 0:
        return 2*np.sum(a*b)/s
    return None


def multiclass_dice(a, b, c):
    return binary_dice(1*(a == c), 1*(b==c))


def build_colormap(i, o=None, bg=0):
    colormap = {bg: 0}
    current_id = 1
    for line in i:
        for x in line:
            if x in colormap:
                continue
            colormap[x] = current_id
            current_id += 1
    if o is not None:
        for line in o:
            for x in line:
                if x in colormap:
                    continue
                colormap[x] = current_id
                current_id += 1
    return colormap


class Field:
    __slots__ = ["data", "multiplier", "colormap", "prop_names"]
    
    def __init__(self, data):
        if isinstance(data, list):
            self.data = np.asarray([[ (x if x >= 0 else 10 - x) for x in line ] for line in data], dtype=np.uint8)
        else:
            self.data = data.copy()
        self.multiplier = 0.5
        self.colormap = None
        self.prop_names = "h w xmin ymin xmax ymax xmean ymean is_convex holes contour_size interior_size".split() + \
            "is_rectangular is_square".split() + [f"flip_{i}" for i in range(10)]+[f"flip_conv_{i}" for i in range(10)]

    def get(self, i, j, default_color=0):
        if i < 0 or j < 0:
            return default_color
        if i >= self.data.shape[0] or j >= self.data.shape[1]:
            return default_color
        return self.data[i, j]

    @property
    def processed(self):
        if self.colormap is None:
            self.colormap = build_colormap(self.data, o=None, bg=0)
        new_data = [[self.colormap.get(x, x) for x in line] for line in self.data]
        return Field(new_data)

    def reconstruct(self, field):
        if self.colormap is None:
            return field
        rev = {v : k for k, v in self.colormap.items() }
        rev = {k : v for k, v in rev.items() if k != v }
        if len(rev) < 1:
            return field
        new_data = [[rev.get(x, x) for x in line] for line in field.data]
        return Field(new_data)
        
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
            #print(11)
        ax.imshow(self.data, cmap=COLORMAP, norm=NORM)
        # sns.cubehelix_palette(10, as_cmap=True),
        #ax.axis("off")
        for edge, spine in ax.spines.items():
            spine.set_visible(False)
        ax.set_xticks(np.arange(self.data.shape[1])+.5, minor=True)
        ax.set_yticks(np.arange(self.data.shape[0])+.5, minor=True)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_aspect("equal")
        #ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        if label is not None:
            ax.set_title(label)
        

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
        
    def build_nxgraph(self, connectivity={0: 4}, properties=None):
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
        
        regions0 = get_data_regions(self.data)
        params0, maps0 = get_region_params(regions0)
        
        regions1 = get_data_regions(self.data, connectivity=1)
        params1, maps1 = get_region_params(regions1, connectivity=1)
        
        
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                new_id = len(node_ids)
                node_ids[new_id] = (i, j)
                node_coords[(i, j)] = new_id
                
                color = self.data[i, j]
                
                #features = [
                left = i == 0  # left
                top = j == 0  # top
                right = i == self.data.shape[0] - 1 # right
                bottom = j == self.data.shape[1] - 1 # bottom
                features = [left, right, top, bottom]
                neighbours = [
                    (i1, j1)
                    for i1, j1 in product([i - 1, i, i + 1], [j - 1, j, j + 1])
                    if (i1 != i or j1 != j) and i1 >= 0 and j1 >= 0
                    and i1 < self.data.shape[0] and j1 < self.data.shape[1]
                ]
                if connectivity.get(color, 4) == 4:
                    neighbours = [(i1, j1) for i1, j1 in neighbours if (i1 == i or j1 == j)]
                # angle 90
                
                angle_props = []
                for d1, d2 in [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]:
                    angle_270 = False
                    left_shift = False
                    top_shift = False
                    if self.get(i + d1, j + d2) != color:
                        angle_270 = self.get(i, j + d2) == color and self.get(i + d1, j) == color
                        left_shift = self.get(i + d1, j) != color
                        top_shift = self.get(i, j+d2) != color
                    angle_props.extend([
                        angle_270,
                        left_shift, top_shift
                    ])
                features.extend(angle_props)
                #for c in self.get(i - 1, j), self.get(i, j - 1), self.get(i+1, j), self.get
                # for i1 in (i - 1, i + 1) if i1 >=0 and i1 < self.data.shape[0]
                # if not left
                ncolors = set([self.data[i1, j1] for i1, j1 in neighbours])
                ncolors = [(i in ncolors)*1 for i in range(10)]
                props = {
                    'features': np.asarray(features).astype(np.float),
                    'neighbours': neighbours,
                    'neighbour_colors': np.asarray(ncolors),
                    'color': self.data[i, j],
                    'x': all_features[:, i, j].astype(np.float64),
                    'pos': (i, j)
                }
                rid0 = regions0[i, j]
                p = [params0[rid0][k] for k in self.prop_names]
                rid1 = regions1[i, j]
                p+= [params1[rid1][k] for k in self.prop_names]
                
                if properties is not None:
                    props['properties'] = properties[i, j]
                props['component_params'] = np.asarray(p)
                
                graph_nx.add_node(new_id,
                    # features=np.asarray(features).astype(np.float),
                    # neighbours=neighbours,
                    # neighbour_colors = np.asarray(ncolors),
                    # color=self.data[i, j],
                    # x=all_features[:, i, j].astype(np.float64),
                    # pos=(i, j)
                    **props
                    )

        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                #neighbours = [
                #    (i1, j1)
                #    for i1, j1 in product([i - 1, i, i + 1], [j - 1, j, j + 1])
                #    if (i1 != i or j1 != j) and i1 >= 0 and j1 >= 0
                #    and i1 < self.data.shape[0] and j1 < self.data.shape[1]
                #]
                id0 = node_coords[(i, j)]
                color0 = self.data[i, j]
                
                neighbours = graph_nx.nodes[id0]['neighbours']
                
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


class ComplexField:
    def __init__(self, data, **params):
        self.data = data
        self.params = params
        self.multiplier = 0.5

    @property
    def shape(self):
        if len(self.data) > 0:
            if isinstance(self.data[0], list):
                return (len(self.data), len(self.data[0]))
        return (len(self.data),)
    
    @property
    def width(self):
        if len(self.shape) == 1:
            return 1
        return self.shape[1]

    @property
    def height(self):
        return self.shape[0]

    def flat_iter(self):
        for line in self.data:
            if isinstance(line, list):
                for x in line:
                    yield x
            else:
                yield line

    def map(self, func):
        new_data = [
            [ func(x) for x in line ]
            for line in self.data
        ]
        return ComplexField(new_data, **self.params)
        
    def show(self, ax=None, label=None):
        if ax is None:
            plt.figure(figsize=(self.width*self.multiplier, self.height*self.multiplier))
            ax = plt.gca()
        pass
    
    def __str__(self):
        return f"ComplexField({self.shape}, {self.params})"