"""
First we define methods for different field to color conversion operations. 
"""

import numpy as np

from skimage.measure import label

from operations.basic import Operation
from base.field import *
from base.iodata import *

class IrreversibleOperation:
    def __init__(self):
        pass
    def do(self):
        pass

def most_frequent_color(data, bg=None):
    #print("most frequent color")
    if bg is None:
        return np.argmax([np.sum(data == i) for i in range(10)])
    s = [np.sum(data == i) for i in range(10) if i != bg]
    if np.sum(s) > 0:
        return np.argmax(s)
    return bg

def least_frequent_color(data, bg=None):
    #print("least frequent color")
    if bg is None:
        s = {i: np.sum(data == i) for i in range(10)}
        s = [(k, v) for k, v in s.items() if v > 0]
        s = sorted(s, key=lambda x: x[1])
        if len(s) > 0:
            #print(s)
            return s[0][0]
        return 0
    s = {i: np.sum(data == i) for i in range(10) if i != bg}
    s = [(k, v) for k, v in s.items() if v > 0]
    s = sorted(s, key=lambda x: x[1])
    if len(s) > 0:
        return s[0][0]
    return bg

def count_color_area(data, bg=0):
    #print("color area")
    return np.max(label(data != bg))

def count_color_area_bg(data, bg=0):
    #print("color area bg")
    return np.max(label(data == bg))

def compute_color_gradient(data):
    cg0 = [np.sum(data[i]) for i in range(data.shape[0])]
    cg1 = [np.sum(data[:, i]) for i in range(data.shape[1])]
    return tuple(cg0), tuple(cg1)

def compute_weight_gradient(data, bg=0):
    return compute_color_gradient(data != bg)

def count_colors(data, bg=None):
    #print("count colors")
    #print(len(np.unique(data)))
    return len(np.unique(data))

def make_positional_color_selector(x, y):
    def pos_color_selector(data, bg=None):
        if x >= data.shape[0] or x < -data.shape[0]:
            x0 = x % data.shape[0]
        else:
            x0 = x
        if y >= data.shape[1] or y < -data.shape[1]:
            y0 = y % data.shape[1]
        else:
            y0 = y
        return data.data[x0, y0]
    return pos_color_selector


class SimpleSummarizeOperation(IrreversibleOperation):
    def __init__(self):
        self.bg = None
        self.func = None #lambda x, bg=0: x

    def train(self, iodata_list):
        if isinstance(iodata_list[0], IOData):
            iodata_list = [(x.input_field, x.output_field) for x in iodata_list]
        # elif isinstance(complex_iodata_list[0][0], ComplexField):
        #     complex_iodata_list = [
        #         (xi, xo) for i, o in complex_iodata_list
        #         for xi, xo in zip(i.flat_iter(), o.flat_iter())]
        candidates = [
            most_frequent_color,
            least_frequent_color,
            count_color_area,
            count_color_area_bg,
            count_colors
        ]
        
        h, w = list(zip(*[i.shape for i, o in iodata_list]))
        hmin = np.min(h)
        wmin = np.min(w)
        for i in range(hmin):
            for j in range(wmin):
                func = [
                    make_positional_color_selector(i, j),
                    make_positional_color_selector(i, -j),
                    make_positional_color_selector(-i, j),
                    make_positional_color_selector(-i, -j)
                ]
                candidates.extend(func)
        
        best_candidate = candidates[0]
        best_bg = dict()
        best_score = 0
        
        for candidate in candidates:
            best_bg[candidate] = None
            candidate_score = 0
            #candidate_bg = None
            scores = []
            best_sample_score = 0
            for bg in list(range(10)) + [ None ]:
                score = [
                    Field.score(
                        Field([[candidate(i.data, bg=bg)]]), o)
                    for i, o in iodata_list
                ]
                mean_score = np.mean(score)
                if mean_score > best_sample_score:
                    best_sample_score = mean_score
                    best_bg[candidate] = [bg]
            candidate_score = best_sample_score
            #print(candidate_score, best_bg)
            #print(candidate_score)
            #best_bg[candidate_score] = (candidatecandidate_bg
            if candidate_score > best_score:
                best_score = candidate_score
                best_candidate = candidate
            
        self.func = best_candidate
        self.bg = best_bg[best_candidate]
        #self.bg = [self.bg[k] for k in sorted(self.bg)]
        #print(self.bg)
        #most_frequent_color
        pass

    def do(self, field, bg=None):
        if len(self.bg) != 1:
            #print("use bg from param", self.bg, bg)
            pixel = self.func(field, bg=bg)
        else:
            #print(self.bg)
            pixel = self.func(field, bg=self.bg[0])
        #print(pixel, self.func)
        #print(np.asarray(pixel))
        return Field([[pixel]])


class ComplexSummarizeOperation(IrreversibleOperation):
    def __init__(self):
        self.bg = None
        self.func = None #lambda x, bg=0: x

    def train(self, complex_iodata_list):
        if isinstance(complex_iodata_list[0], IOData):
            complex_iodata_list = [(x.input_field, x.output_field) for x in complex_iodata_list]
        # elif isinstance(complex_iodata_list[0][0], ComplexField):
        #     complex_iodata_list = [
        #         (xi, xo) for i, o in complex_iodata_list
        #         for xi, xo in zip(i.flat_iter(), o.flat_iter())]
        candidates = [
            most_frequent_color,
            least_frequent_color,
            count_color_area,
            count_color_area_bg,
            count_colors
        ]
        best_candidate = candidates[0]
        best_bg = dict()
        best_score = 0
        
        for candidate in candidates:
            best_bg[candidate] = dict()
            candidate_score = 0
            #candidate_bg = None
            scores = []
            for k, (inp, out) in enumerate(complex_iodata_list):
                
                iodata_list = list(zip(inp.flat_iter(), out.flat_iter()))
                #iodata_list = list(zip([x for xs in inp for x in xs], [x for xs in out for x in xs]))
                best_sample_score = 0
                for bg in list(range(10)) + [ None ]:
                    score = [
                        Field.score(
                        Field([[candidate(i.data, bg=bg)]]), o)
                        for i, o in iodata_list
                    ]
                    mean_score = np.mean(score)
                    if mean_score > best_sample_score:
                        best_sample_score = mean_score
                        best_bg[candidate][k] = bg
                scores.append(best_sample_score)
            candidate_score = np.mean(scores)
            #print(candidate_score, best_bg)
            #print(candidate_score)
            #best_bg[candidate_score] = (candidatecandidate_bg
            if candidate_score > best_score:
                best_score = candidate_score
                best_candidate = candidate
                
        self.func = best_candidate
        self.bg = best_bg[best_candidate]
        self.bg = [self.bg[k] for k in sorted(self.bg)]
        bg = set()
        for k in self.bg:
            bg.add(k)
        self.bg = list(bg)
        #print(self.bg)
        #most_frequent_color
        pass

    def do(self, field, bg=None):
        if len(self.bg) != 1:
            #print("use bg from param", self.bg, bg)
            pixel = self.func(field, bg=bg)
        else:
            #print(self.bg)
            pixel = self.func(field, bg=self.bg[0])
        #print(pixel, self.func)
        #print(np.asarray(pixel))
        return Field([[pixel]])
