import numpy as np

from base.field import *


class ReversibleOperation:
    def __init__(self):
        pass
    def do(self, field):
        pass
    def od(self, field):
        pass


def split2shape(field, target_shape, hsep=0, wsep=0, outer_sep=False):
    h, w = target_shape
    fh, fw = field.shape
    hpart = (fh - outer_sep*hsep) // h - hsep
    wpart = (fw - outer_sep*wsep) // w - wsep

    splitted = []
    
    for i in range(h):
        line = []
        for j in range(w):
            subfield = Field(
                field.data[
                    outer_sep*hsep + i*(hpart + hsep) : outer_sep*hsep + i*(hpart+hsep) + hpart, 
                    outer_sep*wsep + j*(wpart + wsep) : outer_sep*wsep + j*(wpart+wsep) + wpart])
            line.append(subfield)
        splitted.append(line)
    return splitted


def collect_field(multifield, hsep=0, wsep=0, outer_sep=False, sep_color=0):
    all_lines = []
    for line in multifield:
        line_data = []
        shape = list({x.shape for x in line})[0]
        sep = np.ones((shape[0], wsep))*sep_color
        if outer_sep and hsep > 0:
            line_data.append(sep)
        for x in line:
            line_data.append(x.data)
            if wsep > 0:
                line_data.append(sep)
        if not outer_sep and wsep > 0:
            line_data = line_data[:-1]
        line_data = np.concatenate(line_data, 1)  # np.concatenate([x.data for x in line], 1)
        all_lines.append(line_data)
    # collect all line parts
    shape = list({x.shape for x in all_lines})[0]
    sep = np.ones((hsep, shape[1]))*sep_color
    line_data = []
    if outer_sep:
        line_data.append(sep)
    for l in all_lines:
        line_data.append(l)
        if hsep > 0:
            line_data.append(sep)
    if not outer_sep and hsep > 0:
        line_data = line_data[:-1]
    all_lines = np.concatenate(line_data, 0)
    return Field(all_lines)


def increase2shape(data, target_shape):
    h, w = target_shape
    line = np.concatenate([data for i in range(w)], 1)
    d = np.concatenate([line for j in range(h)], 0)
    return d


def decrease2color(data, background=0):
    colors, counts = np.unique(data[np.where(data != background)], return_counts=True)
    if len(colors) < 1:
        return background
    return colors[0]


class ReversibleSplit(ReversibleOperation):
    def __init__(self, shape, hsep=0, wsep=0, outer_sep=False, sep_color=0):
        self.shape = shape
        self.hsep = hsep
        self.wsep = wsep
        self.outer_sep = outer_sep
        self.sep_color = sep_color

    def do(self, field):
        splitted = split2shape(field, self.shape,
            hsep=self.hsep, wsep=self.wsep, outer_sep=self.outer_sep)
        return splitted

    def od(self, multifield):
        field = collect_field(multifield,
            hsep=self.hsep, wsep=self.wsep, outer_sep=self.outer_sep, sep_color=self.sep_color)
        return field

    def __str__(self):
        return f"ReversibleSplit({self.shape})"


class ReversibleCombine(ReversibleOperation):
    def __init__(self, shape, hsep=0, wsep=0, outer_sep=False, sep_color=0):
        self.shape = shape
        self.hsep = hsep
        self.wsep = wsep
        self.outer_sep = outer_sep
        self.sep_color = sep_color
        
    def do(self, multifield):
        field = collect_field(multifield,
            hsep=self.hsep, wsep=self.wsep, outer_sep=self.outer_sep, sep_color=self.sep_color)
        return field
    
    def od(self, field):
        splitted = split2shape(field, self.shape,
            hsep=self.hsep, wsep=self.wsep, outer_sep=self.outer_sep)
        return splitted
    
    def __str__(self):
        return f"ReversibleCombine({self.shape})"



class WrappedOperation:
    def __init__(self, preprocess=None, postprocess=None):
        self.preprocess = preprocess
        self.postprocess = postprocess
        
    def wrap(self, iodata):
        i = iodata.input_field
        o = iodata.output_field
        forward_i = self.preprocess.do(i)
        if self.postprocess is None:
            reverse_o = o
        else:
            reverse_o = self.postprocess.od(o)
        return forward_i, reverse_o
    
    def run(self, field, prev=lambda x: x):
        x = self.preprocess.do(field)
        if self.postprocess is None:
            op = prev
        else:
            op = lambda t: prev(self.postprocess.do(t))
        return x, op

