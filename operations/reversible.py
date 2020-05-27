import numpy as np

from base.field import *
from base.iodata import *
from operations.basic import candidate_functions

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
    splitters = np.ones(field.data.shape, dtype=field.dtype)
    for i in range(h):
        line = []
        hstart = outer_sep*hsep + i*(hpart + hsep)
        for j in range(w):
            wstart = outer_sep*wsep + j*(wpart + wsep)
            subfield = Field(
                field.data[
                    hstart : hstart + hpart, 
                    wstart : wstart + wpart])
            line.append(subfield)
            splitters[ hstart : hstart + hpart, wstart : wstart + wpart] = 0
        splitted.append(line)
    cf = ComplexField(splitted, separator=splitters*field.data, splitter=splitters)
    return cf


def split_by_shape(field, subshape, hsep=0, wsep=0, outer_sep=False):
    h, w = subshape
    fh, fw = field.shape
    #hpart = (fh - outer_sep*hsep) // h - hsep
    #wpart = (fw - outer_sep*wsep) // w - wsep

    splitted = []
    splitters = np.ones(field.data.shape, dtype=field.dtype)
    
    for i in range(outer_sep*1, fh, h + hsep):
        line = []
        sep_line = []
        for j in range(outer_sep*1, fw, w + wsep):
            subfield = Field(
                field.data[
                    i : i + h, 
                    j : j + w])
            line.append(subfield)
            splitters[i : i + h, j: j + w] = 0
            #sep_line.append(Field(field.data[i + h:i+h+hsep]))
        splitted.append(line)
    sep = splitters*field.data
    cf = ComplexField(splitted, separator=sep, splitter=splitters)
    return cf


def collect_field(multifield, hsep=0, wsep=0, outer_sep=False, sep_color=0):
    all_lines = []
    for line in multifield.data:
        line_data = []
        shape = list({x.shape for x in line})[0]
        sep = np.ones((shape[0], wsep), dtype=np.uint8)*sep_color
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
    def __init__(self, shape, hsep=0, wsep=0, outer_sep=False, sep_color=0,
            parent=None, splitter_func=split2shape):
        self.shape = shape
        self.hsep = hsep
        self.wsep = wsep
        self.outer_sep = outer_sep
        self.sep_color = sep_color
        parent=None,
        self.splitter_func = splitter_func

    def do(self, field):
        splitted = self.splitter_func(field, self.shape,
            hsep=self.hsep, wsep=self.wsep, outer_sep=self.outer_sep)
        return splitted

    def od(self, multifield):
        field = collect_field(multifield,
            hsep=self.hsep, wsep=self.wsep, outer_sep=self.outer_sep, sep_color=self.sep_color)
        return field

    def __str__(self):
        return f"ReversibleSplit({self.shape})"


class ReversibleCombine(ReversibleOperation):
    def __init__(self, shape, hsep=0, wsep=0, outer_sep=False, sep_color=0,
            parent=None, splitter_func=split2shape):
        self.shape = shape
        self.hsep = hsep
        self.wsep = wsep
        self.outer_sep = outer_sep
        self.sep_color = sep_color
        self.splitter_func = splitter_func
        self.color_func = None
        self.parent = None

    def train(self, io_list):
        #todo: correctly process case when there is no splitter
        get_color = lambda m: np.unique(m.params['separator'][np.where(m.params['splitter'])])[0]
        pairs = [(m, self.od(output_field)) for m, output_field in io_list]
        color_pairs = [(get_color(m), get_color(o)) for m, o in pairs]
        scores = []
        for func in candidate_functions:
            score = np.mean([func(i) == o for i, o in color_pairs])
            scores.append((-score, func))
        scores = sorted(scores, key=lambda x: x[0])
        score, score_func = scores[0]
        if score > -1:
            color_dict = dict(color_pairs)
            score_func = lambda x: color_dict.get(x, 0)
        self.color_func = score_func

    def do(self, multifield):
        if self.hsep > 0 or self.wsep > 0:
            colors = np.unique(multifield.params['separator'][np.where(multifield.params['splitter'])])
        else:
            colors = []
        sep_color = self.sep_color
        if len(colors) > 0 and self.color_func is not None:
            sep_color = self.color_func(colors[0])

        field = collect_field(multifield,
            hsep=self.hsep, wsep=self.wsep, outer_sep=self.outer_sep, sep_color=sep_color)
        return field
    
    def od(self, field):
        splitted = self.splitter_func(field, self.shape,
            hsep=self.hsep, wsep=self.wsep, outer_sep=self.outer_sep)
        return splitted
    
    def __str__(self):
        return f"ReversibleCombine({self.shape})"



class WrappedOperation:
    def __init__(self, preprocess=None, postprocess=None):
        self.preprocess = preprocess
        self.postprocess = postprocess
        
    def wrap(self, iodata):
        if isinstance(iodata, IOData):
            i = iodata.input_field
            o = iodata.output_field
        else:
            i, o = iodata
        forward_i = self.preprocess.do(i)
        if self.postprocess is None:
            reverse_o = o
        else:
            reverse_o = self.postprocess.od(o)
        return forward_i, reverse_o

    def train(self, iodata_list):
        #TODO: need to implement this
        data = [
            (self.preprocess.do(iodata.input_field), iodata.output_field)
            for iodata in iodata_list
        ]
        if self.postprocess is not None:
            self.postprocess.train(data)

    def run(self, field, prev=lambda x: x):
        x = self.preprocess.do(field)
        if self.postprocess is None:
            op = prev
        else:
            if prev is None:
                op = lambda t: self.postprocess.do(t)
            else:
                op = lambda t: prev(self.postprocess.do(t))
        return x, op


class WrappedOperationList:
    def __init__(self, operations):
        self.operations = operations
        pass

    def train(self, iodata_list):
        il = iodata_list
        for op in self.operations:
            op.train(il)
            il = [ op.wrap(io) for io in il ]
        pass
    def wrap(self, iodata):
        if isinstance(iodata, IOData):
            i = iodata.input_field
            o = iodata.output_field
            x = (i, o)
        else:
            x = iodata
        for op in self.operations:
            x = op.wrap(x)
        return x

    def run(self, field, prev=lambda x: x):
        x = field
        prev = None
        for op in operations:
            x, prev = op.run(x, prev)
        return x, prev
