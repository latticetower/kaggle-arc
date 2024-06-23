import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from base.field import *
from base.iodata import *
from predictors.basic import Predictor
from operations.reversible import *


class SelectorCFPredictor(Predictor):
    """Selects one of the patterns based on some features and returns as a result"""

    def __init__(self):
        pass

    def is_available(self, iodata_list):
        for iodata in iodata_list:
            if isinstance(iodata, IOData):
                return False
            i, o = iodata
            if not isinstance(i, ComplexField):
                return False
            if not isinstance(o, ComplexField):
                return False
            if o.shape != (1, 1):
                return False
        return True

    def train(self, iodata_list):
        pass

    def predict(self, complex_field):
        yield complex_field


class CombinatorCFPredictor(Predictor):
    def __init__(self):
        self.iopairs = dict()
        pass

    def is_available(self, iodata_list):
        for iodata in iodata_list:
            if isinstance(iodata, IOData):
                return False
            i, o = iodata
            if not isinstance(i, ComplexField):
                return False
            if not isinstance(o, ComplexField):
                return False
            if o.shape != (1, 1):
                return False
        return True

    def train(self, iodata_list):
        result = dict()
        for i, o in iodata_list:
            ifields = [[x for line in d.data for x in line] for d in i.flat_iter()]
            ifields = list(zip(*ifields))
            ofields = [x for line in o.data[0][0].data for x in line]
            for inp, out in zip(ifields, ofields):
                if inp in result:
                    if result[inp] != out:
                        continue
                        # raise Exception("incorrect solution")
                else:
                    result[inp] = out
        self.iopairs = result

    def predict(self, complex_field):
        inp = [
            [[x for x in line] for line in d.data] for d in complex_field.flat_iter()
        ]
        inp = list(zip(*inp))

        # print(self.iopairs)
        result = [[self.iopairs.get(x, 0) for x in zip(*line)] for line in inp]
        cf = ComplexField([[Field(result)]])
        yield cf


class WrappedCFPredictor(Predictor):
    def __init__(self):
        self.combinator = CombinatorCFPredictor()
        self.op = None

    def is_available(self, iodata_list):
        for iodata in iodata_list:
            i = iodata.input_field
            o = iodata.output_field
            (oh, ow) = o.shape
            (ih, iw) = i.shape
            if oh == 1 and ow == 1:
                return False
            if oh > ih or ow > iw:
                return False
            if oh == ih and ow == iw:
                return False
        hparts = 1
        wparts = 1
        all_parts = []
        for hsep in (0, 1, 2):
            for wsep in (0, 1, 2):
                for outer_sep in (True, False):
                    if hsep == 0 and wsep == 0 and outer_sep:
                        continue
                    res = []
                    for iodata in iodata_list:
                        if res is None:
                            break
                        i = iodata.input_field.data
                        o = iodata.output_field.data
                        (oh, ow) = o.shape
                        (ih, iw) = i.shape
                        if hsep > 0:
                            hvalues = set(i[: outer_sep * hsep].flatten())
                            for start in range(outer_sep * hsep + oh, ih, hsep + oh):
                                for x in np.unique(i[start : start + hsep]):
                                    hvalues.add(x)
                            if len(hvalues) > 1:
                                res = None
                                break
                            # if len(hvalues) > 1:
                            #    return False
                        if wsep > 0:
                            wvalues = set(i[:, : outer_sep * wsep].flatten())
                            for start in range(outer_sep * wsep + ow, iw, wsep + ow):
                                for x in np.unique(i[:, start : start + wsep]):
                                    wvalues.add(x)
                            if len(wvalues) > 1:
                                res = None
                                break
                                # return False
                        if outer_sep:
                            ih -= hsep
                            iw -= wsep
                        else:
                            ih += hsep
                            iw += wsep
                        h = ih // (oh + hsep)

                        if h * (oh + hsep) != ih or h < 1:
                            res = None
                            continue
                        # h -= hsep
                        w = iw // (ow + wsep)
                        if w * (ow + wsep) != iw or w < 1:
                            res = None
                            break
                        # print(h, w, ih, oh, iw, ow)
                        # w -= wsep
                        res.append((h, w))
                    if res is None:
                        continue
                    res = set(res)
                    if len(res) == 1:
                        all_parts.append([list(res)[0], hsep, wsep, outer_sep])
        if len(all_parts) < 1:
            return False
        if len(all_parts) > 1:
            return False
        (h, w), hsep, wsep, outer_sep = all_parts[0]
        self.shape = (h, w)
        self.hsep = hsep
        self.wsep = wsep
        self.outer_sep = outer_sep
        self.op = WrappedOperation(
            ReversibleSplit(
                (h, w), hsep=hsep, wsep=wsep, outer_sep=outer_sep
            ),  # , splitter_func=split_by_shape),
            ReversibleCombine(
                (1, 1), hsep=0, wsep=0, outer_sep=False, sep_color=0
            ),  # , splitter_func=split_by_shape)
        )
        data = [self.op.wrap(iodata) for iodata in iodata_list]

        return self.combinator.is_available(data)

    def train(self, iodata_list):
        data = [self.op.wrap(iodata) for iodata in iodata_list]
        self.combinator.train(data)

    def predict(self, field):
        field_inp, postprocess = self.op.run(field)
        for x in self.combinator.predict(field_inp):
            yield postprocess(x)
