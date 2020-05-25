from base.field import *
from base.iodata import *
from predictors.basic import Predictor

class SelectorCFPredictor(Predictor):
    """Selects one of the patterns based on some features and returns as a result
    """
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
        self.iopairs=dict()
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
        for (i, o) in iodata_list:
            ifields = [ [ x for line in d.data for x in line ] for d in i.flat_iter() ]
            ifields = list(zip(*ifields))
            ofields = [x for line in o.data[0][0].data for x in line]
            for inp, out in zip(ifields, ofields):
                if inp in result:
                    if result[inp] != out:
                        raise Exception("incorrect solution")
                else:
                    result[inp] = out    
        self.iopairs = result

    def predict(self, complex_field):
        inp = [[[ x for x in line] for line in d.data ] for d in complex_field.flat_iter()]
        inp = list(zip(*inp))
        #print(self.iopairs)
        result = [[ self.iopairs.get(x, 0) for x in zip(*line)] for line in inp]
        cf = ComplexField([[
            Field(result)
        ]])
        yield cf