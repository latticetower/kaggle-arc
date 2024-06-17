import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from predictors.basic import Predictor


def to_tuple(field):
    return tuple([x for line in field.data for x in line])


class SorterCFPredictor(Predictor):
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
            if i.shape != o.shape:
                return False

            it = sorted([to_tuple(f) for f in i.flat_iter()])
            ot = sorted([to_tuple(f) for f in o.flat_iter()])
            if it != ot:
                return False
        return True

    def train(self, iodata_list):
        pass

    def predict(self, complex_field):
        yield complex_field
