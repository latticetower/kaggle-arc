import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from predictors.basic import Predictor


class FillerCFPredictor(Predictor):
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
        return True

    def train(self, iodata_list):
        pass

    def predict(self, complex_field):
        yield complex_field
