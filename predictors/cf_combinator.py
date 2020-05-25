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