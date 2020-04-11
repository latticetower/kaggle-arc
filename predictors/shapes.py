import numpy as np

from base.field import Field
from base.iodata import IOData
from predictors.basic import Predictor, AvailableAll, AvailableWithIntMultiplier

from operations.resizing import Repeater, Resizer, Fractal


class RepeatingPredictor(Predictor, AvailableWithIntMultiplier):
    def __init__(self):
        pass

    def train(self, iodata_list):
        pass

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        repeater = Repeater(self.m1, self.m2)
        result = repeater(field.data)
        yield Field(result)

    def __str__(self):
        return f"RepeatingPredictor(m1={self.m1}, m2={self.m2})"



class ResizingPredictor(Predictor, AvailableWithIntMultiplier):
    def __init__(self):
        pass

    def train(self, iodata_list):
        pass

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        resizer = Resizer(self.m1, self.m2)
        result = resizer(field.data)
        yield Field(result)

    def __str__(self):
        return f"ResizingPredictor(m1={self.m1}, m2={self.m2})"


class FractalPredictor(Predictor, AvailableWithIntMultiplier):
    def __init__(self):
        pass

    def train(self, iodata_list):
        pass

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        fractal = Fractal(self.m1, self.m2)
        result = fractal(field.data)
        yield Field(result)

    def __str__(self):
        return f"FractalPredictor(m1={self.m1}, m2={self.m2})"