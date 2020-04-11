import numpy as np

from base.field import Field
from base.iodata import IOData
from predictors.basic import Predictor, AvailableAll, AvailableWithIntMultiplier

from operations.resizing import Repeater, Resizer, Fractal, Mirror
from utils import check_if_can_be_mirrored

class RepeatingPredictor(Predictor, AvailableWithIntMultiplier):
    def __init__(self, predictor_class, args=[], kwargs=dict()):
        self.predictor = predictor_class(*args, **kwargs)

    def train(self, iodata_list):
        #self.predictor.train(iodata_list)
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


class MirrorPredictor(Predictor, AvailableMirror):
    def __init__(self, predictor):
        self.predictor = predictor

    def train(self, iodata_list):
        self.predictor.train(iodata_list)
        self.freeze_by_score(iodata_list)
        #train_ds[i].show(predictor=predictor)
        pass

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        mirror = Mirror(self.m1, self.m2, vertical=self.vertical, horizontal=self.horizontal)
        result = mirror(field.data)
        yield Field(result)

    def __str__(self):
        return f"ResizingPredictor(m1={self.m1}, m2={self.m2})"


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