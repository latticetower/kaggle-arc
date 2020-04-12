import numpy as np

from base.field import Field
from base.iodata import IOData
from base.transformers import resize_output

from predictors.basic import *
#Predictor, AvailableAll, AvailableWithIntMultiplier, AvailableMirror
from predictors.boosting_tree import BoostingTreePredictor

from operations.basic import Repaint
from operations.resizing import Repeater, Resizer, Fractal, Mirror
from utils import check_if_can_be_mirrored


class RepeatingPredictor(Predictor, AvailableWithIntMultiplier):
    def __init__(self, args=[], kwargs=dict()):
        #self.predictor = predictor_class(*args, **kwargs)
        pass

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
    def __init__(self, predictor=BoostingTreePredictor):
        self.predictor = predictor()

    def train(self, iodata_list):
        self.mirror = Mirror(self.m1, self.m2, vertical=self.vertical, horizontal=self.horizontal)
        self.predictor.train(resize_output(iodata_list))
        #train_ds[i].show(predictor=predictor)

    def freeze_by_score(self, iodata_list):
        self.predictor.freeze_by_score( resize_output(iodata_list))

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        repainter = Repaint(field.data)
        for prediction in self.predictor.predict(field):
            result = self.mirror(prediction.data)
            result = repainter(result)
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