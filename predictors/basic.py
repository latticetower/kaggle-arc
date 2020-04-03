import numpy as np

from base.iodata import IOData
from base.field import Field


class Predictor:
    def train(self, iodata):
        pass
    def predict(self, field):
        pass

    def predict_on(self, ds):
        for sample in ds:
            for i, iodata in enumerate(sample.iterate_test()):
                prediction = self.predict(iodata)
                yield sample.name, i, prediction


class IdPredictor(Predictor):
    def train(self, iodata_list):
        pass

    def predict(self, field):
        if isinstance(field, IOData):
            return self.predict(field.input_field)
        return Field(field.data)


class ZerosPredictor(Predictor):

    def train(self, iodata_list):
        pass

    def predict(self, field):
        if isinstance(field, IOData):
            return self.predict(field.input_field)
        return field.zeros()


class ConstPredictor(Predictor):
    def __init__(self, value=1, multiplier=1):
        self.value = value
        self.multiplier = multiplier

    def train(self, iodata_list):
        pass

    def predict(self, field):
        if isinstance(field, IOData):
            return self.predict(field.input_field)
        return field.consts(self.value, multiplier=self.multiplier)
