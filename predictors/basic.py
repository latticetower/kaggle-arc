import numpy as np

from base.iodata import IOData
from base.field import Field


class Predictor:
    def train(self, sample):
        pass
    def predict(self, field):
        pass
        
    @classmethod
    def predict_on(cls, ds, *args, **kwargs):
        for sample in ds:
            predictor = cls(*args, **kwargs)
            predictor.train(sample)
            for i, iodata in enumerate(sample.iterate_test()):
                prediction = predictor.predict(iodata)
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
