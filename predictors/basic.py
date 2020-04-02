import numpy as np

from base.iodata import IOData
from base.field import Field


class Predictor:
    def train(self, iodata):
        pass
    def predict(self, field):
        pass


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
        return Field(np.zeros((field.height, field.width)))

class OnesPredictor(Predictor):

    def train(self, iodata_list):
        pass

    def predict(self, field):
        if isinstance(field, IOData):
            return self.predict(field.input_field)
        return Field(np.ones((field.height, field.width)))
