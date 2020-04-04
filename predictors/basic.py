import numpy as np

from base.iodata import IOData
from base.field import Field
from itertools import islice

class Predictor:
    def train(self, iodata_list):
        pass
    def predict(self, field):
        pass

    def validate(self, iodata_list, k=3):
        if isinstance(iodata_list, IOData):
            ps = islice(self.predict(iodata_list.input_field), k)
            scores = [Field.score(p, iodata_list.output_field) for p in ps]
            return np.mean(score)
        
        scores = []
        for iodata in iodata_list:
            score = self.validate(iodata)
            scores.append(score)
        return np.mean(scores)

    def freeze_by_score(self, iodata_list, k=3):
        pass
    
    @classmethod
    def predict_on(cls, ds, k=3, args=[], kwargs=dict()):
        for sample in ds:
            predictor = cls(*args, **kwargs)
            predictor.train(sample.train)
            predictor.freeze_by_score(sample.train)
            
            for i, iodata in enumerate(sample.test):
                prediction = list(islice(predictor.predict(iodata), k))
                yield sample.name, i, prediction


class IdPredictor(Predictor):
    def train(self, iodata_list):
        pass

    def predict(self, field):
        if isinstance(field, IOData):
            return self.predict(field.input_field)
        #while True:
        yield Field(field.data)


class ZerosPredictor(Predictor):

    def train(self, iodata_list):
        pass

    def predict(self, field):
        if isinstance(field, IOData):
            return self.predict(field.input_field)
        #while True:
        yield field.zeros()


class ConstPredictor(Predictor):
    def __init__(self, value=1, multiplier=1):
        self.value = value
        self.multiplier = multiplier

    def train(self, iodata_list):
        pass

    def predict(self, field):
        if isinstance(field, IOData):
            return self.predict(field.input_field)
        #while True:
        yield field.consts(self.value, multiplier=self.multiplier)
