import numpy as np

from base.iodata import IOData
from base.field import Field
from itertools import islice

class Predictor:
    # def is_available(sample):
    #     return True

    def train(self, iodata_list):
        pass

    def predict(self, field):
        pass

    def validate(self, iodata_list, k=3):
        if isinstance(iodata_list, IOData):
            ps = islice(self.predict(iodata_list.input_field), k)
            scores = [
                Field.score(p, iodata_list.output_field)
                for p in ps ]
            if len(scores) < 1:
                return 0.0
            return np.mean(scores)
        
        scores = []
        for iodata in iodata_list:
            score = self.validate(iodata)
            scores.append(score)
        if len(scores) < 1:
            return 0.0
        return np.mean(scores)

    def freeze_by_score(self, iodata_list, k=3):
        pass
    
    @classmethod
    def predict_on(cls, ds, k=3, args=[], kwargs=dict()):
        for sample in ds:
            predictor = cls(*args, **kwargs)
            #if not predictor.is_available(sample):
            predictor.train(sample.train)
            predictor.freeze_by_score(sample.train)
            
            for i, iodata in enumerate(sample.test):
                prediction = list(islice(predictor.predict(iodata), k))
                yield sample.name, i, prediction


class AvailableAll():
    def is_available(self, iodata_list):
        return True

class AvailableEqualShape():
    def is_available(self, iodata_list):
        for iodata in iodata_list:
            if iodata.input_field.shape != iodata.output_field.shape:
                return False
        return True

class AvailableWithMultiplier():
    def is_available(self, iodata_list):
        all_sizes = set()
        for iodata in iodata_list:
            m1 = iodata.output_field.height // iodata.input_field.height
            m2 = iodata.output_field.width // iodata.input_field.width
            all_sizes.append((m1, m2))
        if len(all_sizes) == 1:
            return True
        return False


class IdPredictor(Predictor, AvailableAll):
        
    def train(self, iodata_list):
        pass

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        #while True:
        yield Field(field.data)
    def __str__(self):
        return "IdPredictor()"


class ZerosPredictor(Predictor, AvailableAll):
    def __init(self):
        pass

    def train(self, iodata_list):
        pass

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
                return
        #while True:
        yield field.zeros()
        
    def __str__(self):
        return "ZerosPredictor()"


class ConstPredictor(Predictor, AvailableAll):
    def __init__(self, value=1, multiplier=1):
        self.value = value
        self.multiplier = multiplier

    def train(self, iodata_list):
        pass

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        #while True:
        yield field.consts(self.value, multiplier=self.multiplier)

    def __str__(self):
        return f"ConstPredictor(value={self.value}, multiplier={self.multiplier})"
