import numpy as np

from predictors.basic import Predictor
from predictors.basic import AvailableAll

class ComplexPredictor(Predictor, AvailableAll):
    def __init__(self, predictor_classes):
        self.predictors = []
        for data in predictor_classes:
            if isinstance(data, tuple):
                if len(data) == 3:
                    cls, args, kwargs = data
                else:
                    cls, args = data
                    kwargs = dict()
            else:
                cls = data
                args = []
                kwargs = dict()
            self.predictors.append(cls(*args, **kwargs))
        #self.classes = predictor_classes
        #pass

    def train(self, iodata_list):
        for p in self.predictors:
            p.train(iodata_list)

    def validate(self, iodata_list):
        scores = []
        for p in self.predictors:
            score = p.validate(iodata_list)
            scores.append(score)
        return np.mean(scores)
    
    def freeze_by_score(self, iodata_list, k=3):
        scores = []
        for p in self.predictors:
            score = p.validate(iodata_list, k=k)
            scores.append(score)
        ids = np.argsort(scores)[::-1]
        self.predictors = [ self.predictors[i] for i in ids]

    def predict(self, field):
        for p in self.predictors:
            # if not p.is_available(sample):
            #     continue
            for v in p.predict(field):
                yield v
        # for p in self.predictors:
        #     try:
        #         v = next(p.predict(field))
        #     except:
        #         continue
        #     yield v