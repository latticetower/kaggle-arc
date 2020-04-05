import numpy as np
from itertools import islice

from base.field import Field
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
        self.predictors = [
            p for p in self.predictors
            if p.is_available(iodata_list) ]
        for p in self.predictors:
            p.train(iodata_list)

    def validate(self, iodata_list, k=3):
        scores = []
        for iodata in iodata_list:
            pred_scores = []
            for res in islice(self.predict(iodata.input_field), k):
                score = Field.score(res, iodata.output_field)
                pred_scores.append(score)
            scores.append(max(pred_scores))
        # for p in self.predictors[:3]:
        #     score = p.validate(iodata_list)
        #     scores.append(score)
        if len(scores) == 0:
            return 0.0
        return np.mean(scores)
    
    def freeze_by_score(self, iodata_list, k=3):
        scores = []
        for p in self.predictors:
            score = p.validate(iodata_list, k=k)
            scores.append(score)
        ids = np.argsort(scores)[::-1]
        self.predictors = [ self.predictors[i] for i in ids ]

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
    def __str__(self):
        s = ";".join([ str(p) for p in self.predictors ])
        return f"ComplexPredictor({s})"