import numpy as np

from predictors.basic import Predictor


class ComplexPredictor(Predictor):
    def __init__(self, predictor_classes):
        self.predictors = []
        for cls in predictor_classes:
            self.predictors.append(cls())
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
            yield p.predict(field)