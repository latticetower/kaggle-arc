import numpy as np
from itertools import islice

from base.field import Field
from predictors.basic import Predictor
from predictors.basic import AvailableAll
from predictors.basic import IdPredictor, ZerosPredictor, ConstPredictor, FillPredictor

from predictors.color_counting import ColorCountingPredictor
from predictors.shapes import RepeatingPredictor, FractalPredictor, ResizingPredictor, MirrorPredictor, ConstantShaper
from predictors.boosting_tree import BoostingTreePredictor, BoostingTreePredictor2, BoostingTreePredictor3
from predictors.convolution import ConvolutionPredictor
from predictors.graph_boosting_tree import GraphBoostingTreePredictor, GraphBoostingTreePredictor2, GraphBoostingTreePredictor3
from predictors.decision_tree import AugmentedPredictor
from predictors.subpattern import SubpatternMatcherPredictor
from predictors.field2point import SimpleSummarizePredictor

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
        invalid_predictors = set()
        for i, p in enumerate(self.predictors):
            try:
                p.train(iodata_list)
            except Exception as e:
                print(e)
                invalid_predictors.add(i)
        self.predictors = [
            p for i, p in enumerate(self.predictors)
            if i not in invalid_predictors]

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
            score = 0
            try:
                p.freeze_by_score(iodata_list, k=k)
                score = p.validate(iodata_list, k=k)
            except:
                score = -1
            scores.append(score)
        scores = np.asarray(scores)
        #scores = scores[np.argwhere(scores>0)]
        ids = np.argsort(scores)[::-1]
        self.predictors = [ self.predictors[i] for i in ids if scores[i] >=0 ]

    def predict(self, field):
        for p in self.predictors:
            # if not p.is_available(sample):
            #     continue
            try:
                for v in p.predict(field):
                    yield v
            except Exception as e:
                print(e)
                pass
                #continue
        # for p in self.predictors:
        #     try:
        #         v = next(p.predict(field))
        #     except:
        #         continue
        #     yield v

    def __str__(self):
        s = ";".join([ str(p) for p in self.predictors ])
        return f"ComplexPredictor({s})"


class DefaultComplexPredictor(ComplexPredictor):
    def __init__(self):
        predictor_args = [
            IdPredictor,
            ZerosPredictor,
            ColorCountingPredictor,
            RepeatingPredictor,
            FractalPredictor,
            ResizingPredictor,
            GraphBoostingTreePredictor,#no impact
            GraphBoostingTreePredictor3,
            ConstantShaper,
            #BoostingTreePredictor,
            #BoostingTreePredictor2,
            BoostingTreePredictor3,
            SubpatternMatcherPredictor,
            #AugmentedPredictor
            FillPredictor,
            MirrorPredictor,
            SimpleSummarizePredictor,
            #(ConvolutionPredictor, [], {'loss': 'mse'}),
            #(ConvolutionPredictor, [], {'loss': 'dice'})
            ]
        super(DefaultComplexPredictor, self).__init__(predictor_args)