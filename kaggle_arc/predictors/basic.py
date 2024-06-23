import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpy as np
from itertools import islice
# from fractions import Fraction

from base.iodata import IOData
from base.field import Field
# from utils import check_if_can_be_mirrored
from operations.subpatterns import get_subpattern
from operations.subpatterns import check_subpattern
import predictors.availability_mixins as mixins

class Predictor:
    """Base class for all predictors.

    Methods
    -------
    train(iodata_list)
        Trains the given predictor with a list of IOData objects.
        Each object should have both input and output data for all samples.
    predict(field)
        For the input data stored in the variable `field`, tries to predict the output transformations.
    validate(iodata_list, k=3)
        For each of the inputs in iodata_list, predicts `k` outputs.
        After prediction, tries to compare them with the corresponding output and returns the final score.
    predict_on(cls, ds, k=3, args=[], kwargs=dict(), verbose=True, group_predictions=True)
        Utility method to process the dataset puzzles one by one.
    """

    def train(self, iodata_list):
        pass

    def predict(self, field):
        pass

    def validate(self, iodata_list, k=3):
        if isinstance(iodata_list, IOData):
            ps = islice(self.predict(iodata_list.input_field), k)
            # print(list(ps))
            scores = [Field.score(p, iodata_list.output_field) for p in ps]
            if len(scores) < 1:
                return 0.0
            return max(scores)

        scores = []
        for iodata in iodata_list:
            score = self.validate(iodata)
            scores.append(score)
        if len(scores) < 1:
            return 0.0
        # print(scores)
        return np.mean(scores)

    def freeze_by_score(self, iodata_list, k=3):
        pass

    @classmethod
    def predict_on(
        cls, ds, k=3, args=[], kwargs=dict(), verbose=False, group_predictions=True
    ):
        for sample in ds:
            predictor = cls(*args, **kwargs)
            # if not predictor.is_available(sample):
            predictor.train(sample.train)
            predictor.freeze_by_score(sample.train)

            score = predictor.validate(sample.train)
            if score == 1 and verbose:
                print(predictor)

            predictions = []
            for i, iodata in enumerate(sample.test):
                prediction = list(islice(predictor.predict(iodata), k))
                predictions.append(prediction)
                if not group_predictions:
                    yield sample.name, i, prediction
            if group_predictions:
                yield sample.name, predictions


class IdPredictor(Predictor, mixins.AvailableAll):

    def train(self, iodata_list):
        pass

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        # while True:
        yield Field(field.data)

    def __str__(self):
        return "IdPredictor()"


class ZerosPredictor(Predictor, mixins.AvailableAll):
    def __init(self):
        pass

    def train(self, iodata_list):
        pass

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
                return
        # while True:
        yield field.zeros()

    def __str__(self):
        return "ZerosPredictor()"


class ConstPredictor(Predictor, mixins.AvailableAll):
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
        # while True:
        yield field.consts(self.value, multiplier=self.multiplier)

    def __str__(self):
        return f"ConstPredictor(value={self.value}, multiplier={self.multiplier})"


class FillPredictor(Predictor, mixins.AvailableEqualShape):
    def __init__(self):
        self.common_patch = None

    def train(self, iodata_list):
        patches = []
        patch_sizes = set()

        for k, iodata in enumerate(iodata_list):
            i = iodata.input_field
            o = iodata.output_field
            (r0, c0) = get_subpattern(i.data, check_passed=False)
            (r1, c1) = get_subpattern(o.data, check_passed=False)
            # print(k, r0, c0, r1, c1)
            if check_subpattern(i.data, r1, c1):
                patch = self.get_patch(i.data, r1, c1, allow_zeros=True)
                # print(patch)
                # print(patch)
                patches.append(patch)
                patch_sizes.add((r1, c1))
                # print(r1,c1)
                # self.common_patch = patch
        if len(patch_sizes) == 1:
            self.common_patch = self.collect_patches(patches)

    def collect_patches(self, patches):
        # print(patches)
        common_patch = np.zeros(patches[0].shape, dtype=patches[0].dtype)
        for p in patches:
            for i in range(p.shape[0]):
                for j in range(p.shape[1]):
                    if common_patch[i, j] == 0:
                        common_patch[i, j] = p[i, j]
                    elif common_patch[i, j] != p[i, j]:
                        return None
        return common_patch

    def get_patch(self, data, r, c, allow_zeros=False):
        res = np.zeros((r, c), dtype=data.dtype)
        for i in range(r):
            for j in range(c):
                values = data[i::r, j::c]
                values = [v for v in np.unique(values) if v != 0]

                # if len(values) != 1:
                #        return None
                if len(values) == 1:
                    res[i, j] = values[0]
        return res

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        (r, c) = get_subpattern(field.data, wildcard=0)
        if self.common_patch is not None:
            patch = self.common_patch
        else:
            patch = self.get_patch(field.data, r, c, True)
        # print(patch)
        if patch is None or np.any(patch == 0):
            yield Field(field.data)
            return
        result = field.data.copy()
        coords = np.where(result == 0)
        for x, y in zip(*coords):
            result[x, y] = patch[x % r, y % c]
        yield Field(result)

    def __str__(self):
        return "FillPredictor()"
