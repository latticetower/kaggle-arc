import numpy as np

from base.iodata import IOData
from base.field import Field
from itertools import islice
from utils import check_if_can_be_mirrored
from operations.subpatterns import get_subpattern
from operations.subpatterns import check_subpattern

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
            #print(list(ps))
            scores = [
                Field.score(p, iodata_list.output_field)
                for p in ps ]
            if len(scores) < 1:
                return 0.0
            return max(scores)
        
        scores = []
        for iodata in iodata_list:
            score = self.validate(iodata)
            scores.append(score)
        if len(scores) < 1:
            return 0.0
        #print(scores)
        return np.mean(scores)

    def freeze_by_score(self, iodata_list, k=3):
        pass
    
    @classmethod
    def predict_on(cls, ds, k=3, args=[], kwargs=dict(), verbose=True):
        for sample in ds:
            predictor = cls(*args, **kwargs)
            #if not predictor.is_available(sample):
            predictor.train(sample.train)
            predictor.freeze_by_score(sample.train)
            
            score = predictor.validate(sample.train)
            if score == 1:
                print(predictor)

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


class AvailableShape2Point():
    def is_available(self, iodata_list):
        for iodata in iodata_list:
            if iodata.output_field.shape != (1, 1):
                return False
        return True


class AvailableShape2PointOrConstColor():
    def is_available(self, iodata_list):
        for iodata in iodata_list:
            if iodata.output_field.shape != (1, 1):
                if len(np.unique(iodata.output_field.data)) != 1:
                    return False
        return True


class AvailableEqualShapeAndMaxNColors():
        
    def is_available(self, iodata_list):
        ncolors=4
        for iodata in iodata_list:
            if iodata.input_field.shape != iodata.output_field.shape:
                return False
            if len(np.unique(iodata.input_field.data)) > ncolors:
                return False
            if len(np.unique(iodata.output_field.data)) > ncolors:
                return False
        return True

class AvailableWithIntMultiplier():
    def is_available(self, iodata_list):
        all_sizes = set()
        for iodata in iodata_list:
            m1 = iodata.output_field.height // iodata.input_field.height
            m2 = iodata.output_field.width // iodata.input_field.width
            all_sizes.add((m1, m2))
        if len(all_sizes) == 1:
            h, w = all_sizes.pop()
            if w > 1 and h > 1:
                self.m1 = h
                self.m2 = w
                return True
        return False


class AvailableMirror(AvailableWithIntMultiplier):
    def is_available(self, iodata_list):
        availability_check = AvailableWithIntMultiplier()
        #print(isinstance(self, AvailableMirror))
        if not availability_check.is_available(iodata_list):
            #print(11)
            return False
        self.m1 = availability_check.m1
        self.m2 = availability_check.m2
        results = set()
        for iodata in iodata_list:
            h, w = iodata.input_field.shape
            res = check_if_can_be_mirrored(iodata.output_field.data, h=h, w=w)
            #print(res)
            if res is None:
                return False
            results.add(res)
        (vertical, horizontal) = results.pop()
        if len(results) > 0:
            return False
        self.vertical = vertical
        self.horizontal = horizontal
        return True


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


class FillPredictor(Predictor, AvailableEqualShape):
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
            #print(k, r0, c0, r1, c1)
            if check_subpattern(i.data, r1, c1):
                patch = self.get_patch(i.data, r1, c1, allow_zeros=True)
                #print(patch)
                #print(patch)
                patches.append(patch)
                patch_sizes.add((r1, c1))
                #print(r1,c1)
                #self.common_patch = patch
        if len(patch_sizes)==1:
            self.common_patch = self.collect_patches(patches)
        
    def collect_patches(self, patches):
        #print(patches)
        common_patch = np.zeros(patches[0].shape, dtype=patches[0].dtype)
        for p in patches:
            for i in range(p.shape[0]):
                for j in range(p.shape[1]):
                    if common_patch[i, j] == 0:
                        common_patch[i, j] = p[i, j]
                    elif common_patch[i, j]!= p[i, j]:
                        return None
        return common_patch

    def get_patch(self, data, r, c, allow_zeros=False):
        res = np.zeros((r, c), dtype=data.dtype)
        for i in range(r):
            for j in range(c):
                values = data[i::r, j::c]
                values = [v for v in np.unique(values) if v!=0]
                
                #if len(values) != 1:
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
        #print(patch)
        if patch is None or np.any(patch == 0):
            yield Field(field.data)
            return
        result = field.data.copy()
        coords = np.where(result == 0)
        for (x, y) in zip(*coords):
            result[x, y] = patch[x % r, y % c]
        yield Field(result)

    def __str__(self):
        return "FillPredictor()"
