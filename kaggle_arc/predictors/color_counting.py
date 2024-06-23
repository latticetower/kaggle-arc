"""
Predictor based on this notebook
https://www.kaggle.com/szabo7zoltan/colorandcountingmoduloq
"""

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpy as np

from base.iodata import IOData
from base.field import Field
from predictors.basic import Predictor
import predictors.availability_mixins as mixins

def get_p1_p2(i, j, n, k, v, q1, q2):
    if v == 0 or v == 2:
        p1 = i % q1
    else:
        p1 = (n - 1 - i) % q1
    if v == 0 or v == 3:
        p2 = j % q2
    else:
        p2 = (k - 1 - j) % q2
    return p1, p2


class ColorCountingPredictor(Predictor, mixins.AvailableEqualShape):
    def __init__(self):
        self.best_Dict = None
        self.best_Q1 = -1
        self.best_Q2 = -1
        self.best_v = -1

    def train(self, iodata_list):
        pairs = [
            (Q1, Q2)
            for t in range(15)
            for Q1 in range(1, 8)
            for Q2 in range(1, 8)
            if Q1 + Q2 == t
        ]
        h, w = list(zip(*[iodata.input_field.shape for iodata in iodata_list]))
        hmax = max(h)
        wmax = max(w)
        pairs = [(Q1, Q2) for Q1, Q2 in pairs if Q1 < hmax and Q2 < wmax]
        possible = True
        for Q1, Q2 in pairs:
            for v in range(4):
                if self.best_Dict is not None:
                    return
                possible = True
                Dict = {}
                for iodata in iodata_list:
                    (n, k) = iodata.input_field.shape
                    for i in range(n):
                        for j in range(k):
                            p1, p2 = get_p1_p2(i, j, n, k, v, Q1, Q2)
                            color1 = iodata.input_field.data[i, j]
                            color2 = iodata.output_field.data[i, j]
                            if color1 != color2:
                                rule = (p1, p2, color1)
                                if rule not in Dict:
                                    Dict[rule] = color2
                                elif Dict[rule] != color2:
                                    possible = False
                if not possible:
                    continue
                for iodata in iodata_list:
                    (n, k) = iodata.input_field.shape
                    for i in range(n):
                        for j in range(k):
                            p1, p2 = get_p1_p2(i, j, n, k, v, Q1, Q2)
                            color1 = iodata.input_field.data[i, j]
                            rule = (p1, p2, color1)
                            if rule in Dict:
                                color2 = 0 + Dict[rule]
                            else:
                                color2 = 0 + iodata.output_field.data[i, j]
                            if color2 != iodata.output_field.data[i, j]:
                                possible = False
                                break
                        if not possible:
                            break
                    if not possible:
                        break
                if possible:
                    self.best_Dict = Dict
                    self.best_Q1 = Q1
                    self.best_Q2 = Q2
                    self.best_v = v
                    return
        pass

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        # while True:
        if self.best_Dict is None:
            return

        n, k = field.shape
        answer = np.zeros(field.shape, dtype=field.dtype)
        for i in range(n):
            for j in range(k):
                p1, p2 = get_p1_p2(i, j, n, k, self.best_v, self.best_Q1, self.best_Q2)
                color1 = field.data[i, j]
                rule = (p1, p2, color1)
                answer[i, j] = self.best_Dict.get(rule, color1)
        yield Field(answer)
        # yield field.consts(self.value, multiplier=self.multiplier)

    def __str__(self):
        if self.best_Dict is None:
            return "ColorCountingPredictor(undefined)"
        return f"ColorCountingPredictor({self.best_Dict})"
