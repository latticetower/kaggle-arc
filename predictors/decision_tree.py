"""Code in next predictor is based on this kernel

https://www.kaggle.com/adityaork/decision-tree-smart-data-augmentation/comments
"""

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from pathlib import Path
from collections import defaultdict
from itertools import product
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations,permutations
from sklearn.tree import *
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
import random
from math import floor


from predictors.basic import *

class Augmenter:
    __slots__ = ()
    
    @staticmethod
    def getiorc(iodata):
        inp = iodata.input_field
        return iodata.input_field, iodata.output_field, inp.shape[0], inp.shape[1]
    
    @classmethod
    def getBkgColor(cls, iodata_list):
        color_dict = defaultdict(int)

        for iodata in iodata_list:
            inp, oup, r, c = cls.getiorc(iodata)
            for i in range(r):
                for j in range(c):
                    color_dict[inp.data[i, j]] += 1
        color = -1
        max_count = 0
        for col, cnt in color_dict.items():
            if(cnt > max_count):
                color = col
                max_count = cnt
        return color

    @classmethod
    def get_bl_cols(cls, iodata_list):
        result = []
        bkg_col = cls.getBkgColor(iodata_list)
        result.append(bkg_col)
        # num_input,input_cnt,num_output,output_cnt
        met_map = {}
        for i in range(10):
            met_map[i] = [0,0,0,0]

        total_ex = 0
        for iodata in iodata_list:
            inp, oup = iodata.input_field, iodata.output_field
            u, uc = np.unique(inp.data, return_counts=True)
            inp_cnt_map = dict(zip(u, uc))
            u, uc = np.unique(oup.data, return_counts=True)
            oup_cnt_map = dict(zip(u, uc))

            for col,cnt in inp_cnt_map.items():
                met_map[col][0] = met_map[col][0] + 1
                met_map[col][1] = met_map[col][1] + cnt
            for col, cnt in oup_cnt_map.items():
                met_map[col][2] = met_map[col][2] + 1
                met_map[col][3] = met_map[col][3] + cnt
            total_ex += 1

        for col, met in met_map.items():
            num_input, input_cnt, num_output, output_cnt = met
            if(num_input == total_ex or num_output == total_ex):
                result.append(col)
            elif(num_input == 0 and num_output > 0):
                result.append(col)

        result = np.unique(result).tolist()
        if(len(result) == 10):
            result.append(bkg_col)
        return np.unique(result).tolist()

    @staticmethod
    def getAround(i, j, inp, size=1):
        #v = [-1,-1,-1,-1,-1,-1,-1,-1,-1]
        r, c = inp.shape
        v = []
        sc = [0]
        for q in range(size):
            sc.append(q + 1)
            sc.append(-(q+1))
        for idx, (x,y) in enumerate(product(sc, sc)):
            ii = (i+x)
            jj = (j+y)
            v.append(-1)
            if((0<= ii < r) and (0<= jj < c)):
                v[idx] = (inp.data[ii, jj])
        return v
    
    @classmethod
    def getX(cls, inp, i, j, size):
        n_inp = inp.data
        z = [i, j]
        r, c = inp.shape
        for m in range(5):
            z.append(i % (m + 1))
            z.append(j % (m + 1))
        z.append(i + j)
        z.append(i * j)
        #     z.append(i%j)
        #     z.append(j%i)
        z.append((i+1)/(j+1))
        z.append((j+1)/(i+1))
        z.append(r)
        z.append(c)
        z.append(len(np.unique(n_inp[i,:])))
        z.append(len(np.unique(n_inp[:,j])))
        arnd = cls.getAround(i,j,inp,size)
        z.append(len(np.unique(arnd)))
        z.extend(arnd)
        return z

    @classmethod
    def getXy(cls, inp, oup, size):
        x = []
        y = []
        r, c = inp.shape
        for i in range(r):
            for j in range(c):
                #print(inp)
                x.append(cls.getX(inp, i, j, size))
                y.append(oup.data[i][j])
        return x,y
    
    @staticmethod
    def replace(inp, uni, perm):
        # uni = '234' perm = ['5','7','9']
        #print(uni,perm)
        #print(uni, perm)
        r_map = { int(c):int(s) for c,s in zip(uni,perm)}
        r,c = inp.shape
        rp = inp.data.tolist()
        #print(rp)
        for i in range(r):
            for j in range(c):
                if(rp[i][j] in r_map):
                    rp[i][j] = r_map[rp[i][j]]
        return Field(rp)
    
    @classmethod
    def augment(cls, inp, oup, bl_cols):
        cols = "0123456789"
        npr_map = [1,9,72,3024,15120,60480,181440,362880,362880]
        uni = "".join([str(x) for x in np.unique(inp.data).tolist()])
        for c in bl_cols:
            cols = cols.replace(str(c),"")
            uni = uni.replace(str(c),"")

        exp_size = inp.shape[0]*inp.shape[1]*npr_map[len(uni)]

        mod = floor(exp_size/120000)
        mod = 1 if mod == 0 else mod

        #print(exp_size,mod,len(uni))
        result = []
        count = 0
        for comb in combinations(cols,len(uni)):
            for perm in permutations(comb):
                count += 1
                if(count % mod == 0):
                    #print(uni)
                    result.append(
                        (cls.replace(inp, uni, perm),
                         cls.replace(oup, uni, perm)))
        return result
    
    @staticmethod
    def get_flips(i, o):
        result = []
        #inp = input_field.data
        #oup = output_field.data
        operations = [
            lambda inp: np.fliplr(inp),
            lambda inp: np.rot90(np.fliplr(inp), 1),
            lambda inp: np.rot90(np.fliplr(inp), 2),
            lambda inp: np.rot90(np.fliplr(inp), 3),
            lambda inp: np.flipud(inp),
            lambda inp: np.rot90(np.flipud(inp), 1),
            lambda inp: np.rot90(np.flipud(inp), 2),
            lambda inp: np.rot90(np.flipud(inp), 3),
            lambda inp: np.fliplr(np.flipud(inp)),
            lambda inp: np.flipud(np.fliplr(inp))
        ]
        for op in operations:
            yield Field(op(i.data)), Field(op(o.data))
        #result.append((np.fliplr(inp).tolist(),np.fliplr(oup).tolist()))
        #result.append((np.rot90(np.fliplr(inp),1).tolist(),np.rot90(np.fliplr(oup),1).tolist()))
        #result.append((np.rot90(np.fliplr(inp),2).tolist(),np.rot90(np.fliplr(oup),2).tolist()))
        #result.append((np.rot90(np.fliplr(inp),3).tolist(),np.rot90(np.fliplr(oup),3).tolist()))
        #result.append((np.flipud(inp).tolist(),np.flipud(oup).tolist()))
        #result.append((np.rot90(np.flipud(inp),1).tolist(),np.rot90(np.flipud(oup),1).tolist()))
        #result.append((np.rot90(np.flipud(inp),2).tolist(),np.rot90(np.flipud(oup),2).tolist()))
        #result.append((np.rot90(np.flipud(inp),3).tolist(),np.rot90(np.flipud(oup),3).tolist()))
        #result.append((np.fliplr(np.flipud(inp)).tolist(),np.fliplr(np.flipud(oup)).tolist()))
        #result.append((np.flipud(np.fliplr(inp)).tolist(),np.flipud(np.fliplr(oup)).tolist()))
        #return result
    
    @classmethod
    def gettaskxy(cls, iodata_list, aug, around_size, bl_cols, flip=True):    
        X = []
        Y = []
        for iodata in iodata_list:
            inp, oup = iodata.input_field, iodata.output_field
            tx,ty = cls.getXy(inp, oup, around_size)
            X.extend(tx)
            Y.extend(ty)
            if(flip):
                for ainp, aoup in cls.get_flips(inp, oup):
                    tx,ty = cls.getXy(ainp, aoup, around_size)
                    X.extend(tx)
                    Y.extend(ty)
                    if(aug):
                        augs = cls.augment(ainp, aoup, bl_cols)
                        for ainp, aoup in augs:
                            #print("1", ainp)
                            tx, ty = cls.getXy(ainp, aoup, around_size)
                            X.extend(tx)
                            Y.extend(ty)
            if (aug):
                augs = cls.augment(inp, oup, bl_cols)
                for ainp, aoup in augs:
                    #print("2", ainp)
                    tx,ty = cls.getXy(ainp, aoup, around_size)
                    X.extend(tx)
                    Y.extend(ty)
        return X,Y


class AugmentedPredictor(Predictor, AvailableEqualShape):
    def __init__(self):
        #self.value = value
        #self.multiplier = multiplier
        pass
    
    def predict_on_tree_model(self, inp, model, size):
        r, c = inp.shape
        oup = np.zeros(inp.shape, dtype=int)
        for i in range(r):
            for j in range(c):
                x = Augmenter.getX(inp, i, j, size)
                o = int(model.predict([x]))
                o = 0 if o < 0 else o
                oup[i][j] = o
        return Field(oup)

    def train(self, iodata_list):
        a_size = 4 #get_a_size(task_json)
        bl_cols = Augmenter.get_bl_cols(iodata_list)
        
        isflip = False
        X1,Y1 = Augmenter.gettaskxy(iodata_list, True, 1, bl_cols, isflip)
        X3,Y3 = Augmenter.gettaskxy(iodata_list, True, 3, bl_cols, isflip)
        X5,Y5 = Augmenter.gettaskxy(iodata_list, True, 5, bl_cols, isflip)
        
        self.model_1 = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100).fit(X1, Y1)
        self.model_3 = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100).fit(X3, Y3)
        self.model_5 = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100).fit(X5, Y5)
        pass

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        #while True:
        #pred_map_1 = submit_predict(task_json,model_1, 1)
        pred1 = self.predict_on_tree_model(field, self.model_1, 1)
        yield pred1
        pred3 = self.predict_on_tree_model(field, self.model_3, 3)
        yield pred3
        pred5 = self.predict_on_tree_model(field, self.model_5, 5)
        yield pred5

    def __str__(self):
        return f"AugmentedPredictor()"
