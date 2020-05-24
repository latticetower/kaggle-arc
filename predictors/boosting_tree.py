"""
Based on https://www.kaggle.com/meaninglesslives/using-decision-trees-for-arc
"""
from xgboost import XGBClassifier
from itertools import product
import itertools
from skimage.measure import label
import numpy as np

from base.field import Field
from base.iodata import IOData
from base.utils import *

from predictors.basic import *
from operations.basic import Repaint
from operations.reversible import *
from operations.field2point import ComplexSummarizeOperation


class BTFeatureExtractor:
    @staticmethod
    def get_moore_neighbours(field, cur_row, cur_col, nrows, ncols, color=0):
        if cur_row <= 0:
            top = color
        else:
            top = field.data[cur_row - 1, cur_col]
            
        if cur_row >= nrows - 1:
            bottom = color
        else:
            bottom = field.data[cur_row + 1, cur_col]
            
        if cur_col <= 0:
            left = color
        else:
            left = field.data[cur_row, cur_col - 1]
            
        if cur_col >= ncols - 1:
            right = color
        else:
            right = field.data[cur_row, cur_col + 1]
            
        return top, bottom, left, right

    @staticmethod
    def get_tl_tr(field, cur_row, cur_col, nrows, ncols, color=0):
        if cur_row == 0:
            top_left = color
            top_right = color
        else:
            if cur_col == 0:
                top_left = color
            else:
                top_left = field.data[cur_row - 1, cur_col - 1]
            if cur_col == ncols - 1:
                top_right = color
            else:
                top_right = field.data[cur_row - 1, cur_col + 1]   
            
        return top_left, top_right
        
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
            #v.append(-1)
            new_el = inp.data[ii, jj] if((0<= ii < r) and (0<= jj < c)) else -1
            v.append(new_el)
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

    @staticmethod
    def make_features(field, nfeat=13, local_neighb=5, all_square=False):
        nrows, ncols = field.shape
        #feat = np.zeros((nrows*ncols, nfeat))
        all_features = []
        cur_idx = 0
        for i in range(nrows):
            for j in range(ncols):
                color = field.data[i, j]
                features = [
                    i,
                    j,
                    i*j,
                    field.data[i, j]]
                features.extend(
                    BTFeatureExtractor.get_moore_neighbours(field, i, j, nrows, ncols))
                features.extend(
                    BTFeatureExtractor.get_tl_tr(field, i, j, nrows, ncols))
                features.extend([
                    len(np.unique(field.data[i,:])),
                    len(np.unique(field.data[:,j])),
                    #next goes count of non-zero points
                    np.sum(field.data[i, :] > 0),
                    np.sum(field.data[:, j] > 0),
                    (i+j),
                    len(np.unique(field.data[
                        i-local_neighb:i+local_neighb,
                        j-local_neighb:j+local_neighb]))
                ])
                
                #feat[cur_idx,13]
                features.extend([
                    (i + ncols - j - 1),
                    (i + j) % 2,
                    (i + j + 1) % 2,
                    (i + ncols - j - 1) % 2, #
                    (nrows - 1 - i + ncols - j - 1),#
                    (nrows - 1 - i + j) #
                ])
                features.extend([
                    field.get(i + k, j + v)
                    for k, v in product([-1, 0, 1], [-1, 0, 1])
                ])
                features.extend([
                    field.data[nrows - 1 - i, j],
                    field.data[nrows - 1 - i, ncols - 1 - j],
                    field.data[i, ncols - 1 - j]
                ])
                if all_square: #and field.data.shape[0] == field.data.shape[1]:
                    features.extend([
                        field.get(j, i),
                        field.get(j, nrows - 1 - i),
                        field.get(ncols - 1 - j, nrows - 1 - i),
                        field.get(ncols - 1 - j, i)
                    ])
                features.extend([
                    field.data[i, j] != 0,
                    np.sum([ field.get(i+k, j+v) == color
                        for k, v in product([-1, 1], [-1, 1])]),
                    np.sum([
                        field.get(i + 1, j) == color,
                        field.get(i - 1, j) == color,
                        field.get(i, j + 1) == color,
                        field.get(i, j - 1) == color
                    ]),
                    #next were commented
                    np.sum([ field.get(i + k, j + v) == 0
                        for k, v in product([-1, 1], [-1, 1])]),
                    np.sum([
                        field.get(i + 1, j) == 0,
                        field.get(i - 1, j) == 0,
                        field.get(i, j + 1) == 0,
                        field.get(i, j - 1) == 0
                    ])
                ])
                all_features.append(features)

        feat = np.asarray(all_features)
        return feat

    @classmethod
    def make_features_v2(cls, field, nfeat=13, local_neighb=5, all_square=False):
        nrows, ncols = field.shape
        #feat = np.zeros((nrows*ncols, nfeat))
        all_features = []
        regions = [label(field.data==i) for i in range(10)]
        cur_idx = 0
        for i in range(nrows):
            for j in range(ncols):
                color = field.data[i, j]
                features = [
                    i,
                    j,
                    i*j,
                    field.data[i, j]]
                    
                for m in range(1, 6):
                    features.extend([
                        i % m,
                        j % m
                    ])
                features.extend([
                        (i+1)/(j+1),
                        (j+1)/(i+1),
                        nrows, ncols
                ])
                for size in [1, 3, 5]:
                    arnd = cls.getAround(i,j, field, size)
                    features.append(len(np.unique(arnd)))
                    features.extend(arnd)
                features.extend(
                    BTFeatureExtractor.get_moore_neighbours(field, i, j, nrows, ncols))
                features.extend(
                    BTFeatureExtractor.get_tl_tr(field, i, j, nrows, ncols))
                features.extend([
                    len(np.unique(field.data[i,:])),
                    len(np.unique(field.data[:,j])),
                    #next goes count of non-zero points
                    np.sum(field.data[i, :] > 0),
                    np.sum(field.data[:, j] > 0),
                    (i+j),
                    len(np.unique(field.data[
                        i-local_neighb:i+local_neighb,
                        j-local_neighb:j+local_neighb]))
                ])
                
                #feat[cur_idx,13]
                features.extend([
                    (i + ncols - j - 1),
                    (i + j) % 2,
                    (i + j + 1) % 2,
                    (i + ncols - j - 1) % 2, #
                    (nrows - 1 - i + ncols - j - 1) % 2,#
                    (nrows - 1 - i + j) % 2  #
                ])
                features.extend([
                    field.get(i + k, j + v)
                    for k, v in product([-1, 0, 1], [-1, 0, 1])
                ])
                features.extend([
                    field.data[nrows - 1 - i, j],
                    field.data[nrows - 1 - i, ncols - 1 - j],
                    field.data[i, ncols - 1 - j]
                ])
                if all_square: #and field.data.shape[0] == field.data.shape[1]:
                    features.extend([
                        field.get(j, i),
                        field.get(j, nrows - 1 - i),
                        field.get(ncols - 1 - j, nrows - 1 - i),
                        field.get(ncols - 1 - j, i)
                    ])
                features.extend([
                    field.data[i, j] != 0,
                    np.sum([ field.get(i+k, j+v) == color
                        for k, v in product([-1, 1], [-1, 1])]),
                    np.sum([
                        field.get(i + 1, j) == color,
                        field.get(i - 1, j) == color,
                        field.get(i, j + 1) == color,
                        field.get(i, j - 1) == color
                    ]),
                    #next were commented
                    np.sum([ field.get(i + k, j + v) == 0
                        for k, v in product([-1, 1], [-1, 1])]),
                    np.sum([
                        field.get(i + 1, j) == 0,
                        field.get(i - 1, j) == 0,
                        field.get(i, j + 1) == 0,
                        field.get(i, j - 1) == 0
                    ])
                ])
                all_features.append(features)

        feat = np.asarray(all_features)
        return feat

    @classmethod
    def make_features_v3(cls, field, nfeat=13, local_neighb=5, all_square=False):
        nrows, ncols = field.shape
        prop_names = "h w is_convex is_rectangular is_square holes contour_size interior_size".split()+\
            [f"flip_{i}" for i in range(10)] + [f"flip_conv_{i}" for i in range(10)]
        
        regions0 = get_data_regions(field.data)
        params0, maps0 = get_region_params(regions0)
        
        regions1 = get_data_regions(field.data, connectivity=1)
        params1, maps1 = get_region_params(regions1, connectivity=1)

        #feat = np.zeros((nrows*ncols, nfeat))
        all_features = []
        regions = [label(field.data==i) for i in range(10)]
        cur_idx = 0
        for i in range(nrows):
            for j in range(ncols):
                rid0 = regions0[i, j]
                rid1 = regions1[i, j]
                
                color = field.data[i, j]
                features = [
                    i,
                    j,
                    i*j,
                    field.data[i, j]]
                features.extend([params0[rid0][n] for n in prop_names])
                features.extend([params1[rid1][n] for n in prop_names])
                features.append(maps0[rid0]['contour'][i, j])
                features.append(maps0[rid0]['interior'][i, j])
                features.append(maps1[rid1]['contour'][i, j])
                features.append(maps1[rid1]['interior'][i, j])
                
                for m in range(1, 6):
                    features.extend([
                        i % m,
                        j % m
                    ])
                features.extend([
                        (i+1)/(j+1),
                        (j+1)/(i+1),
                        nrows, ncols
                ])
                for size in [1, 3, 5]:
                    arnd = cls.getAround(i,j, field, size)
                    features.append(len(np.unique(arnd)))
                    features.extend(arnd)
                features.extend(
                    BTFeatureExtractor.get_moore_neighbours(field, i, j, nrows, ncols))
                features.extend(
                    BTFeatureExtractor.get_tl_tr(field, i, j, nrows, ncols))
                features.extend([
                    len(np.unique(field.data[i,:])),
                    len(np.unique(field.data[:,j])),
                    #next goes count of non-zero points
                    np.sum(field.data[i, :] > 0),
                    np.sum(field.data[:, j] > 0),
                    (i+j),
                    len(np.unique(field.data[
                        i-local_neighb:i+local_neighb,
                        j-local_neighb:j+local_neighb]))
                ])
                
                #feat[cur_idx,13]
                features.extend([
                    (i + ncols - j - 1),
                    (i + j) % 2,
                    (i + j + 1) % 2,
                    (i + ncols - j - 1) % 2, #
                    (nrows - 1 - i + ncols - j - 1) % 2,#
                    (nrows - 1 - i + j) % 2  #
                ])
                features.extend([
                    field.get(i + k, j + v)
                    for k, v in product([-1, 0, 1], [-1, 0, 1])
                ])
                features.extend([
                    field.data[nrows - 1 - i, j],
                    field.data[nrows - 1 - i, ncols - 1 - j],
                    field.data[i, ncols - 1 - j]
                ])
                if all_square: #and field.data.shape[0] == field.data.shape[1]:
                    features.extend([
                        field.get(j, i),
                        field.get(j, nrows - 1 - i),
                        field.get(ncols - 1 - j, nrows - 1 - i),
                        field.get(ncols - 1 - j, i)
                    ])
                features.extend([
                    field.data[i, j] != 0,
                    np.sum([ field.get(i+k, j+v) == color
                        for k, v in product([-1, 1], [-1, 1])]),
                    np.sum([
                        field.get(i + 1, j) == color,
                        field.get(i - 1, j) == color,
                        field.get(i, j + 1) == color,
                        field.get(i, j - 1) == color
                    ]),
                    #next were commented
                    np.sum([ field.get(i + k, j + v) == 0
                        for k, v in product([-1, 1], [-1, 1])]),
                    np.sum([
                        field.get(i + 1, j) == 0,
                        field.get(i - 1, j) == 0,
                        field.get(i, j + 1) == 0,
                        field.get(i, j - 1) == 0
                    ])
                ])
                all_features.append(features)

        feat = np.asarray(all_features)
        return feat

    @staticmethod
    def get_features(iodata_list, all_square=False, features_maker=make_features):
        feat = []
        target = []
            
        for i, iodata in enumerate(iodata_list):
            if isinstance(iodata, IOData):
                input_field = iodata.input_field.data
                output_field = iodata.output_field.data
            else:
                input_field, output_field = iodata
                input_field = input_field.data
                output_field = output_field.data
            nrows, ncols = input_field.shape
            #output_field = output_field.data

            target_rows, target_cols = output_field.shape
            if output_field.shape == (1, 1): #and input_field.shape != output_field.shape:
                #print(input_field.shape)
                #print(input_field)
                output_field = increase2shape(output_field, input_field.shape)
                # i = np.asarray([[ np.sum(input_field == i) for i in range(10)]])
                # o = output_field
                # print(i.shape, o.shape)
                # feat.extend(i)
                # target.extend(o)
                # continue
            elif (target_rows != nrows) or (target_cols != ncols):
                print('Number of input rows:', nrows,'cols:',ncols)
                print('Number of target rows:',target_rows,'cols:',target_cols)
                not_valid=1
                return None, None, 1

            feat.extend(features_maker(
                Field(input_field),
                all_square=all_square))
            target.extend(np.array(output_field).reshape(-1,))
        return np.array(feat), np.array(target), 0


def get_augmented_iodata(i, o):
    values = list(set(np.unique(i)) | set(np.unique(o)))
    permutations = list(set(tuple(np.random.permutation(len(values))) for k in range(5)))
    for permutation in permutations:
        rv = {k: values[v] for k, v in zip(values, permutation)}
        inp = np.asarray([[rv[x] for x in line] for line in i])
        out = np.asarray([[rv[x] for x in line] for line in o])
        yield Field(inp), Field(out)


class BoostingTreePredictor(Predictor, AvailableEqualShape):
    def __init__(self):
        self.xgb =  XGBClassifier(n_estimators=100, booster="dart", n_jobs=-1,
            objective="multi:softmax", num_class=10)

    def train(self, iodata_list):
        self.all_square = np.all([
            iodata.input_field.shape[0] == iodata.output_field.shape[1]
            for iodata in iodata_list
        ])
        iodata_list_ = list(
            itertools.chain(*[
                get_augmented_iodata(
                    iodata.input_field.data, iodata.output_field.data
                )
                for iodata in iodata_list
            ]))
        feat, target, _ = BTFeatureExtractor.get_features(
            iodata_list, all_square=self.all_square,
            features_maker=BTFeatureExtractor.make_features_v3
            )
        self.xgb.fit(feat, target, verbose=-1)

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        #repainter = Repaint(field.data)
        nrows, ncols = field.shape
        feat = BTFeatureExtractor.make_features_v3(field, all_square=self.all_square)
        preds = self.xgb.predict(feat).reshape(nrows, ncols)
        preds = preds.astype(int)#.tolist()
        #preds = field.reconstruct(Field(preds))
        #preds = repainter(preds).tolist()
        preds = Field(preds)
        yield preds

    def __str__(self):
        return "BoostingTreePredictor()"
    

class BoostingTreePredictor2(Predictor):
    """This class needs renaming:
    actually it takes image, splits it to subparts and then tries to use single pixel for each image.
    """
    def __init__(self):
        self.xgb =  XGBClassifier(n_estimators=10, booster="dart", n_jobs=-1,
            objective="multi:softmax", num_class=10) # currently not in use
        self.bgr_color = None
        self.simple_operation = ComplexSummarizeOperation()

    def is_available(self, iodata_list):
        all_sizes = set()
        for iodata in iodata_list:
            if iodata.input_field.height <= iodata.output_field.height or \
                    iodata.input_field.width <= iodata.output_field.width:
                return False
            m1 = iodata.output_field.height  # // iodata.input_field.height
            m2 = iodata.output_field.width  # // iodata.input_field.width
            all_sizes.add((m1, m2))
        if len(all_sizes) == 1:
            h, w = all_sizes.pop()
            if w > 1 and h > 1:
                self.m1 = h
                self.m2 = w
                self.op = WrappedOperation(
                    ReversibleSplit((h, w)),
                    ReversibleCombine((h, w)))
                return True

        return False

    def get_bgr_color(self, iodata_list):
        features = np.asarray([[np.sum(x[0].data == i) for i in range(10)] for x in iodata_list])
        targets = np.asarray([[np.sum(x[1].data == i) for i in range(10)] for x in iodata_list])
        ids = np.sum(features > 0, 1) > 1
        bgr = (targets[ids] == 0)*features[ids]
        bgr_color = np.argwhere(bgr.sum(0)).flatten()
        return bgr_color

    def get_bgr_color_by_features(self, features):
        #print(features)
        #features = np.asarray([[np.sum(x[0].data == i) for i in range(10)] for x in iodata_list])
        #targets = np.asarray([[np.sum(x[1].data == i) for i in range(10)] for x in iodata_list])
        colors = np.argmax(features, 1)
        
        return np.unique(colors)
         
    def get_target_color(self, features, bgr_color):
        a = np.argwhere(features > 0).flatten()
        for x in a:
            if not x in bgr_color:
                return x
        for xs in [a, bgr_color, [0]]:
            if len(xs) > 0:
                return xs[0]

    def train(self, iodata_list):
        all_samples = []
        for iodata in iodata_list:
            i, o = self.op.wrap(iodata)
            all_samples.append((i, o))
        self.simple_operation.train(all_samples)
        #print(all_samples)
        #print(all_samples)
        ##self.bgr_color = self.get_bgr_color(all_samples)
        #for (i, o) in all_samples:
        #    print(i.shape, o.shape)
        ##feat, target, _ = BTFeatureExtractor.get_features(all_samples)
        # print(feat.shape, target.shape)
        ##self.xgb.fit(feat, target, verbose=-1)

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        #repainter = Repaint(field.data)
        nrows, ncols = field.shape
        feature_field, postprocess = self.op.run(field)
        #for line in feature_field:
        #    for x in line:
        #        print(x.data)
        #print(field.shape, self.m1, self.m2)
        #print(feature_field)
        features = np.asarray([
            [np.sum(x.data==c) for c in range(10)]
            for x in feature_field.flat_iter()])
        
        bg = self.get_bgr_color_by_features(features)
        bg = bg[0] if len(bg) > 0 else None
        def make_subfield_func(bg):
            #features = [np.sum(x == c) for c in range(10)]
            return lambda x: self.simple_operation.do(x, bg=bg)
        o = feature_field.map(make_subfield_func(bg))
        # 
        # features = np.asarray([ [ np.sum(x == c) for c in range(10)]
        #     for l in feature_field for x in l])
        # bg = self.get_bgr_color_by_features(features)
        # #print(features)
        # #print(bg[0])
        # bg = bg[0] if len(bg) > 0 else None
        # o = [
        #     [
        #         self.simple_operation.do(x.data, bg=bg) for x in line
        #     ]
        #     for line in feature_field
        # ]
        
        result = postprocess(o)
        yield result
        #return
        # all_lines = []
        # if self.bgr_color is not None and len(self.bgr_color) > 0:
        #     features = np.asarray([
        #         [ np.sum(x.data == i) for i in range(10)]
        #         for line in feature_field for x in line
        #     ])
        #     #print(features.shape)
        #     bgr_color = self.get_bgr_color_by_features(features)
        #     #print(bgr_color)
        #     for line in feature_field:
        #         line_result = []
        #         for x in line:
        #             features = np.asarray([ np.sum(x.data == i) for i in range(10)])
        #             #bgr_color = self.get_bgr_color_by_features(features)
        #             color = self.get_target_color(features, bgr_color=bgr_color)
        #             preds = Field([[color]])
        #             line_result.append(preds)
        #         all_lines.append(line_result)
        #     result = postprocess(all_lines)
        #     yield result
        #     return
        # for line in feature_field:
        #     line_result = []
        #     for x in line:
        #         nrows, ncols = x.shape
        #         feat = BTFeatureExtractor.make_features(x)
        #         features = np.asarray([ np.sum(x.data) == i for i in range(10)])
        # 
        #         preds = self.xgb.predict(feat).reshape(nrows, ncols)
        #         preds = preds.astype(int)#.tolist()
        #         #preds = field.reconstruct(Field(preds))
        #         preds = [[decrease2color(preds)]]
        #         if self.bgr_color is not None:
        #             #print(self.bgr_color)
        #             #bgr_color = self.get_bgr_color(all_samples)
        #             color = self.get_target_color(features, self.bgr_color)
        #             preds = Field([[color]])
        #         line_result.append(preds)
        #     all_lines.append(line_result)
        # result = postprocess(all_lines)
        # #result = repainter(preds).tolist()
        # yield result

    def __str__(self):
        return "BoostingTreePredictor2()"
    

class BoostingTreePredictor3(Predictor, AvailableEqualShape):
    def __init__(self):
        self.xgb =  XGBClassifier(n_estimators=100, booster="dart", n_jobs=-1,
            objective="multi:softmax", num_class=10)

    def train(self, iodata_list):
        self.all_square = np.all([
            iodata.input_field.shape[0] == iodata.output_field.shape[1]
            for iodata in iodata_list
        ])
        iodata_list_ = list(
            itertools.chain(*[
                get_augmented_iodata(
                    iodata.input_field.data, iodata.output_field.data
                )
                for iodata in iodata_list
            ]))
        feat, target, _ = BTFeatureExtractor.get_features(
            iodata_list, all_square=self.all_square,
            features_maker=BTFeatureExtractor.make_features_v3)
        #print(feat.shape, target.shape)
        self.xgb.fit(feat, target, verbose=-1)

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        #repainter = Repaint(field.data)
        nrows, ncols = field.shape
        feat = BTFeatureExtractor.make_features_v3(field, all_square=self.all_square)
        preds = self.xgb.predict(feat).reshape(nrows, ncols)
        preds = preds.astype(int)#.tolist()
        #preds = field.reconstruct(Field(preds))
        #preds = repainter(preds).tolist()
        preds = Field(preds)
        yield preds

    def __str__(self):
        return "BoostingTreePredictor3()"