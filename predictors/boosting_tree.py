"""
Based on https://www.kaggle.com/meaninglesslives/using-decision-trees-for-arc
"""
from xgboost import XGBClassifier
from itertools import product
import itertools
import numpy as np
from base.field import Field
from base.iodata import IOData

from predictors.basic import *
from operations.basic import Repaint
from operations.reversible import *
from operations.field2point import SimpleSummarizeOperation

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
    def make_features(field, nfeat=13, local_neighb=5):
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
                    (nrows - 1 - i + j)#
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
    def get_features(iodata_list):
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

            feat.extend(BTFeatureExtractor.make_features(Field(input_field)))
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
        iodata_list_ = list(
            itertools.chain(*[
                get_augmented_iodata(
                    iodata.input_field.data, iodata.output_field.data
                )
                for iodata in iodata_list
            ]))
        feat, target, _ = BTFeatureExtractor.get_features(iodata_list)
        self.xgb.fit(feat, target, verbose=-1)

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        #repainter = Repaint(field.data)
        nrows, ncols = field.shape
        feat = BTFeatureExtractor.make_features(field)
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
        self.simple_operation = SimpleSummarizeOperation()

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
        features = np.asarray([ [ np.sum(x == c) for c in range(10)]
            for l in feature_field for x in l])
        bg = self.get_bgr_color_by_features(features)
        #print(features)
        #print(bg[0])
        bg = bg[0] if len(bg) > 0 else None
        o = [
            [
                self.simple_operation.do(x.data, bg=bg) for x in line
            ]
            for line in feature_field
        ]
        
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
    

class BoostingTreePredictor3(Predictor):
    """This class is similar to previous, but uses xgboost inside
    """
    def __init__(self):
        self.xgb =  XGBClassifier(n_estimators=10, booster="dart", n_jobs=-1,
            objective="multi:softmax", num_class=10) # currently not in use
        self.bgr_color = None
        self.simple_operation = SimpleSummarizeOperation()

    def is_available(self, iodata_list):
        all_sizes = set()
        for iodata in iodata_list:
            if iodata.input_field.height <= iodata.output_field.height or \
                    iodata.input_field.width <= iodata.output_field.width:
                return False
            # if iodata.input_field.shape != iodata.output_field.shape:
            #     return False
            # size = 1
            # widths = []
            # heights = []
            # for isize in range(2, d.shape[0]//2):
            #     if d[(isize)::(size + isize)].std(1).sum() == 0:
            #         heights.append(isize)
            # for isize in range(2, d.shape[1]//2):
            #     if d[:, (isize)::(size+isize)].std(1).sum() == 0:
            #         widths.append(isize)
            # if len(heights) == 0 and len(widths) == 0:
            #     return False
            # if len(heights) == 0:
            #     heights.append(d.shape[0])
            # if len(widths) == 0:
            #     widths.append(d.shape[1])
            m1 = iodata.output_field.height  # // iodata.input_field.height
            m2 = iodata.output_field.width  # // iodata.input_field.width
            all_sizes.add((m1, m2))
        if len(all_sizes) == 1:
            h, w = all_sizes.pop()
            if (iodata.input_field.height-1) % (h + 1) != 0 or (iodata.input_field.width-1) % (w + 1) != 0:
                return False
            
            if w > 1 and h > 1:
                self.m1 = h
                self.m2 = w
                self.op = WrappedOperation(
                    ReversibleSplit((h, w), hsep=1, wsep=1, outer_sep=True),
                    #ReversibleSelect((h, w))
                    )
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
        #self.simple_operation.train(all_samples)
        #print(all_samples)
        #print(all_samples)
        ##self.bgr_color = self.get_bgr_color(all_samples)
        #for (i, o) in all_samples:
        #    print(i.shape, o.shape)
        all_samples = [
            (xi, o)
            for (i, o) in all_samples
            for li in i
            for xi in li
        ]
        feat, target, _ = BTFeatureExtractor.get_features(all_samples)
        # print(feat.shape, target.shape)
        self.xgb.fit(feat, target, verbose=-1)

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        #repainter = Repaint(field.data)
        
        feature_field, postprocess = self.op.run(field)
        lines = []
        for line in feature_field:
            line_result = []
            for x in line:
                nrows, ncols = x.shape
                feat = BTFeatureExtractor.make_features(x)
                preds = self.xgb.predict(feat).reshape(nrows, ncols)
                preds = preds.astype(int)#.tolist()
                #print(x.data)
                line_result.append(Field(preds))
            lines.append(line_result)
        #print(field.shape, self.m1, self.m2)
        #print(feature_field)
        #features = np.asarray([ [ np.sum(x == c) for c in range(10)]
        #    for l in feature_field for x in l])
        #print(lines)
        result = postprocess(lines)
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
        return "BoostingTreePredictor3()"
    
