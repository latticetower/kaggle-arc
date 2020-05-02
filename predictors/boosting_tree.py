"""
Based on https://www.kaggle.com/meaninglesslives/using-decision-trees-for-arc
"""
from xgboost import XGBClassifier
from itertools import product

from base.field import Field
from base.iodata import IOData

from predictors.basic import *
from operations.basic import Repaint

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
            input_field = iodata.input_processed
            output_field = iodata.output_processed
            nrows, ncols = input_field.shape

            target_rows, target_cols = output_field.shape
            
            if (target_rows != nrows) or (target_cols != ncols):
                print('Number of input rows:', nrows,'cols:',ncols)
                print('Number of target rows:',target_rows,'cols:',target_cols)
                not_valid=1
                return None, None, 1

            feat.extend(BTFeatureExtractor.make_features(input_field))
            target.extend(np.array(output_field.data).reshape(-1,))
        return np.array(feat), np.array(target), 0


class BoostingTreePredictor(Predictor, AvailableEqualShape):
    def __init__(self):
        self.xgb =  XGBClassifier(n_estimators=100, booster="dart", n_jobs=-1,
            objective="multi:softmax", num_class=10)

    def train(self, iodata_list):
        feat, target, _ = BTFeatureExtractor.get_features(iodata_list)
        self.xgb.fit(feat, target, verbose=-1)

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field.data):
                yield field.reconstruct(v)
            return
        #repainter = Repaint(field.data)
        nrows, ncols = field.shape
        feat = BTFeatureExtractor.make_features(field.processed)
        preds = self.xgb.predict(feat).reshape(nrows, ncols)
        preds = preds.astype(int)#.tolist()
        preds = field.reconstruct(preds)
        #result = repainter(preds).tolist()
        yield preds

    def __str__(self):
        return "BoostingTreePredictor()"
    
