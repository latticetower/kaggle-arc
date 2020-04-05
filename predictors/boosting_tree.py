"""
Based on https://www.kaggle.com/meaninglesslives/using-decision-trees-for-arc
"""
from xgboost import XGBClassifier

from base.field import Field
from base.iodata import IOData

from predictors.basic import *


def get_moore_neighbours(field, cur_row, cur_col, nrows, ncols):
    if cur_row <= 0:
        top = -1
    else:
        top = field.data[cur_row - 1, cur_col]
        
    if cur_row >= nrows - 1:
        bottom = -1
    else:
        bottom = field.data[cur_row + 1, cur_col]
        
    if cur_col <= 0:
        left = -1
    else:
        left = field.data[cur_row, cur_col - 1]
        
    if cur_col >= ncols - 1:
        right = -1
    else:
        right = field.data[cur_row, cur_col + 1]
        
    return top, bottom, left, right


def get_tl_tr(field, cur_row, cur_col, nrows, ncols):
    if cur_row == 0:
        top_left = -1
        top_right = -1
    else:
        if cur_col == 0:
            top_left =- 1
        else:
            top_left = field.data[cur_row - 1, cur_col - 1]
        if cur_col == ncols - 1:
            top_right =- 1
        else:
            top_right = field.data[cur_row - 1, cur_col + 1]   
        
    return top_left, top_right


def make_features(field, nfeat=13, local_neighb=5):
    nrows, ncols = field.shape
    feat = np.zeros((nrows*ncols, nfeat))
    cur_idx = 0
    for i in range(nrows):
        for j in range(ncols):
            feat[cur_idx,0] = i
            feat[cur_idx,1] = j
            feat[cur_idx,2] = field.data[i, j]
            feat[cur_idx,3:7] = get_moore_neighbours(field, i, j, nrows, ncols)
            feat[cur_idx,7:9] = get_tl_tr(field, i, j, nrows, ncols)
            feat[cur_idx,9] = len(np.unique(field.data[i,:]))
            feat[cur_idx,10] = len(np.unique(field.data[:,j]))
            feat[cur_idx,11] = (i+j)
            feat[cur_idx,12] = len(np.unique(field.data[i-local_neighb:i+local_neighb,
                                                         j-local_neighb:j+local_neighb]))

            cur_idx += 1
        
    return feat

def get_features(iodata_list):
    feat = []
    target = []
    for i, iodata in enumerate(iodata_list):
        input_field = iodata.input_field
        output_field = iodata.output_field
        nrows, ncols = input_field.shape

        target_rows, target_cols = output_field.shape
        
        if (target_rows != nrows) or (target_cols != ncols):
            print('Number of input rows:', nrows,'cols:',ncols)
            print('Number of target rows:',target_rows,'cols:',target_cols)
            not_valid=1
            return None, None, 1

        feat.extend(make_features(input_field))
        target.extend(np.array(output_field.data).reshape(-1,))
    return np.array(feat), np.array(target), 0


class BoostingTreePredictor(Predictor, AvailableEqualShape):
    def __init__(self):
        self.xgb =  XGBClassifier(n_estimators=25, n_jobs=-1)

    def train(self, iodata_list):
        feat, target, _ = get_features(iodata_list)
        self.xgb.fit(feat, target, verbose=-1)

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        nrows, ncols = field.shape
        feat = make_features(field)
        preds = self.xgb.predict(feat).reshape(nrows, ncols)
        preds = preds.astype(int).tolist()
        yield Field(preds)

    def __str__(self):
        return "BoostingTreePredictor()"
    
