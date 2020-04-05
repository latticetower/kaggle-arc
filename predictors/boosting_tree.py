"""
Based on https://www.kaggle.com/meaninglesslives/using-decision-trees-for-arc
"""
from xgboost import XGBClassifier

from base.field import Field
from base.iodata import IOData

from predictors.basic import *


class BoostingTreePredictor(Predictor, AvailableAll):
    def __init__(self):
        pass

    def train(self, iodata_list):
        pass

    def predict(self, field):
        pass

    def __str__(self):
        return "BoostingTreePredictor()"
    
