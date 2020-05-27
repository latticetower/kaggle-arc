import os
import pandas as pd

from constants import *
from base.field import Field

from utils import *
from base.field import *

from predictors.basic import IdPredictor, ZerosPredictor, ConstPredictor, FillPredictor
from predictors.complex import ComplexPredictor
from predictors.color_counting import ColorCountingPredictor
from predictors.shapes import RepeatingPredictor, FractalPredictor, ResizingPredictor, MirrorPredictor, ConstantShaper
from predictors.boosting_tree import BoostingTreePredictor, BoostingTreePredictor2, BoostingTreePredictor3
from predictors.convolution import ConvolutionPredictor
from predictors.graph_boosting_tree import GraphBoostingTreePredictor, GraphBoostingTreePredictor2, GraphBoostingTreePredictor3
from predictors.decision_tree import AugmentedPredictor
from predictors.subpattern import SubpatternMatcherPredictor
from predictors.connector import PointConnectorPredictor
from predictors.cam_predictor import *
from predictors.connector import PointConnectorPredictor
from predictors.cf_combinator import WrappedCFPredictor

datasets = read_datasets(DATADIR)
train_ds, eval_ds, test_ds = [ convert2samples(x) for x in datasets ]

#predictor = IdPredictor()
predictor_args = [
    IdPredictor,
    ZerosPredictor,
    ColorCountingPredictor,
    RepeatingPredictor,
    FractalPredictor,
    ResizingPredictor,
    #GraphBoostingTreePredictor,#no impact
    GraphBoostingTreePredictor3,
    ConstantShaper,
    #BoostingTreePredictor,
    #BoostingTreePredictor2,
    BoostingTreePredictor3,
    SubpatternMatcherPredictor,
    #GraphBoostingTreePredictor2,
    PointConnectorPredictor,
    #AugmentedPredictor,
    FillPredictor,
    WrappedCFPredictor,
    MirrorPredictor,
    #(ConvolutionPredictor, [], {'loss': 'mse'}),
    #(ConvolutionPredictor, [], {'loss': 'dice'})
    ]
#for i in range(1, 10):
#    predictor_args.append((ConstPredictor, [], {'value': i}))

save_predictions(ComplexPredictor, test_ds, TEST_SAVEPATH,
    k=3, args=[ predictor_args ])