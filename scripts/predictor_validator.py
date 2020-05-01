import os
import sys
import pandas as pd
import json
import argparse

from constants import *
from base.field import Field

from utils import *
from base.field import *

from predictors.basic import IdPredictor, ZerosPredictor, ConstPredictor, FillPredictor, Predictor
from predictors.complex import ComplexPredictor
from predictors.color_counting import ColorCountingPredictor
from predictors.shapes import RepeatingPredictor, FractalPredictor, ResizingPredictor, MirrorPredictor, ConstantShaper
from predictors.boosting_tree import BoostingTreePredictor
from predictors.convolution import ConvolutionPredictor
from predictors.graph_boosting_tree import GraphBoostingTreePredictor, GraphBoostingTreePredictor2, GraphBoostingTreePredictor3

datasets = read_datasets(DATADIR)
train_ds, eval_ds, test_ds = [ convert2samples(x) for x in datasets ]

def evaluate_on_dataset(predictor_class, ds, cutoff=1.0):
    nsamples = 0
    train1 = 0
    test1 = 0
    params = {}
    params['total'] = len(ds)
    params['train_score'] = dict()
    params['test_score'] = dict()
    for i, sample in enumerate(ds):
        predictor = predictor_class()
        if not predictor.is_available(sample.train):
            continue
        nsamples += 1
        predictor.train(sample.train)
        predictor.freeze_by_score(sample.train)
        score_train = predictor.validate(sample.train)
        params['train_score'][i] = score_train
        score_test = predictor.validate(sample.test)
        params['test_score'][i] = score_test
        if score_train >= cutoff:
            train1 += 1
        if score_test >= cutoff:
            test1 += 1
    return train1, test1, nsamples, params

if len(sys.argv) < 1:
    print("no predictor classes were provided")

names = sys.argv[1: ] + [ n + "Predictor" for n in sys.argv[1:] if n.find("Predictor") < 0 ]
savedir = "../temp/eval"
if not os.path.exists(savedir):
    os.makedirs(savedir)

for name in names:
    if not name in globals():
        #print(f"{name} predictor not found")
        continue
    predictor_class = globals()[name]
    # if not isinstance(predictor_class, Predictor):
    #     print(f"{name} is not a predictor")
    #     continue
    all_results = [name]
    for i, ds in enumerate([train_ds, eval_ds]):
        result = evaluate_on_dataset(predictor_class, ds, cutoff=1.0)
        params = result[-1]
        with open(os.path.join(savedir, f"{name}_{i}.json"), 'w') as f:
            json.dump(params, f)
        result = result[:-1]
        result = " / ".join(([f"{r:d}" for r in result]))
        all_results.append(result)
    all_results = " | ".join(all_results)
    print(all_results)