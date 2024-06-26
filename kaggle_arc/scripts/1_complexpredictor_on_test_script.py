import os
import pandas as pd

from constants import *
from base.field import Field

from utils import *
from base.field import *

from predictors.basic import IdPredictor, ZerosPredictor, ConstPredictor
from predictors.complex import ComplexPredictor


datasets = read_datasets(DATADIR)
train_ds, eval_ds, test_ds = [convert2samples(x) for x in datasets]

# predictor = IdPredictor()
predictor_args = [IdPredictor, ZerosPredictor]
for i in range(1, 10):
    predictor_args.append((ConstPredictor, [], {"value": i}))

save_predictions(ComplexPredictor, test_ds, TEST_SAVEPATH, k=3, args=[predictor_args])
