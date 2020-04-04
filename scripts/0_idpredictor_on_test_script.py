import os
import pandas as pd

from constants import *
from base.field import Field

from utils import *
from base.field import *

from predictors.basic import IdPredictor


datasets = read_datasets(DATADIR)
train_ds, eval_ds, test_ds = [ convert2samples(x) for x in datasets ]

predictor = IdPredictor()

save_predictions(predictor, test_ds, TEST_SAVEPATH)