import json
import os
from collections import OrderedDict
import pandas as pd
from base.iodata import Sample


def read_datasets(basedir="../data"):
    train_dir = os.path.join(basedir, "training")
    train_data = OrderedDict(
        (os.path.splitext(x)[0], os.path.join(train_dir, x)) 
        for x in os.listdir(train_dir))
    eval_dir = os.path.join(basedir, "evaluation")
    eval_data = OrderedDict(
        (os.path.splitext(x)[0], os.path.join(eval_dir, x))
        for x in os.listdir(eval_dir))
    test_dir = os.path.join(basedir, "test")
    test_data = OrderedDict(
        (os.path.splitext(x)[0], os.path.join(test_dir, x))
        for x in os.listdir(test_dir))
    return train_data, eval_data, test_data
    

def convert2samples(data):
    return [ Sample(name, path) for name, path in data.items() ]
    

def save_predictions(predictor, ds, savepath):
    all_data = []
    for name, i, prediction in predictor.predict_on(ds):
        all_data.append({
            'output_id': f"{name}_{i}",
            'output': " ".join([str(prediction)]*3)
        })
    pd.DataFrame(all_data, columns=['output_id', "output"]).to_csv(savepath, index=None)
