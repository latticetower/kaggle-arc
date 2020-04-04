import json
import os
from collections import OrderedDict
import pandas as pd
from base.iodata import Sample
from base.field import Field


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
    

def save_predictions(predictor, ds, savepath, k=3, args=[], kwargs=dict()):
    all_data = []
    for name, i, prediction in predictor.predict_on(ds, k=k, args=args, kwargs=kwargs):
        if isinstance(prediction, Field):
            preds = [str(prediction)]*k
        if isinstance(prediction, list):
            preds = [str(p) for p in prediction]
            if len(preds) < k:
                preds = (preds * k)[:k]
        preds = " ".join(preds)
        all_data.append({
            'output_id': f"{name}_{i}",
            'output': preds
        })
    pd.DataFrame(all_data, columns=['output_id', "output"]).to_csv(savepath, index=None)
