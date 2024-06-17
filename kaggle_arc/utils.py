import json
import os
from collections import OrderedDict

import pandas as pd
import numpy as np

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
    

def save_predictions(predictor, ds, savepath, k=3, args=[], kwargs=dict(), verbose=True):
    all_data = []
    for name, i, prediction in predictor.predict_on(
            ds, k=k, args=args, kwargs=kwargs, verbose=verbose):
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


def check_if_can_be_mirrored(data, h=14, w=9):
    #w, h = iodata.input_field.shape
    sample = data[:h, :w]
    h1, w1 = data.shape
    m1, m2 = h1//h, w1//w
    buf = dict()
    buf[(0, 0)] = sample
    for i in range(m1):
        for j in range(m2):
            if i == 0 and j == 0:
                continue
            current = data[i*h : i*h + h, j*w : j*w + w]
            p = (i % 2, j % 2)
            #print(p, h, w)
            if p in buf:
                if not np.all(buf[p] == current):
                    return None
            else:
                buf[p] = current
    a1 = np.all(sample == buf[0, 1])
    a2 = np.all(sample == buf[1, 0])
    a3 = np.all(sample == buf[1, 1])
    if a1 and a2 and a3:
        return (False, False)
    b1 = np.all(sample[:, ::-1] == buf[0, 1])
    b2 = np.all(sample[::-1, :] == buf[1, 0])
    b3 = np.all(buf[1, 1] == buf[1, 0])
    b4 = np.all(buf[1, 1] == buf[0, 1])
    b5 = np.all(sample[::-1, ::-1] == buf[1, 1])
    if b1 and b2 and b5:
        return (True, True)
    if b1 and a2 and b4:
        return (False, True)
    if b2 and a1 and b3:
        return (True, False)
    return None
