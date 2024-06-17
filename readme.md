
![Project image](https://github.com/latticetower/kaggle-arc/blob/main/images/ministry_of_predictors.png)

# Description

Legacy code with my solutions from https://www.kaggle.com/c/abstraction-and-reasoning-challenge. 

I'm refactoring it to reuse in the ongoing kaggle ARC Prize competition.

It is not guaranteed that there are no typos or missing code pieces. If you use this, use at your own risk, especially predictor classes which use xgboost (I've noticed that they crash kaggle notebook on some data samples, but haven't figured out how to fix).

## Installation & usage
```bash
git clone https://github.com/latticetower/kaggle-arc.git kaggle-arc
pip install kaggle-arc
```

```mermaid
---
title: Main project classes
---
classDiagram
    Field --o IOData
    IOData --o Sample
    class Field["kaggle_arc.base.Field"]{
        numpy.array data
        show(ax=None, label=None)
    }
    class IOData["kaggle_arc.iodata.IOData"]{
        Field input_field
        Field output_field
        show(predictor=None, npredictions=1, ...)
    }
    class Sample["kaggle_arc.iodata.Sample"]{
        String name
        List[IOData] train
        List[IOData] test
        show(predictor=None, npredictions=3, ...)
    }
    Predictor <|-- ComplexPredictor
    AvailableAll <|-- ComplexPredictor
    note for Predictor "has many descendant classes"
    class Predictor["kaggle_arc.predictors.basic.Predictor"]{
        <<interface>>
        train(iodata_list)
        predict(field)
        validate(iodata_list, k=3)
        predict_on(predictor_class, ds, ...)
    }
    class ComplexPredictor["kaggle_arc.predictors.complex.ComplexPredictor"]{
        List[Predictor] predictors
        train(iodata_list)
        predict(field)
        validate(iodata_list, k=3)
        freeze_by_score(iodata_list, k=3)
    }
    note for AvailableAll "mixin class\nhas many descendant classes"
    class AvailableAll["kaggle_arc.predictors.basic.AvailableAll"]{
        <<interface>>
        is_available(iodata_list)
    }
```
