# kaggle-arc
https://www.kaggle.com/c/abstraction-and-reasoning-challenge

# Usage:

Direct script calls look similar to this:
```
cd scripts

PYTHONPATH=$(pwd)/..:$PYTHONPATH python 1_complexpredictor_on_test_script.py 1>out.txt
```

To convert script to single ipynb notebook, call command from the root repo folder:
```
python scripts/convert2ipynb.py scripts/runner.py temp/sample1.ipynb
```
first parameter is a path to main script, second is a path to save notebook.

# Evaluation

To evaluate predictors on train and evaluation datasets, the following command was used
```
PYTHONPATH=$(pwd)/..:$PYTHONPATH python predictor_validator.py Id Zeros ColorCounting Repeating Fractal
```
Table format: correct predictions on train / correct predictions on test / count of samples for which predictor is available
Class name | Train | Evaluation
-----------|-------|-----------
IdPredictor | 4 / 0 / 400 | 2 / 0 / 400
ZerosPredictor | 0 / 0 / 400 | 0 / 0 / 400
ColorCountingPredictor | 16 / 5 / 262 | 2 / 1 / 270
RepeatingPredictor | 0 / 0 / 17 | 0 / 0 / 23
FractalPredictor | 1 / 1 / 17 | 2 / 1 / 23
ResizingPredictor | 2 / 2 / 17 | 1 / 1 / 23
ConstantShaper | 4 / 4 / 15 | 3 / 3 / 10
BoostingTreePredictor | 136 / 24 / 262 | 134 / 8 / 270
No augmentation, with painter | 217 / 31 / 262 | 188 / 8 / 270
with square features  | 214 / 31 / 262 | 191 / 9 / 270
with new features | 161 / 32 / 262 | 135 / 7 / 270
Augmentation + repainter: | 79 / 8 / 262 | 43 / 2 / 270
Augmentation, w/o repainter: | 83 / 7 / 262 | 37 / 1 / 270
BoostingTreePredictor2 | 2 / 1 / 31 | 3 / 3 / 27
BoostingTreePredictor3 | 218 / 31 / 262 | 198 / 12 / 270
SubpatternMatcherPredictor | 2 / 2 / 10 | 0 / 0 / 2
SimpleSummarizePredictor | 1 / 1 / 6 | 0 / 2 / 2
GraphBoostingTreePredictor | 17 / 13 / 43 | 9 / 6 / 33
GraphBoostingTreePredictor3 | 155 / 28 / 262 | 143 / 6 / 270
PointConnectorPredictor | 24 / 3 / 90 | 11 / 2 / 68
ConvolutionPredictor | 11 / 1 / 262 | 3 / 0 / 270