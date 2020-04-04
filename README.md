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