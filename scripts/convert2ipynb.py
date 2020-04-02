import argparse
import json


TEMPLATE = {
    "cells": [],
    "metadata": {
    "kernelspec": {
     "display_name": "Python 3",
     "language": "python",
     "name": "python3"
    },
    "language_info": {
     "codemirror_mode": {
      "name": "ipython",
      "version": 3
     },
     "file_extension": ".py",
     "mimetype": "text/x-python",
     "name": "python",
     "nbconvert_exporter": "python",
     "pygments_lexer": "ipython3",
     "version": "3.7.3"
    }
   },
   "nbformat": 4,
   "nbformat_minor": 2
}

class DepGraph:
    def __init__(self, mainpy):
        self.mainpy = mainpy
    def sorted_files(self):
        return [self.mainpy]

def read_file(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    return lines

def wrap2cell(data):
    return {
        "cell_type": "code",
        "execution_count": 0,
        "metadata": {},
        "outputs": [],
        "source": data
        }


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mainpy", help="path or name of main file which contains runnable code")
    parser.add_argument("savepath", help="path where .ipynb notebook should be saved")
    args = parser.parse_args()
    graph = DepGraph(args.mainpy)
    for file_path in graph.sorted_files():
        data = read_file(file_path)
        TEMPLATE['cells'].append(wrap2cell(data))
    with open(args.savepath, 'w') as f:
        json.dump(TEMPLATE, f, indent=2)