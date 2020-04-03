import argparse
import json
import os
import networkx as nx


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


def filter_imports(lines):
    ids = dict()
    for i, l in enumerate(lines):
        if l.startswith("import "):
            ids[i] = l[6:].strip().split()[0]
            continue
        import_pos = l.find(" import ")
        if import_pos < 0:
            continue
        package = l[:import_pos].split()[1]
        ids[i] = package
    return ids


def filter_local(ids, local_dirs=[], basedir=".."):
    result = dict()
    for k, v in ids.items():
        s = v.split(".")[0]
        if not (s in local_dirs or s + ".py" in local_dirs):
            continue
        path = os.path.join(basedir, v.replace(".", os.sep))
        if os.path.exists(path + ".py"):
            result[k] = (v, path + ".py")
        if os.path.exists(path) and os.path.isdir(path) and os.path.exists(path+"/__init__.py"):
            result[k] = (v, path+"/__init__.py")
    return result
    

def walk_deps(filename, processed=set()):
    with open(filename) as f:
        lines = f.readlines()
    basedir = os.path.dirname(filename)
    if basedir == "":
        basedir = "."
    local_dirs = [
        x for x in os.listdir(basedir)
        if not x.startswith(".") and os.path.isdir(os.path.join(basedir, x))
    ]
    local_files = [
        x for x in os.listdir(basedir)
        if not x.startswith(".") and os.path.splitext(x)[-1]==".py"
    ]
    #print(local_files)

    ids = filter_local(filter_imports(lines), local_dirs=local_dirs + local_files, basedir=basedir)
    ids_={ k: (package, path) 
        for k, (package, path) in ids.items()
        if not path in processed}
    #if len(ids_) < 1:
    lines = [ l for i, l in enumerate(lines) if not i in ids ]
    yield filename, lines, ids_
    
    lines = [ l for i, l in enumerate(lines) if not i in ids ]
    paths = set()
    for k, (package, path) in ids_.items():
        if path in processed:
            continue
        paths.add(path)
    new_processed = set()
    for path in paths:
        new_processed.add(path)
        for w in walk_deps(path, {*processed, *new_processed}):
            yield w


def make_graph(start_file="../runner.py"):
    data = []
    nodes = dict()
    node_names = []
    G = nx.DiGraph()
    for file, lines, deps in walk_deps(start_file):
        dependencies = set([dep for i, (package, dep) in deps.items()])
        if file not in nodes:
            nodes[file] = len(nodes)
            node_names.append(file)
        for d in dependencies:
            nodes[d] = len(nodes)
            node_names.append(d)
        if nodes[file] in G.nodes:
            G.nodes[nodes[file]]['lines'] = lines
        else:
            G.add_node(nodes[file], lines=lines)
        for d in dependencies:
            if not nodes[d] in G.nodes:
                G.add_node(nodes[d])
            e = (nodes[d], nodes[file])
            if not e in G.edges:
                G.add_edge(*e)
    return G


class DepGraph:
    def __init__(self, mainpy):
        self.graph = make_graph(mainpy)

    def sorted_files(self):
        for i in nx.topological_sort(self.graph):
            lines = self.graph.nodes[i].get('lines', [])
            if len(lines) < 1:
                continue
            yield lines
    

def read_file(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    return lines

def wrap2cell(data, ctype="code"):
    return {
        "cell_type": ctype,
        "execution_count": 0,
        "metadata": {},
        "outputs": [],
        "source": data
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mainpy", help="path or name of main file which contains runnable code")
    parser.add_argument("savepath", help="path where .ipynb notebook should be saved")
    args = parser.parse_args()

    graph = DepGraph(args.mainpy)
    for lines in graph.sorted_files():
        TEMPLATE['cells'].append(wrap2cell(lines))

    with open(args.savepath, 'w') as f:
        json.dump(TEMPLATE, f, indent=2)