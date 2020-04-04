import argparse
import json
import os
import networkx as nx
import matplotlib
try:
    matplotlib.use("svg")
except:
    pass
import matplotlib.pyplot as plt

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


def filter_imports(lines, debug=False):
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
    if debug:
        print(ids)
    return ids


def filter_local(ids, local_dirs=[], basedir="..", debug=False):
    result = dict()

    for k, v in ids.items():
        path = os.path.join(basedir, v.replace(".", os.sep))
        if not (path + ".py" in local_dirs):
            continue
        
        #path = os.path.join(basedir, s)
        if os.path.exists(path + ".py"):
            result[k] = (v, path + ".py")
        if os.path.exists(path) and os.path.isdir(path) and os.path.exists(path+"/__init__.py"):
            result[k] = (v, path+"/__init__.py")
    return result
    

def walk_deps(filename, processed=set(), local_dirs=[], basedir="..", debug=False):
    with open(filename) as f:
        lines = f.readlines()
    # local_dirs = [ 
    #     os.path.join(base, f) for base, dirs, files in os.walk(basedir)
    #     for f in files
    #     if os.path.splitext(f)[-1]==".py"]

    #print(local_files)

    ids = filter_local(
        filter_imports(lines, debug=debug),
        local_dirs=local_dirs, basedir=basedir, debug=debug)
    ids_ = { k: (package, path) 
        for k, (package, path) in ids.items()
        #if not path in processed
        }
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
        for w in walk_deps(
                path, {*processed, *new_processed},
                local_dirs=local_dirs, basedir=basedir):
            yield w


def make_graph(start_file="../scripts/runner.py", basedir=".."):
    data = []
    nodes = dict()
    node_names = []
    local_dirs = [ 
        os.path.join(base, f) for base, dirs, files in os.walk(basedir)
        for f in files
        if os.path.splitext(f)[-1]==".py"]
    print(local_dirs)
    G = nx.DiGraph()
    for file, lines, deps in walk_deps(
            start_file, local_dirs=local_dirs, basedir=basedir, debug=False):

        dependencies = set([dep for i, (package, dep) in deps.items()])
        print("-"*10)
        print(file, dependencies, deps)
        if file not in nodes:
            nodes[file] = len(nodes)
            node_names.append(file)
        for d in dependencies:
            if not d in node_names:
                nodes[d] = len(nodes)
                node_names.append(d)
        if nodes[file] in G.nodes:
            G.nodes[nodes[file]]['lines'] = lines
            G.nodes[nodes[file]]['name'] = file
        else:
            G.add_node(nodes[file], lines=lines, name=file)
        for d in dependencies:
            if not nodes[d] in G.nodes:
                G.add_node(nodes[d], name=d)
            e = (  nodes[d], nodes[file])
            if not e in G.edges:
                G.add_edge(*e)
    return G


class DepGraph:
    def __init__(self, mainpy, basedir="."):
        self.graph = make_graph(mainpy, basedir=basedir)

    def sorted_files(self):
        for i in nx.topological_sort(self.graph):
            lines = self.graph.nodes[i].get('lines', [])
            name = self.graph.nodes[i].get('name', [])
            if len(lines) < 1:
                continue
            yield name, lines
    def draw(self):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos=pos)
        labels = {i: self.graph.nodes[i]['name'] for i in self.graph.nodes}
        nx.draw_networkx_labels(self.graph, pos=pos, labels=labels)
        plt.savefig("filename.png")
    

def read_file(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    return lines


def wrap2cell(data, ctype="code"):
    code_params = {
        "code": {
            "execution_count": 0,
            "outputs": []
        }
    }
    base = {
        "cell_type": ctype,
        "metadata": {},
        "source": data
    }
    return {**base, **code_params.get(ctype, {})}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mainpy", help="path or name of main file which contains runnable code")
    parser.add_argument("savepath", help="path where .ipynb notebook should be saved")
    parser.add_argument("--draw", action="store_true",
        help="use this flag to indicate that graph should be saved")
    args = parser.parse_args()

    graph = DepGraph(args.mainpy)
    if args.draw:
        graph.draw()

    for name, lines in graph.sorted_files():
        TEMPLATE['cells'].append(wrap2cell([name], ctype="markdown"))
        TEMPLATE['cells'].append(wrap2cell(lines))

    with open(args.savepath, 'w') as f:
        json.dump(TEMPLATE, f, indent=2)