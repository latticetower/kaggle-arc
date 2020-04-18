"""
Functions for subpattern extraction
"""

import numpy as np

def get_suffixes(s, wildcard=0):
    suffix_len = 0
    m = len(s)
    suffixes = [0 for i in range(m)]
    i=1
    while i < m:
        if s[i] == wildcard or s[suffix_len] == wildcard or s[i] == s[suffix_len]:
            suffix_len += 1
            suffixes[i] = suffix_len
            i += 1
        elif suffix_len!=0:
            suffix_len = suffixes[suffix_len-1]
        else:
            suffixes[i] = 0
            i+=1
    return suffixes

def get_repeat_length(suffixes):
    n = len(suffixes)
    k = suffixes[-1]
    if k < n - k:
        return n
    return n - k

def check_subpattern(data, r, c, wildcard=0):
    for line in data:
        condition = np.all([
            x == y or x==wildcard or y==wildcard
            for x, y in zip(line, line[c:])
        ])
        if not condition:
            return False
    for line in data.T[:c]:
        condition = np.all([
            x == y or x==wildcard or y==wildcard
            for x, y in zip(line, line[c:])
        ])
        if not condition:
            return False
    return True

def get_subpattern(data, wildcard=0, check_passed=True):
    repeats = []
    for line in data:
        s = get_suffixes(line, wildcard)
        r = get_repeat_length(s)
        repeats.append(r)
    #print(repeats)
    if check_passed:
        col = int(np.median(repeats))
    else:
        col = np.lcm.reduce(repeats)
    #print(col)
    crepeats=[]
    if check_passed:
        subset = data.T
    else:
        subset = data.T[:col]
    for line in subset:
        s = get_suffixes(line, wildcard)
        r = get_repeat_length(s)
        if check_passed:
            if r == len(s):
                continue
        crepeats.append(r)
    #print(crepeats)
    if check_passed:
        if len(crepeats) == 0:
            row = len(data)
        else:
            row = int(np.median(crepeats))
    else:
        row = np.lcm.reduce(crepeats)
    return row, col