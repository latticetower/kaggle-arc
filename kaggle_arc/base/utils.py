"""
Common helper functions
"""
import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from skimage.measure import label
from itertools import product
import numpy as np


def get_data_regions(data, connectivity=None):
    "returns distinct regions for colors 0-9"
    l = label(data, connectivity=connectivity)
    lz = label(data==0, connectivity=connectivity)
    m = np.max(l)
    lz += m
    ids = np.where(data==0)
    l[ids] = lz[ids]
    return l
    
def make_convex(r):
    mask = np.zeros(r.shape)
    mask[np.where(r)] = 1
    coords = np.argwhere(r)
    for xcoord in np.unique(coords[:, 0]):
        y = coords[np.argwhere(coords[:, 0] == xcoord), 1]
        mask[xcoord, np.min(y):np.max(y)] = 1
    for ycoord in np.unique(coords[:, 1]):
        x = coords[np.argwhere(coords[:, 1] ==ycoord), 0]
        mask[np.min(x):np.max(x), ycoord] = 1
    #print)np.unique(coords[:, 0])
    return mask == 1

def fill_region_holes(r):
    mask = np.zeros(r.shape)
    mask[np.where(r)] = 1
    coords = np.argwhere(r)
    xmin, ymin = np.min(coords, 0)
    xmax, ymax = np.max(coords, 0)
    for i in range(xmin, xmax+1):
        for j in range(ymin, ymax+1):
            x = coords[np.argwhere(coords[:, 1] == j), 0]
            if not (np.min(x) <=i and i <= np.max(x)):
                continue
            y = coords[np.argwhere(coords[:, 0] == i), 1]
            if not (np.min(y) <= j and j <= np.max(y)):
                continue
            mask[i, j] = 1
    return mask == 1


def split_interior(r, connectivity=None):
    mask = np.zeros(r.shape)
    shifts = [(i, j) for i, j in product([-1, 0, 1], [-1, 0, 1]) if not (i==0 and j==0)]
    if connectivity is not None:
        shifts = [(i, j) for (i, j) in shifts if i == 0 or j == 0]
    for x, y in np.argwhere(r):
        neighbours = [(x + i, y + j) for i, j in shifts]
        neighbours = [(i, j) for i, j in neighbours if i >= 0 and j >=0 and i<r.shape[0] and j < r.shape[1]]
        if np.any([r[i, j] != 1 for i, j in neighbours]):
            mask[x, y] = 1
    mask = mask == 1
    return mask, (~mask)*r


def get_region_params(r, connectivity=None):
    params = dict()
    maps = dict()
    for rid in np.unique(r):
        params[rid] = dict()
        maps[rid] = dict()
        region = r == rid
        m = np.argwhere(region)
        xmin, ymin = np.min(m, 0)
        xmax, ymax = np.max(m, 0)
        xmean, ymean = np.mean(m, 0)
        #print(xmin)
        params[rid]['h'] = ymax - ymin + 1
        params[rid]['w'] = xmax - xmin + 1
        params[rid]['xmin'] = xmin
        params[rid]['xmax'] = xmax
        params[rid]['ymin'] = ymin
        params[rid]['ymax'] = ymax
        params[rid]['xmean'] = int(xmean)
        params[rid]['ymean'] = int(ymean)
        conv = make_convex(region)
        
        maps[rid]['convex'] = conv
        
        params[rid]['is_convex'] = np.all(conv == region)
        no_holes = fill_region_holes(region)
        
        maps[rid]['no_holes'] = no_holes
        
        is_rectangular = no_holes[xmin:xmax+1, ymin:ymax+1].mean() == 1
        params[rid]['is_rectangular'] = is_rectangular
        params[rid]['is_square'] = is_rectangular and xmax - xmin + 1 == ymax - ymin + 1
        area = region[xmin: xmax + 1, ymin:ymax+1]
        area2 = conv[xmin: xmax + 1, ymin:ymax+1]
        
        operations = [
            lambda inp: np.fliplr(inp),
            lambda inp: np.rot90(np.fliplr(inp), 1),
            lambda inp: np.rot90(np.fliplr(inp), 2),
            lambda inp: np.rot90(np.fliplr(inp), 3),
            lambda inp: np.flipud(inp),
            lambda inp: np.rot90(np.flipud(inp), 1),
            lambda inp: np.rot90(np.flipud(inp), 2),
            lambda inp: np.rot90(np.flipud(inp), 3),
            lambda inp: np.fliplr(np.flipud(inp)),
            lambda inp: np.flipud(np.fliplr(inp))
        ]
        for i, op in enumerate(operations):
            params[rid][f"flip_{i}"] = np.all(op(area) == area)
            params[rid][f"flip_conv_{i}"] = np.all(op(area) == area)
        inner_regions = [x for x in np.unique(r[np.where(no_holes)]) if x != rid and x != 0]
        params[rid]['inner_regions'] = inner_regions
        params[rid]['holes'] = len(inner_regions)
        contour, interior = split_interior(region, connectivity=connectivity)
        
        maps[rid]['contour'] = contour
        maps[rid]['interior'] = interior
        
        params[rid]['contour_size'] = np.sum(contour)
        params[rid]['interior_size'] = np.sum(interior)
        #params[rid]['has_holes']
    return params, maps
