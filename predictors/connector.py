import numpy as np
from collections import OrderedDict
from itertools import product

from predictors.basic import *

from base.field import *
from base.iodata import *
from utils import *
from constants import *




class Colorizer():
    def __init__(self):
        self.colors = []
        #self.position = 0
        self.info = []
        self.path=None
        pass
    
    def add_color(self, move, color):
        self.colors.append(color)
        self.info.append((move))
        
    def change_palette(self, start):
        #print(start)
        return self
        if len(self.colors) == 0:
            return self
        newc = Colorizer()
        newc.path = self.path
        
        if self.path.start[1, 1] == self.colors[0]:
            colormap = {xo: xi for lo, li in zip(self.path.start, start)
                for xo, xi in zip(lo, li)}
            newc.colors = [ colormap.get(c, c) for c in self.colors]
            return newc
        return self
        
    def __iter__(self):
        position = 0
        if len(self.colors) == 0:
            raise StopIteration
        while True:
            yield self.colors[position]
            position = (position + 1)%len(self.colors)
    def __repr__(self):
        return str(self.colors)
    

class PathDescription:
    def __init__(self, x, y, dx, dy):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.start = None
        self.moves = []
        self.turn_conditions = []
        self.stop_conditions = []
        #self.total_length = 0
        self.positions = []
        self.rem = None
        self.colorizer = Colorizer()
        self.colorizer.path = self
        
    def __eq__(self, o):
        if not (self.dx == o.dx and self.dy == o.dy):
            return False
        if self.start is not None and o.start is None:
            return False
        if self.start is None and o.start is not None:
            return False
        if self.start is None and o.start is None:
            return True
        if np.any(self.start != o.start):
            return False
        if set([tuple(m.flatten()) for m in self.moves])!= set([tuple(m.flatten()) for m in o.moves]):
            return False
        if np.any(self.end!=o.end):
            return False
        if self.rem is None and o.rem is not None:
            return False
        if self.rem is not None and o.rem is None:
            return False
        if self.rem is None:
            return True
        return self.rem == o.rem
        return True
        
    def score(self):
        if self.rem is None:
            return len(self.positions)
        return len(self.positions) + self.rem.score()

    def merge(self, p):
        new_path = PathDescription(self.x, self.y, self.dx, self.dy)
        new_path.moves = self.moves + p.moves
        new_path.start = self.start
        new_path.end = p.end
        new_path.turn_conditions = self.turn_conditions + p.turn_conditions
        new_path.turn_conditions += [(self.end, p.dx, p.dy)]
        new_path.stop_conditions = self.stop_conditions + p.stop_conditions + [self.end]
        #new_path = self.total_length+ p.total_length
        new_path.positions = self.positions + p.positions
        return new_path

    def get_nsegments(self):
        if self.rem is None:
            return 1
        return 1+self.rem.get_nsegments()

    def flat_positions(self):
        if self.rem is None:
            return self.positions
        return self.positions + self.rem.flat_positions()

    def copy(self):
        new_path = PathDescription(self.x, self.y, self.dx, self.dy)
        new_path.moves = self.moves
        new_path.start = self.start
        new_path.end = self.end
        new_path.turn_conditions = self.turn_conditions #+ p.turn_conditions
        #new_path.turn_conditions += [(self.end, p.dx, p.dy)]
        new_path.stop_conditions = self.stop_conditions #+ p.stop_conditions + [self.end]
        #new_path = self.total_length+ p.total_length
        new_path.positions = self.positions # + p.positions
        return new_path
    
    def add_path(self, p):
        new_path = self.copy()
        new_path.rem = p.copy()
        return new_path
    
    def add_paths(self, paths):
        for p in paths:
            new_path = PathDescription(self.x, self.y, self.dx, self.dy)
            new_path.moves = self.moves + p.moves
            new_path.turn_conditions = self.turn_conditions + p.turn_conditions
            new_path.stop_conditions = self.stop_conditions + p.stop_conditions
            #new_path = self.total_length+ p.total_length
            p.positions = self.positions + p.positions
            yield p

    def get_last_pos(self):
        x, y = ([(self.x, self.y)] + self.positions)[-1]
        return x, y

    def __repr__(self):
        x, y = ([(self.x, self.y)] + self.positions)[-1]
        path_line = f"Path of {len(self.positions)} steps from {(self.x, self.y)} to {(x, y)} ({self.positions}), {self.dx, self.dy}"
        if self.rem is None:
            return path_line
        path_line+= "\n->\n"+str(self.rem)
        return path_line


class StopConditions:
    def __init__(self):
        self.info = OrderedDict()
        pass
    def add(self, dx, dy, cell):
        if not (dx, dy) in self.info:
            self.info[(dx, dy)] = set()
        self.info[(dx, dy)].add(tuple(cell.flatten()))
    def add_square(self, cell):
        for dx, dy in product([-1, 0, 1], [-1, 0, 1]):
            if dx == 0 and dy==0:
                continue
            self.add(dx, dy, cell)

    def is_stop(self, dx, dy, cell):
        if not (dx, dy) in self.info:
            return False
        i = tuple(cell.flatten())
        return i in self.info[(dx, dy)]
    def __repr__(self):
        return str(self.info)



class PathCollection:
    def __init__(self, paths):
        self.collection = OrderedDict()
        for p in paths:
            self.add_path(p)
        pass
    def add_path(self, path):
        start = tuple(path.start.flatten())
        if not start in self.collection:
            self.collection[start] = []
        if not path in self.collection[start]:
            #print("path already present", str(path))
            self.collection[start].append(path)
    def __repr__(self):
        result = []
        for k in self.collection:
            result.append(f"{np.asarray(list(k)).reshape(3,3)}")
            for p in self.collection[k]:
                result.append(f"  {str(p)}")

                moves = "\n->\n".join([str(move) for move in p.moves ])
                result.append(moves)
            result.append(str(p.end))
        return "\n".join(result)
    
    @classmethod
    def build(cls, paths):
        path_collection = cls(paths)
        return path_collection


class PointConnectorUtils:
    @staticmethod
    def get_conditions(inp, out, sx, sy, dx, dy, stop_conditions=StopConditions()):
        inp_ = np.pad(inp, 1, constant_values=-1).copy()
        out_ = np.pad(out, 1, constant_values=-1).copy()
        path = PathDescription(sx, sy, dx, dy)
        path.start = inp_[sx:sx+3, sy: sy+3].copy()
        if stop_conditions.is_stop(dx, dy, path.start):
            return None
    #     moves = []
    #     positions = []
    #     turn_conditions = []
    #     stop_conditions = []
        x = sx
        y = sy
        for step in range(100):
            x += dx
            y += dy
            if (inp_[x + 1, y + 1] != 10 and inp_[x + 1, y + 1] != out_[x + 1, y + 1]):
                path.positions.append((x, y))
            else:
                break
        if len(path.positions) == 0:
            #stop_conditions.add(dx, dy, path.start)
            return None
            path.stop_conditions.append(inp_[sx:sx+3, sy: sy+3].copy())
            return path, inp_, out_
        for k, (x, y) in enumerate(path.positions):
            move = inp_[x: x+3, y:y+3].copy()
            if inp_[x+1, y+1] == 10:
                return None
            color = out_[x+1, y+1]
            path.colorizer.add_color(move, color)
            path.moves.append(move)
            inp_[x+1, y+1] = 10
        path.end = inp_[x: x+3, y:y+3].copy()
        #stop_conditions.add(dx, dy, path.end)
        path.moves = OrderedDict.fromkeys([tuple(m.flatten()) for m in path.moves])
        path.moves = [np.asarray(list(m)).reshape(3,3) for m in path.moves]
        #     for dx_, dy_ in product([-1, 0, 1], [-1, 0, 1]):
        #         if dx_ == -dx and dy_ == -dy:
        #             continue
        #         if dx_ == 0 and dy_ == 0:
        #             continue
        #         new_direction = get_conditions(inp_[1:-1], out_[1:-1], x, y, dx_, dy_)
        return path, inp_[1:-1, 1:-1], out_[1:-1, 1:-1]

    @staticmethod
    def filter_paths(res):
        #r = [x for x in res]
        allpos = OrderedDict()
        for k, (score, (p, i, o)) in enumerate(res):
            for pos in p.positions:
                if not pos in allpos:
                    allpos[pos] = [k]
                else:
                    l = allpos[pos][0]
                    if res[l][0] == score:
                        allpos[pos].append(k)
                    elif res[l][0] < score:
                        allpos[pos] = [k]
        paths = set([x for xs in allpos.values() for x in xs])
        return [res[k] for k in paths]

    
    @staticmethod
    def filter_by_segments(res):
        allpos = OrderedDict()
        for k, (score, (p, i, o)) in enumerate(res):
            pos = tuple(sorted(set(p.flat_positions())))
            if pos not in allpos:
                allpos[pos] = []
            n = (p.get_nsegments(), k)
            allpos[pos].append(n)
        for k in allpos:
            nseg = min([x[0] for x in allpos[k]])
            allpos[k] = [v for u, v in allpos[k] if u == nseg]
            
        return [res[k] for ks in allpos.values() for k in ks]


    @classmethod
    def get_paths_from_point(cls, inp, out, x, y, last_dx=0, last_dy=0, stop_conditions=StopConditions()):
        if inp[x, y] not in np.arange(10):
            return []
        scores = []
        for dx, dy in product([-1, 0, 1], [-1, 0, 1]):
            if dx==0 and dy == 0:
                continue
            if dx == last_dx and dy == last_dy:
                continue
            if dx == -last_dx and dy == -last_dy:
                continue
            res = cls.get_conditions(inp, out, x, y, dx, dy, stop_conditions=stop_conditions)
            #path, inp, out = res
            if res is None:
                continue
            if res is not None:
                path, inp_, out_ = res
                score = len(path.positions)
                scores.append((score, (path, inp_, out_)))
        all_scores = cls.filter_paths(scores)
        result = []
        for score, (p, i, o) in all_scores:
            x_, y_ = p.get_last_pos()
            continuations = cls.get_paths_from_point(i, o, x_, y_, last_dx=p.dx, last_dy=p.dy, stop_conditions=stop_conditions)
            
            if len(continuations) == 0:
                result.append((score, (p, i, o)))
            else:
                for sc, (p_, i_, o_) in continuations:
                    new_path = p.add_path(p_)
                    result.append((new_path.score(), (new_path, i_, o_)))
        return result
    
    @classmethod
    def find_direction(cls, inp, out, stop_conditions=StopConditions()):
        all_scores = []
        for i in range(inp.shape[0]):
            for j in range(out.shape[1]):
                best_scores = cls.get_paths_from_point(
                    inp, out, i, j, last_dx=0, last_dy=0, stop_conditions=stop_conditions)
                
                best_scores = sorted(best_scores, key=lambda x:x[0], reverse=True)
                if len(best_scores) == 0:
                    continue
                #print(best_scores)
                #max_score = best_scores[0][0]
                all_scores.extend(best_scores)#[x for x in best_scores if x[0] ==max_score])
        if len(all_scores) == 0:
            return None
        all_scores = cls.filter_paths(all_scores)
        all_scores = cls.filter_by_segments(all_scores)
        return sorted(all_scores, key=lambda x:x[0], reverse=True)
        
    @staticmethod
    def compare_recolored(a, b):
        ab = dict()
        ba = dict()
        for k, v in zip(a.flatten(), b.flatten()):
            if k in ab:
                return False
            else:
                ab[k] = v
            if v in ba:
                return False
            else:
                ba[v] = k
        return True

    @classmethod
    def get_path_from_pos(cls, inp_, path, sx, sy):
        i = inp_.copy()
        start = tuple(i[sx:sx+3, sy:sy+3].flatten())
        path_start = tuple(path.start.flatten())
        #todo: add colorizer
        if path_start != start and not cls.compare_recolored(path.start, i[sx:sx+3, sy:sy+3]):
            print(path_start, start)
            return None
        x = sx
        y = sy
        positions = []
        flat_moves = [tuple(m.flatten())for m in path.moves]
        for k in range(100):
            x += path.dx
            y += path.dy
            if x < 0 or y < 0 or x > i.shape[0]-3 or y > i.shape[1]-3:
                break
            move = i[x:x+3, y:y+3]
            flat_move = tuple(move.flatten())
            if flat_move in flat_moves: #or move[1 + path.dx, 1 + path.dy] in [
                #m[1 + path.dx, 1 + path.dy] for m in path.moves]:
                positions.append((x, y))
                i[x + 1, y + 1] = 10
                #print(k, move, flat_moves)
            else:
                #print(k, move)#, flat_moves)
                break
        # check path.end
        x, y = ([(sx, sy)] + positions)[-1]
        end = tuple(i[x:x + 3, y:y + 3].flatten())
        path_end = tuple(path.end.flatten())
        if end != path_end:
            return None
        return positions

    @classmethod  
    def apply_path(cls, inp_, paths):
        data = OrderedDict()
        ext = OrderedDict()
        for i in range(inp_.shape[0] - 3):
            for j in range(inp_.shape[1] - 3):
                key = tuple(inp_[i:i + 3, j:j + 3].flatten())
                ext[(i, j)] = inp_[i:i + 3, j:j + 3]
                if not key in data:
                    data[key] = []
                data[key].append((i, j))
        #print(data)
        all_paths = set()
        for p in paths:
            start = (tuple(p.start.flatten()))
            if not start in data:
                continue
            for x, y in data[start]:
                start = inp_[x : x + 3, y : y + 3]
                res = cls.get_path_from_pos(inp_, p, x, y)
                if res is not None:
                    all_paths.add((tuple(sorted(res)), p.colorizer.change_palette(start)))
        return all_paths

    @classmethod
    def apply_pathc(cls, inp_, paths):
        data = OrderedDict()
        ext = OrderedDict()
        for i in range(inp_.shape[0]-3):
            for j in range(inp_.shape[1]-3):
                key = tuple(inp_[i:i+3, j:j+3].flatten())
                ext[(i, j)] = inp_[i:i+3, j:j+3]
                if not key in data:
                    data[key] = []
                data[key].append((i, j))
        #print(data)
        all_paths = set()
        for start in paths.collection:
            if not start in data:
                continue
            for p in paths.collection[start]:
                #print(start)
                for x, y in data[start]:
                    start_ = inp_[x:x+3, y:y+3]
                    res = cls.get_path_from_pos(inp_, p, x, y)
                    #print(res)
                    if res is not None:
                        all_paths.add((tuple(sorted(res)), p.colorizer))#.change_palette(start_)))
        return all_paths

    @classmethod
    def validate(cls, paths, iodata_list):
        for iodata in iodata_list:
            i = iodata.input_field.data
            o = iodata.output_field.data
            inp_ = np.pad(i, 1, constant_values=-1).copy()
            out_ = np.pad(o, 1, constant_values=-1).copy()

            filled_paths = cls.apply_pathc(inp_, paths)
            #print(filled_paths)
            res = np.zeros(i.shape, dtype=int)
            for path, colorizer in filled_paths:
                for (x, y) in path:
                    res[x, y] = 1
            if not np.all((o)*(1 - res)==i):
                return False
        return True

    @classmethod
    def predict(cls, paths, field):
        inp = field.data.copy()
        inp_ = np.pad(inp, 1, constant_values=-1).copy()
        filled_paths = cls.apply_pathc(inp_, paths)
        #print(filled_paths)
        #res = #np.zeros(i.shape, dtype=int)
        for path, colorizer in filled_paths:
            for (x, y), c in zip(path, colorizer):
                inp[x, y] = c
        return inp

    @classmethod
    def extract_paths(cls, iodata_list, debug=False):
        result = []
        for iodata in iodata_list:
            i = iodata.input_field.data
            o = iodata.output_field.data
            if debug:
                print("=="*19)
            stop_conditions = StopConditions()
            res = cls.find_direction(i, o)
            if res is None:
                if debug:
                    print(None)
                continue
            res = cls.filter_paths(res)
            for score, (path, inp_, out_) in res:
                result.append((score, (path, inp_, out_)))
                if debug:
                    if score == 0:
                        continue
                    print(score)
                    print(path)
                    print(inp_)
                    print(out_)
        return result

    @classmethod
    def make_path_collection(cls, iodata_list):
        result = cls.extract_paths(iodata_list)
        paths = [p for (score, (p, i, o)) in result]
        pc = PathCollection.build(paths)
        return pc


class PointConnectorPredictor(Predictor, AvailableAll):
    def __init__(self, multiplier=1):
        self.multiplier = multiplier

    def is_available(self, iodata_list):
        for iodata in iodata_list:
            if iodata.input_field.shape != iodata.output_field.shape:
                return False
        self.path_collection = PointConnectorUtils.make_path_collection(iodata_list)
        return PointConnectorUtils.validate(self.path_collection, iodata_list)
        
    def train(self, iodata_list):
        pass

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        result = PointConnectorUtils.predict(self.path_collection, field)
        #while True:
        yield Field(result)

    def __str__(self):
        return f"PointConnectorPredictor()"
