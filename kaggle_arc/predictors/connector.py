import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpy as np
from collections import OrderedDict
from itertools import product

from predictors.basic import *

from base.field import *
from base.iodata import *
from utils import *
from constants import *


class Colorizer:
    def __init__(self):
        self.colors = []
        # self.position = 0
        self.info = []
        self.path = None
        self.start_color = None

    def copy(self):
        newc = Colorizer()
        newc.colors = self.colors.copy()
        newc.info = self.info.copy()
        newc.path = self.path
        newc.start_color = self.start_color
        return newc

    def add_start_color(self, start, color):
        self.start_color = (start, color)

    def add_color(self, move, color):
        self.colors.append(color)
        self.info.append((move))

    def change_palette(self, start):
        # print(start)
        return self
        if len(self.colors) == 0:
            return self
        newc = Colorizer()
        newc.path = self.path
        h, w = self.path.start.shape
        if self.path.start[h // 2, w // 2] == self.colors[0]:
            colormap = {
                xo: xi
                for lo, li in zip(self.path.start, start)
                for xo, xi in zip(lo, li)
            }
            newc.colors = [colormap.get(c, c) for c in self.colors]
            return newc
        return self

    def __iter__(self):
        if self.path.dx == 0 and self.path.dy == 0:
            while True:
                yield self.start_color[1]
            return
        position = 0
        if len(self.colors) == 0:
            raise StopIteration
        while True:
            yield self.colors[position]
            position = (position + 1) % len(self.colors)

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
        # self.total_length = 0
        self.positions = []
        self.rem = None
        self.colorizer = Colorizer()
        self.colorizer.path = self
        self.end = None

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
        if set([tuple(m.flatten()) for m in self.moves]) != set(
            [tuple(m.flatten()) for m in o.moves]
        ):
            return False
        if (self.end is None) != (o.end is None):
            return False
        if self.end is not None:
            if np.any(self.end != o.end):
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
        # new_path = self.total_length+ p.total_length
        new_path.positions = self.positions + p.positions
        return new_path

    def get_nsegments(self):
        if self.rem is None:
            return 1
        return 1 + self.rem.get_nsegments()

    def flat_positions(self):
        if self.dx == 0 and self.dy == 0:
            return [(self.x, self.y)]
        if self.rem is None:
            return self.positions
        return self.positions + self.rem.flat_positions()

    def copy(self):
        new_path = PathDescription(self.x, self.y, self.dx, self.dy)
        new_path.moves = self.moves
        new_path.start = self.start
        new_path.end = self.end
        new_path.turn_conditions = self.turn_conditions  # + p.turn_conditions
        # new_path.turn_conditions += [(self.end, p.dx, p.dy)]
        new_path.stop_conditions = (
            self.stop_conditions
        )  # + p.stop_conditions + [self.end]
        # new_path = self.total_length+ p.total_length
        new_path.positions = self.positions  # + p.positions
        new_path.colorizer = self.colorizer.copy()
        new_path.colorizer.path = new_path
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
            # new_path = self.total_length+ p.total_length
            p.positions = self.positions + p.positions
            yield p

    def get_last_pos(self):
        if self.rem is None:
            x, y = ([(self.x, self.y)] + self.positions)[-1]
        else:
            print("rem is not None")
            print(self.rem)
            x, y = self.rem.get_last_pos()
        return x, y

    def __repr__(self):
        x, y = ([(self.x, self.y)] + self.positions)[-1]
        path_line = f"Path of {len(self.positions)} steps from {(self.x, self.y)} to {(x, y)} ({self.positions}), {self.dx, self.dy}"
        if self.rem is None:
            return path_line
        path_line += "\n->\n" + str(self.rem)
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
            if dx == 0 and dy == 0:
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
        # self.kernel_size = kernel_size
        # self.hwkernel = 2*kernel_size + 1
        for p in paths:
            self.add_path(p)
        pass

    def add_path(self, path):
        start = tuple(path.start.flatten())
        if not start in self.collection:
            self.collection[start] = []
        if not path in self.collection[start]:
            # print("path already present", str(path))
            self.collection[start].append(path)

    def __repr__(self):
        result = []
        for k in self.collection:
            size = int(np.sqrt(len(k)))
            result.append(f"{np.asarray(list(k)).reshape(size, -1)}")
            for p in self.collection[k]:
                result.append(f"  {str(p)}")

                moves = "\n->\n".join([str(move) for move in p.moves])
                result.append(moves)
            result.append(str(p.end))
        return "\n".join(result)

    @classmethod
    def build(cls, paths):
        path_collection = cls(paths)
        return path_collection


class PointConnectorUtils:
    @staticmethod
    def get_conditions(
        inp,
        out,
        sx,
        sy,
        dx,
        dy,
        stop_conditions=StopConditions(),
        kernel_size=1,
        debug=False,
    ):
        inp_ = np.pad(inp, kernel_size, constant_values=-1).copy()
        out_ = np.pad(out, kernel_size, constant_values=-1).copy()
        hwkernel = 2 * kernel_size + 1
        if (
            inp_[sx + kernel_size + dx, sy + kernel_size + dy]
            == out_[sx + kernel_size + dx, sy + kernel_size + dy]
        ):
            return None
        if inp_[sx + kernel_size + dx, sy + kernel_size + dy] not in np.arange(10):
            if debug:
                print(inp_[sx + kernel_size + dx, sy + kernel_size + dy])
            return None
        path = PathDescription(sx, sy, dx, dy)
        path.start = inp_[sx : sx + hwkernel, sy : sy + hwkernel].copy()
        path.colorizer.add_start_color(
            inp_[sx + kernel_size, sy + kernel_size],
            out_[sx + kernel_size, sy + kernel_size],
        )
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
            # print(dx, dy, inp_[x + kernel_size, y + kernel_size])
            if (
                inp_[x + kernel_size, y + kernel_size] != 10
                and inp_[x + kernel_size, y + kernel_size]
                != out_[x + kernel_size, y + kernel_size]
            ):
                path.positions.append((x, y))
            else:
                break
        if len(path.positions) == 0:
            # stop_conditions.add(dx, dy, path.start)
            return None
            # path.stop_conditions.append(inp_[sx:sx + hwkernel, sy: sy + hwkernel].copy())
            # return path, inp_, out_
        for k, (x, y) in enumerate(path.positions):
            move = inp_[x : x + hwkernel, y : y + hwkernel].copy()
            if inp_[x + kernel_size, y + kernel_size] == 10:
                return None
            color = out_[x + kernel_size, y + kernel_size]
            path.colorizer.add_color(move, color)
            path.moves.append(move)
            inp_[x + kernel_size, y + kernel_size] = 10
        path.end = inp_[x : x + hwkernel, y : y + hwkernel].copy()
        # stop_conditions.add(dx, dy, path.end)
        path.moves = OrderedDict.fromkeys([tuple(m.flatten()) for m in path.moves])
        path.moves = [
            np.asarray(list(m)).reshape(hwkernel, hwkernel) for m in path.moves
        ]
        #     for dx_, dy_ in product([-1, 0, 1], [-1, 0, 1]):
        #         if dx_ == -dx and dy_ == -dy:
        #             continue
        #         if dx_ == 0 and dy_ == 0:
        #             continue
        #         new_direction = get_conditions(inp_[1:-1], out_[1:-1], x, y, dx_, dy_)
        return (
            path,
            inp_[kernel_size:-kernel_size, kernel_size:-kernel_size],
            out_[kernel_size:-kernel_size, kernel_size:-kernel_size],
        )

    @staticmethod
    def filter_paths(res):
        # r = [x for x in res]
        allpos = OrderedDict()
        for k, (score, (p, i, o)) in enumerate(res):
            if p.dx == 0 and p.dy == 0:
                positions = [(p.x, p.y)]
            else:
                positions = p.positions
            for pos in positions:
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
    def get_point_changes(
        cls,
        inp,
        out,
        x,
        y,
        stop_conditions=StopConditions(),
        kernel_size=1,
        debug=False,
    ):
        inp_ = np.pad(inp, kernel_size, constant_values=-1).copy()
        out_ = np.pad(out, kernel_size, constant_values=-1).copy()
        hwkernel = 2 * kernel_size + 1
        start = inp_[x : x + hwkernel, y : y + hwkernel]
        start_out = out_[x : x + hwkernel, y : y + hwkernel]
        if np.all(start == start_out):
            return None
        for i in range(0, start.shape[0]):
            for j in range(0, start.shape[1]):
                if i == kernel_size and j == kernel_size:
                    continue
                if start[i, j] != start_out[i, j]:
                    return None
        path = PathDescription(x, y, 0, 0)
        path.start = inp_[x : x + hwkernel, y : y + hwkernel].copy()
        path.end = out_[x : x + hwkernel, y : y + hwkernel].copy()
        path.colorizer.add_start_color(
            inp_[x + kernel_size, y + kernel_size],
            out_[x + kernel_size, y + kernel_size],
        )
        inp_[x + kernel_size, y + kernel_size] = 10
        return 0, (path, inp_[kernel_size:-kernel_size], out_[kernel_size:-kernel_size])

    @classmethod
    def get_paths_from_point(
        cls,
        inp,
        out,
        x,
        y,
        last_dx=0,
        last_dy=0,
        stop_conditions=StopConditions(),
        kernel_size=1,
        debug=False,
        recursive=False,
    ):
        if debug:
            print("get_path_from_point call")
            # print(inp, out)
            print(x, y, last_dx, last_dy)
        # if inp[x, y] not in np.arange(10):
        #    return []
        scores = []
        for dx, dy in product([-1, 0, 1], [-1, 0, 1]):
            if dx == 0 and dy == 0:
                continue
            if dx == last_dx and dy == last_dy:
                continue
            if dx == -last_dx and dy == -last_dy:
                continue
            res = cls.get_conditions(
                inp,
                out,
                x,
                y,
                dx,
                dy,
                stop_conditions=stop_conditions,
                kernel_size=kernel_size,
                debug=debug,
            )
            if debug:
                print(x, y, last_dx, last_dy, dx, dy, res)
            # path, inp, out = res
            if res is None:
                continue
            if res is not None:
                path, inp_, out_ = res
                score = path.score()
                # print("score", score)
                # if score==1:
                #    print(inp, inp_)
                # if last_dx == 0 and last_dy == 0 and score == 1 and recursive:
                #    continue
                scores.append((score, (path, inp_, out_)))
        # if len(scores) < 1:
        #     return []
        # max_score = max([s for s, _ in scores])
        # if max_score > 1 and last_dx == 0 and last_dy == 0:
        #     scores = [(s, _) for s, _ in scores if s > 1]
        #
        all_scores = cls.filter_paths(scores)
        if not recursive:
            return all_scores
        result = []

        for score, (p, i, o) in all_scores:
            x_, y_ = p.get_last_pos()
            continuations = cls.get_paths_from_point(
                i,
                o,
                x_,
                y_,
                last_dx=p.dx,
                last_dy=p.dy,
                stop_conditions=stop_conditions,
                kernel_size=kernel_size,
                debug=debug,
                recursive=True,
            )
            # print("continuations", x_, y_, len(continuations))
            if debug:
                print("continuation", x_, y_, continuations)
            if len(continuations) == 0:
                result.append((score, (p, i, o)))
            else:
                for sc, (p_, i_, o_) in continuations:
                    new_path = p.add_path(p_)
                    result.append((new_path.score(), (new_path, i_, o_)))
        return result

    @classmethod
    def find_direction(
        cls,
        inp,
        out,
        stop_conditions=StopConditions(),
        kernel_size=1,
        debug=False,
        use_recursive_lines=True,
    ):
        all_scores = []
        for i in range(inp.shape[0]):
            for j in range(out.shape[1]):
                best_scores = cls.get_paths_from_point(
                    inp,
                    out,
                    i,
                    j,
                    last_dx=0,
                    last_dy=0,
                    stop_conditions=stop_conditions,
                    kernel_size=kernel_size,
                    debug=debug,
                    recursive=use_recursive_lines,
                )
                res = cls.get_point_changes(inp, out, i, j, kernel_size=kernel_size)
                if res is not None:
                    all_scores.append(res)
                best_scores = sorted(best_scores, key=lambda x: x[0], reverse=True)
                if len(best_scores) == 0:
                    continue
                # print(best_scores)
                # max_score = best_scores[0][0]
                all_scores.extend(
                    best_scores
                )  # [x for x in best_scores if x[0] ==max_score])
        if len(all_scores) == 0:
            return None
        # all_scores = cls.filter_paths(all_scores)
        # all_scores = cls.filter_by_segments(all_scores)
        return sorted(all_scores, key=lambda x: x[0], reverse=True)

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
    def get_path_from_pos(cls, inp_, path, sx, sy, kernel_size=1):
        i = inp_.copy()
        hwkernel = 2 * kernel_size + 1
        start = tuple(i[sx : sx + hwkernel, sy : sy + hwkernel].flatten())
        path_start = tuple(path.start.flatten())
        # todo: add colorizer
        if path_start != start and not cls.compare_recolored(
            path.start, i[sx : sx + hwkernel, sy : sy + hwkernel]
        ):
            # print(path_start, start)
            return None
        x = sx
        y = sy
        positions = []
        flat_moves = [tuple(m.flatten()) for m in path.moves]
        for k in range(100):
            x += path.dx
            y += path.dy
            if x < 0 or y < 0 or x > i.shape[0] - hwkernel or y > i.shape[1] - hwkernel:
                break
            move = i[x : x + hwkernel, y : y + hwkernel]
            flat_move = tuple(move.flatten())
            if flat_move in flat_moves:  # or move[1 + path.dx, 1 + path.dy] in [
                # m[1 + path.dx, 1 + path.dy] for m in path.moves]:
                positions.append((x, y))
                i[x + kernel_size, y + kernel_size] = 10
                # print(k, move, flat_moves)
            else:
                # print(k, move)#, flat_moves)
                break
        # check path.end
        x, y = ([(sx, sy)] + positions)[-1]
        end = tuple(i[x : x + hwkernel, y : y + hwkernel].flatten())
        path_end = tuple(path.end.flatten())
        if end != path_end:
            return None
        return positions

    @classmethod
    def apply_path(cls, inp_, paths, kernel_size=1):
        data = OrderedDict()
        ext = OrderedDict()
        hwkernel = 2 * kernel_size + 1
        for i in range(inp_.shape[0] - hwkernel):
            for j in range(inp_.shape[1] - hwkernel):
                key = tuple(inp_[i : i + hwkernel, j : j + hwkernel].flatten())
                ext[(i, j)] = inp_[i : i + hwkernel, j : j + hwkernel]
                if not key in data:
                    data[key] = []
                data[key].append((i, j))
        # print(data)
        all_paths = set()
        for p in paths:
            start = tuple(p.start.flatten())
            if not start in data:
                continue
            for x, y in data[start]:
                start = inp_[x : x + hwkernel, y : y + hwkernel]
                res = cls.get_path_from_pos(inp_, p, x, y, kernel_size=kernel_size)
                if res is not None:
                    all_paths.add(
                        (tuple(sorted(res)), p.colorizer.change_palette(start))
                    )
        return all_paths

    @classmethod
    def apply_pathc(cls, inp_, paths, kernel_size=1):
        data = OrderedDict()
        ext = OrderedDict()
        hwkernel = 2 * kernel_size + 1
        for i in range(inp_.shape[0] - hwkernel):
            for j in range(inp_.shape[1] - hwkernel):
                key = tuple(inp_[i : i + hwkernel, j : j + hwkernel].flatten())
                ext[(i, j)] = inp_[i : i + hwkernel, j : j + hwkernel]
                if not key in data:
                    data[key] = []
                data[key].append((i, j))
        # print(data)
        all_paths = set()
        for start in paths.collection:
            if not start in data:
                continue
            for p in paths.collection[start]:
                # print(start)
                for x, y in data[start]:
                    start_ = inp_[x : x + hwkernel, y : y + hwkernel]
                    res = cls.get_path_from_pos(inp_, p, x, y, kernel_size=kernel_size)
                    # print(res)
                    if res is not None:
                        all_paths.add(
                            (tuple(sorted(res)), p.colorizer)
                        )  # .change_palette(start_)))
        return all_paths

    @classmethod
    def validate(cls, paths, iodata_list, kernel_size=1):
        for iodata in iodata_list:
            i = iodata.input_field.data
            o = iodata.output_field.data
            inp_ = np.pad(i, kernel_size, constant_values=-1).copy()
            out_ = np.pad(o, kernel_size, constant_values=-1).copy()

            filled_paths = cls.apply_pathc(inp_, paths, kernel_size=kernel_size)
            # print(filled_paths)
            res = np.zeros(i.shape, dtype=int)
            for path, colorizer in filled_paths:
                for x, y in path:
                    res[x, y] = 1
            if not np.all((o) * (1 - res) == i):
                return False
        return True

    @classmethod
    def predict(cls, paths, field, kernel_size=1):
        inp = field.data.copy()
        inp_ = np.pad(inp, kernel_size, constant_values=-1).copy()
        filled_paths = cls.apply_pathc(inp_, paths, kernel_size=kernel_size)
        # print(filled_paths)
        # res = #np.zeros(i.shape, dtype=int)
        for path, colorizer in filled_paths:
            for (x, y), c in zip(path, colorizer):
                inp[x, y] = c
        return inp

    @classmethod
    def extract_paths(
        cls, iodata_list, kernel_size=1, debug=False, use_recursive_lines=True
    ):
        result = []
        for iodata in iodata_list:
            i = iodata.input_field.data
            o = iodata.output_field.data
            if debug:
                print("==" * 19)
            stop_conditions = StopConditions()
            res = cls.find_direction(
                i,
                o,
                kernel_size=kernel_size,
                debug=debug,
                use_recursive_lines=use_recursive_lines,
            )
            # print("extract_", res)
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
    def make_path_collection(
        cls, iodata_list, kernel_size=1, debug=False, use_recursive_lines=True
    ):
        result = cls.extract_paths(
            iodata_list,
            kernel_size=kernel_size,
            debug=debug,
            use_recursive_lines=use_recursive_lines,
        )
        paths = [p for (score, (p, i, o)) in result]
        # print(paths)
        pc = PathCollection.build(paths)
        return pc


class PointConnectorPredictor(Predictor, AvailableAll):
    def __init__(
        self, multiplier=1, kernel_size=1, debug=False, use_recursive_lines=False
    ):
        self.multiplier = multiplier
        self.kernel_size = 1
        self.debug = debug
        self.use_recursive_lines = use_recursive_lines

    def is_available(self, iodata_list):
        for k, iodata in enumerate(iodata_list):
            if iodata.input_field.shape != iodata.output_field.shape:
                return False
        self.path_collection = PointConnectorUtils.make_path_collection(
            iodata_list,
            kernel_size=self.kernel_size,
            debug=self.debug,
            use_recursive_lines=self.use_recursive_lines,
        )
        if self.debug:
            print(self.path_collection)
        return PointConnectorUtils.validate(
            self.path_collection, iodata_list, kernel_size=self.kernel_size
        )

    def train(self, iodata_list):
        pass

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        result = PointConnectorUtils.predict(
            self.path_collection, field, kernel_size=self.kernel_size
        )
        # while True:
        yield Field(result)

    def __str__(self):
        return f"PointConnectorPredictor()"
