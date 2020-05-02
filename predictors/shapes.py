import numpy as np

from base.field import Field
from base.iodata import IOData
from base.transformers import resize_output, crop_data

from predictors.basic import *
#Predictor, AvailableAll, AvailableWithIntMultiplier, AvailableMirror
from predictors.boosting_tree import BoostingTreePredictor

from operations.basic import Repaint
from operations.resizing import Repeater, Resizer, Fractal, Mirror
from utils import check_if_can_be_mirrored


class RepeatingPredictor(Predictor, AvailableWithIntMultiplier):
    def __init__(self, args=[], kwargs=dict()):
        #self.predictor = predictor_class(*args, **kwargs)
        pass

    def train(self, iodata_list):
        #self.predictor.train(iodata_list)
        pass

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        repeater = Repeater(self.m1, self.m2)
        result = repeater(field.data)
        yield Field(result)

    def __str__(self):
        return f"RepeatingPredictor(m1={self.m1}, m2={self.m2})"


class MirrorPredictor(Predictor, AvailableMirror):
    def __init__(self, predictor=BoostingTreePredictor):
        self.predictor = predictor()

    def train(self, iodata_list):
        self.mirror = Mirror(self.m1, self.m2, vertical=self.vertical, horizontal=self.horizontal)
        self.predictor.train(resize_output(iodata_list))
        #train_ds[i].show(predictor=predictor)

    def freeze_by_score(self, iodata_list):
        self.predictor.freeze_by_score( resize_output(iodata_list))

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        repainter = Repaint(field.data)
        for prediction in self.predictor.predict(field):
            result = self.mirror(prediction.data)
            result = repainter(result)
            yield Field(result)

    def __str__(self):
        return f"ResizingPredictor(m1={self.m1}, m2={self.m2})"


class ResizingPredictor(Predictor, AvailableWithIntMultiplier):
    def __init__(self):
        pass

    def train(self, iodata_list):
        pass

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        resizer = Resizer(self.m1, self.m2)
        result = resizer(field.data)
        yield Field(result)

    def __str__(self):
        return f"ResizingPredictor(m1={self.m1}, m2={self.m2})"


class FractalPredictor(Predictor, AvailableWithIntMultiplier):
    def __init__(self):
        pass

    def train(self, iodata_list):
        pass

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        fractal = Fractal(self.m1, self.m2)
        result = fractal(field.data)
        yield Field(result)

    def __str__(self):
        return f"FractalPredictor(m1={self.m1}, m2={self.m2})"
        

def change_colors(data, background_colors=[]):
    colormap = { c: 0 for c in background_colors }
    #colormap = dict()
    if len(np.unique(data)) == len(background_colors):
        colormap = dict()

    current_id = 1
    for line in data:
        if len(colormap) > 10:
            break
        for c in line:
            if not c in colormap:
                colormap[c] = current_id
                current_id += 1# chr(ord(current_id) + 1)
        #print(line)
    # redraw
    #print(colormap)
    data_modified = np.asarray([ [
        colormap[c] for c in line]
        for line in data
    ])
    return data_modified

def process_iodata_input(iodata, pattern, crop_func=crop_data):
    i = iodata.input_processed
    o = iodata.output_processed
    #bg = {c: 0 for c in o.data[np.where(pattern == 0)]}
    bg = dict(list(zip(*np.stack([o.data, pattern], 0).reshape(2, -1))))
    # current_id = 1
    cropped_data = crop_func(i.data)
    # #print(cropped_data, id(i.data))
    # for line in cropped_data:
    #     for x in line:
    #         if x in bg:
    #             continue
    #         bg[x] = current_id
    #         current_id += 1
    #return np.asarray([ [ bg[x] for x in line ] for line in cropped_data ])
    data = [
        [ bg.get(x, 0) for x in line ]
        for line in cropped_data
    ]
    data = Field(data)
    return iodata.reconstruct(data).data


class ConstantShaper(Predictor):
    def __init__(self):
        self.pattern = None

    def is_available(self, iodata_list):
        colormaps = [
            change_colors(iodata.output_field.data,
                background_colors=[
                    c for c in range(10)
                    if np.sum(iodata.input_field.data == c) >= np.sum(iodata.output_field.data == c)
                ])
            for iodata in iodata_list
        ]
        if len(colormaps) < 1:
            return False
        shapes = { c.shape for c in colormaps }
        if len(shapes) != 1:
            return False
        #print(colormaps, np.stack(colormaps).std())
        #print((np.unique(np.stack(colormaps))))
        if np.stack(colormaps).std(0).max() > 0 or len(np.unique(np.stack(colormaps))) == 1:
            for background_colors in [[]] + [[i] for i in range(10)]:
                colormaps = [
                    change_colors(iodata.output_field.data,
                        background_colors=background_colors)
                    for iodata in iodata_list
                ]
                #print(background_colors, colormaps)
                if len(colormaps) < 1:
                    return False
                shapes = { c.shape for c in colormaps }
                if len(shapes) != 1:
                    return False
                if np.stack(colormaps).std(0).max() <= 0 and np.stack(colormaps).std() > 0:
                    break
        # print(np.stack(colormaps).std(0))
        if np.stack(colormaps).std(0).max() > 0:
            return False
        self.pattern = colormaps[0]
        # if self.pattern.std() == 0:
        #     return False
        return True

    def train(self, iodata_list):
        self.input_pattern = None
        self.crop_func = lambda x: x
        if self.pattern.std() == 0:
             return

        for crop_func in [lambda x: x, crop_data]:
            colormap = [
                process_iodata_input(iodata, self.pattern, crop_func) for iodata in iodata_list
            ]
            #print(colormap)
            if len({ x.shape for x in colormap }) == 1:
                break
        #print({ x.shape for x in colormap }, colormap)
        if len({ x.shape for x in colormap }) == 1:
            self.crop_func = crop_func
        
            if np.stack(colormap, 0).std(0).max() <= 0.0:
                self.input_pattern = colormap[0]
        # self.input_pattern = None
        # # actual training is done in is_available method
        # for iodata in iodata_list:
        #     i = iodata.input_field
        #     o = iodata.output_field
        #     np.stack([o.data, self.pattern], 0)
        # pass
        pass

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_processed):
                yield field.reconstruct(v)
            return
        if self.input_pattern is not None:
            color_convertor = dict(set(
                list(zip(*np.stack([self.input_pattern, self.crop_func(field.processed.data)], 0).reshape(2, -1)))
            ))
            result = np.asarray([ [ color_convertor.get(x, x) for x in line ] for line in self.pattern ])
            result = Field(result)
            yield field.reconstruct(result)
            #return
        data = self.crop_func(field.processed.data)
        h, w = data.shape
        h = min(self.pattern.shape[0], h)
        w = min(self.pattern.shape[1], w)
        ss = self.pattern[:h, :w]
        background_colors = np.unique(data[np.where(ss == 0)])
        colormap = {
            i: [c for c in np.unique(data[np.where(ss == i)])
                if i == 0 or c not in background_colors]
            for i in np.unique(ss)
        }
        result = np.zeros(self.pattern.shape, dtype=np.uint8)
        result[:h, :w] = data[:h, :w]
        for key in colormap:
            if key == 0:
                continue
            value = colormap[key]
            if len(value) < 1:
                continue
            coords = np.where(self.pattern == key)
            result[coords] = value[0]
        yield field.reconstruct(Field(result))

        
    def __str__(self):
        return "ConstantShaper"