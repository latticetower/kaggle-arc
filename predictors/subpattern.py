from xgboost import XGBClassifier

from operations.reversible import *

from predictors.basic import * #BTFeatureExtractor, BoostingTreePredictor2
from predictors.boosting_tree import *

class SubpatternMatcher:
    @staticmethod
    def get_separator_length(sequence, size):
        if len(sequence) == 0:
            return None
        if len(sequence) == 1 and size//2 == sequence[0]:
            return sequence[0]
        w = sequence[0]
        if w <= 1 or w > size//2:
            return None
        xlast = w
        for x in sequence[1:]:
            if x != xlast + w + 1:
                return None
        return w

    @staticmethod
    def get_separating_lines(i):
        for c in np.unique(i.flatten()):
            bmap = i == c
            s0 = bmap.std(0) == 0
            s1 = bmap.std(1) == 0
            s0 = bmap.all(0)*s0
            s1 = bmap.all(1)*s1

            #print(bmap[:, s0==0])
            s0 = np.argwhere(s0).flatten()
            s1 = np.argwhere(s1).flatten()
            r0 = SubpatternMatcher.get_separator_length(s0, i.shape[1])
            r1 = SubpatternMatcher.get_separator_length(s1, i.shape[0])
            if r0 is None and r1 is None:
                #yield c, i.shape[0], i.shape[1]
                continue
            if r0 is None:
                yield c, r1, i.shape[1]
                continue
            if r1 is None:
                yield c, i.shape[0], r0
                continue
            yield c, r1, r0

    @staticmethod
    def get_availability_param(iodata):
        i = iodata.input_field.data
        o = iodata.output_field.data
        isep = {(h, w): c for c, h, w in SubpatternMatcher.get_separating_lines(i)}
        osep = {(h, w): c for c, h, w in SubpatternMatcher.get_separating_lines(o)}

        common_areas = isep.keys() & osep.keys()
        if len(common_areas) < 1:
            return None
        return { k: (isep[k], osep[k]) for k in common_areas }

    @staticmethod
    def process_iodata_list(iodata_list):
        all_params = []
        total = set()
        for t in iodata_list:
            param = SubpatternMatcher.get_availability_param(t)
            if param is None:
                return set(), []
            for k in param.keys():
                total.add(k)
            all_params.append(param)
        sizes = {k for k in total if np.all([k in x for x in all_params])}
        all_params = [{k: x[k] for k in sizes} for x in all_params]
        return sizes, all_params


class SubpatternMatcherPredictor(Predictor):

    def __init__(self):
        self.xgb = XGBClassifier(n_estimators=10, booster="dart", n_jobs=-1,
            objective="multi:softmax", num_class=10)
        pass

    def is_available(self, iodata_list):
        for iodata in iodata_list:
            if iodata.input_field.shape != iodata.output_field.shape:
                return False
            #m1 = iodata.output_field.shape # iodata.input_field.height // iodata.output_field.height
            #m2 = iodata.output_field.width  # iodata.input_field.width // iodata.output_field.width
            #all_sizes.add((m1, m2))
        sizes, params = SubpatternMatcher.process_iodata_list(iodata_list)
        if len(sizes) < 1:
            return False
        self.sizes = sizes
        self.params = params
        (h, w) = list(sizes)[0]
        self.op = WrappedOperation(
            ReversibleSplit((h, w), hsep=1, wsep=1, outer_sep=False, splitter_func=split_by_shape),
            ReversibleCombine((h, w), hsep=1, wsep=1, outer_sep=False, sep_color=5, splitter_func=split_by_shape)
        )
        #self.op.train(iodata_list)
        return True
    
    def train(self, iodata_list):
        all_samples = []
        self.op.train(iodata_list)
        for iodata in iodata_list:
            i, o = self.op.wrap(iodata)
            all_samples.append((i, o))
        all_samples = [
            (xi, xo)
            for (i, o) in all_samples
            for xi, xo in zip(i.flat_iter(), o.flat_iter())
            #for li, lo in zip(i, o)
            #for xi, xo in zip(li, lo)
        ]
        #print(all_samples)
        feat, target, _ = BTFeatureExtractor.get_features(all_samples)
        # print(feat.shape, target.shape)
        self.xgb.fit(feat, target, verbose=-1)

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        #repainter = Repaint(field.data)
        
        feature_field, postprocess = self.op.run(field)
        #print(feature_field)
        def predict_on_subfield(x):
            nrows, ncols = x.shape
            feat = BTFeatureExtractor.make_features(x)
            preds = self.xgb.predict(feat).reshape(nrows, ncols)
            preds = preds.astype(int)#.tolist()
            #print(x.data)
            return Field(preds)

        lines = feature_field.map(predict_on_subfield)
        result = postprocess(lines)
        yield result
        pass