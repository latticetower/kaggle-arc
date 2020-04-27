"""

"""
import networkx as nx
import numpy as np
from xgboost import XGBClassifier

from predictors.basic import Predictor, AvailableEqualShape
from base.field import Field
from base.iodata import IOData

class GraphFeatureExtractor:
    @staticmethod
    def compare_components(GI, GO):
        if nx.number_connected_components(GI) != nx.number_connected_components(GO):
            return False
        for x, y in zip(nx.connected_components(GI), nx.connected_components(GO)):
            if len(x) != len(y):
                return False
        return True

    @staticmethod
    def get_comp_params(G):
        for x in nx.connected_components(G):
            gx = G.subgraph(x)
            nfeatures = []
            positions = set()
            ncolors = []
            for n in gx.nodes.values():
                ncolors.append(n['neighbour_colors'])
                color = n['color']
                nfeatures.append(n['features'])
                positions.add(n['pos'])
            yield {'color': color, 'features': np.stack(nfeatures, 0).sum(0), 'ncolors': np.stack(ncolors, 0).sum(0), 
                    'pos': positions, 'size': len(x) }

    @staticmethod
    def reorder(component_params_in, component_params_out):
        comp_dict = dict()
        for i, comp in enumerate(component_params_out):
            for pos in comp['pos']:
                comp_dict[pos] = i
        order = [
            comp_dict.get(list(comp['pos'])[0])
            for comp in component_params_in
        ]
        component_params_out = [component_params_out[i] for i in order]
        return component_params_in, component_params_out

    @staticmethod
    def get_data(cpi, cpo=None, use_zeros=True):
        if cpo is None:
            for gi in cpi:
                if not use_zeros and gi['color'] == 0:
                    continue
                yield gi['color'], gi['features'], gi['ncolors'], gi['size']
            return
        for gi, go in zip(cpi, cpo):
            if not use_zeros and gi['color'] == 0:
                continue
            target = gi['color'] != go['color']
            yield gi['color'], gi['features'], gi['ncolors'], gi['size'], target*1.0, go['color']

        
    @staticmethod
    def collect_graph_data(cpi, cpo=None, use_zeros=True):
        if cpo is None:
            colors, features, ncolors, sizes = list(
                zip(*GraphFeatureExtractor.get_data(cpi, use_zeros=use_zeros)))
        else:
            colors, features, ncolors, sizes, targets_bin, targets_color = list(
                zip(*GraphFeatureExtractor.get_data(cpi, cpo, use_zeros=use_zeros)))

        colors = np.asarray([[i==c for i in range(10)] for c in colors]).astype(np.float)
        features = (np.stack(features, 0) > 0)*1.0
        ncolors = (np.stack(ncolors, 0) > 0).astype(np.float)
        sizes = np.asarray(sizes).reshape(-1, 1)
        inputs = np.concatenate([colors, features, ncolors, sizes], 1)
        if cpo is None:
            return inputs
        targets = np.asarray(targets_bin)#.reshape(-1, 1)
        targets_color = np.asarray(targets_color)
        #targets_color = np.asarray([[(c == i)*1.0 for i in range(10)]for c in targets_color])
        return inputs, targets, targets_color

    @staticmethod
    def prepare_graph_features(iodata, use_zeros=True):
        GI = iodata.input_field.build_nxgraph(connectivity={i: 4 for i in range(10)})
        GO = iodata.output_field.build_nxgraph(connectivity={i: 4 for i in range(10)})
        component_params_in, component_params_out = GraphFeatureExtractor.reorder(
            list(GraphFeatureExtractor.get_comp_params(GI)),
            list(GraphFeatureExtractor.get_comp_params(GO)))
            
        inputs, targets, targets_color = GraphFeatureExtractor.collect_graph_data(
            component_params_in, component_params_out, use_zeros=use_zeros)

        return inputs, targets, targets_color
    

    @staticmethod
    def prepare_graph_features_for_eval(field, use_zeros=True):
        GI = field.build_nxgraph(connectivity={i: 4 for i in range(10)})
        graph_data = list(GraphFeatureExtractor.get_comp_params(GI))
        if not use_zeros:
            graph_data = [g for g in graph_data if g['color']!=0]
        inputs = GraphFeatureExtractor.collect_graph_data(graph_data, use_zeros=use_zeros)

        return graph_data, inputs#, targets, targets_color



class AvailableEqualShapeAndComponents():
    def is_available(self, iodata_list):
        for iodata in iodata_list:
            if iodata.input_field.shape != iodata.output_field.shape:
                return False
        for iodata in iodata_list:
            GI = iodata.input_field.build_nxgraph(connectivity={i: 4 for i in range(10)})
            GO = iodata.output_field.build_nxgraph(connectivity={i: 4 for i in range(10)})
            equal_shapes_of_components = GraphFeatureExtractor.compare_components(GI, GO)
            if not equal_shapes_of_components:
                return False
        return True


class GraphBoostingTreePredictor(Predictor, AvailableEqualShapeAndComponents):
    def __init__(self, n_estimators=10):
        self.xgb_binary =  XGBClassifier(n_estimators=n_estimators, booster="dart", n_jobs=-1)
        self.xgb =  XGBClassifier(n_estimators=n_estimators, booster="dart", n_jobs=-1,
            objective="multi:softmax", num_class=10)


    def train(self, iodata_list):
        train_x, train_y_bin, train_y = list(
            zip(*[
                    GraphFeatureExtractor.prepare_graph_features(iodata)
                    for iodata in iodata_list
                  ]))
        train_x = np.concatenate(train_x, 0)
        train_y_bin = np.concatenate(train_y_bin, 0)
        train_y = np.concatenate(train_y, 0)
        #feat, target, _ = GraphFeatureExtractor.prepare_graph_features(iodata_list)
        self.xgb_binary.fit(train_x, train_y_bin, verbose=-1)
        self.xgb.fit(train_x, train_y, verbose=-1)

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        #repainter = Repaint(field.data)
        prediction_data = np.zeros(field.shape)
        graph_data, inputs = GraphFeatureExtractor.prepare_graph_features_for_eval(field)
        preds_binary = self.xgb_binary.predict(inputs)
        preds_colors = self.xgb.predict(inputs) #.tolist()
        #result = repainter(preds).tolist()
        for comp, cbin, new_col in zip(graph_data, preds_binary, preds_colors):
            color = int(new_col) if cbin > 0.5 else comp['color']
            #if cbin > 0.5:
            #    print("new color", new_col, "old_color", comp['color'])
            for i, j in comp['pos']:
                prediction_data[i, j] = color

        yield Field(prediction_data)

    def __str__(self):
        return "GraphBoostingTreePredictor()"



class GraphBoostingTreePredictor2(Predictor, AvailableEqualShapeAndComponents):
    def __init__(self, n_estimators=10):
        #self.xgb_binary =  XGBClassifier(n_estimators=n_estimators, booster="dart", n_jobs=-1)
        #self.xgb =  XGBClassifier(n_estimators=n_estimators, booster="dart", n_jobs=-1,
        #    objective="multi:softmax", num_class=10)
        self.xgb_classifiers = []

    def is_available(self, iodata_list):
        for iodata in iodata_list:
            if iodata.input_field.shape != iodata.output_field.shape:
                return False
        components = []
        components_nonzero = []
        for iodata in iodata_list:
            GI = iodata.input_field.build_nxgraph(connectivity={i: 4 for i in range(10)})
            GO = iodata.output_field.build_nxgraph(connectivity={i: 4 for i in range(10)})
            equal_shapes_of_components = GraphFeatureExtractor.compare_components(GI, GO)
            if not equal_shapes_of_components:
                return False
            compdata = list(GraphFeatureExtractor.get_comp_params(GI))
            components.append(len(compdata))
            components_nonzero.append(len([gi for gi in compdata if gi['color']!=0]))
        self.ncomponents = -1
        #print(components, components_nonzero)
        if len(np.unique(components)) == 1:
            self.use_zeros = True
            self.ncomponents = np.unique(components)[0]
        if len(np.unique(components_nonzero))==1:
            self.use_zeros = False
            self.ncomponents = np.unique(components_nonzero)[0]
        #[GraphFeatureExtractor.prepare_graph_features(iodata)
        #    for iodata in iodata_list]))
        if self.ncomponents < 1:
            return False
        return True

    def train(self, iodata_list, n_estimators=20):
        #train_x, train_y_bin, train_y = list(
        train_sets = [[] for i in range(self.ncomponents)]
        for iodata in iodata_list:
            features, target_binary, target = GraphFeatureExtractor.prepare_graph_features(iodata, self.use_zeros)
            for i in range(self.ncomponents):
                train_sets[i].append((features[i], target_binary[i], target[i]))
        #print(len(train_sets))
        for ts in train_sets:
            features, target_binary, target = list(zip(*ts))
            features = np.stack(features, 0)
            target = np.stack(target, 0)
            xgb =  XGBClassifier(n_estimators=n_estimators, booster="dart", n_jobs=-1,
                objective="multi:softmax", num_class=10)
            xgb.fit(features, target)
            self.xgb_classifiers.append(xgb)

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        #repainter = Repaint(field.data)
        prediction_data = np.zeros(field.shape)
        #print(self.use_zeros)
        graph_data, inputs = GraphFeatureExtractor.prepare_graph_features_for_eval(field, self.use_zeros)
        predictions = []
        #print(inputs.shape, len(graph_data), len(self.xgb_classifiers))
        for i in range(inputs.shape[0]):
            xgb = self.xgb_classifiers[i]
            predictions.append(xgb.predict([inputs[i]]))
        #result = repainter(preds).tolist()
        for comp, pred in zip(graph_data, predictions):
            color = pred 
            for i, j in comp['pos']:
                prediction_data[i, j] = color

        yield Field(prediction_data)

    def __str__(self):
        return "GraphBoostingTreePredictor2()"
    
