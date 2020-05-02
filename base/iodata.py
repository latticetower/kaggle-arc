import json
import matplotlib
from matplotlib import cm
import matplotlib.gridspec as gridspec
import seaborn as sns
from itertools import islice

from base.field import *


class IOData:
    __slots__ = ["input_field", "output_field"]
    def __init__(self, data=None, input_field=None, output_field=None):
        #data['train'][0]['input']
        self.input_field = input_field
        self.output_field = output_field
        if data is not None:
            if 'input' in data:
                self.input_field = Field(data['input'])
            if 'output' in data:
                self.output_field = Field(data['output'])

    def show(self, fig=None, axes=None, predictor=None, npredictions=1):
        if fig is None:
            if predictor is not None:
                fig, axes = plt.subplots(nrows=1, ncols=2+npredictions)
            else:
                fig, axes = plt.subplots(nrows=1, ncols=2)
        ax0, ax1 = axes[:2]
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax1.set_xticks([])
        ax1.set_yticks([])
        if self.input_field is not None:
            self.input_field.show(ax0, label="input")
        #ax0.axis("off")
        if self.output_field is not None:
            self.output_field.show(ax1, label="output")
        #ax1.axis("off")
        if predictor is not None:
            for i, prediction in enumerate(
                    islice(predictor.predict(self.input_field), npredictions)):
                ax = axes[2 + i]
                ax.set_xticks([])
                ax.set_yticks([])
                prediction.show(ax)
                #ax.axis("off")
    def t(self):
        result = [self.input_field.t()]
        if self.output_field is not None:
            result.append(self.output_field.t())
        return tuple(result)

    def t_splitted(self):
        result = [self.input_field.t_splitted()]
        if self.output_field is not None:
            result.append(self.output_field.t_splitted())
        return tuple(result)


class Sample:
    __slots__ = ["name", "train", "test"]
    
    def __init__(self, name, path):
        with open(path) as f:
            data = json.load(f)
        self.name = name
        self.train = [
            IOData(sample)
            for sample in data.get('train', [])]
        self.test = [
            IOData(sample)
            for sample in data.get('test', [])
        ]

    # @property
    # def train(self):
    #     return self._train
    # 
    # @property
    # def test(self):
    #     return self._test
    # 
    # @property
    # def name(self):
    #     return self._name
    
    def predict(self, predictors):
        predictions = []
        for sample in self.iterate_test():
            pred = [
                predictor.predict(sample)
                for predictor in predictors
            ]
            predictions.append(pred)
        return predictions

    def show(self, fig=None, grids=[None, None, None], w=2, h=2, ncols=2, predictor=None, npredictions=1, title=""):
        ntrain = len(self.train)
        ntest = len(self.test)
        ncols += npredictions
        if predictor is not None:
            if not predictor.is_available(self.train):
                predictor=None
        gs, train_gs, test_gs = grids
        if fig is None:
            fig = plt.figure(figsize=(ncols * w, (ntrain + ntest) * h))
            plt.title(title)
            ax = plt.gca()
            for edge, spine in ax.spines.items():
                spine.set_visible(False)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            # ax = plt.gca()
            # fig.colorbar(
            #     cm.ScalarMappable(
            #         norm=matplotlib.colors.Normalize(vmin=0, vmax=10),
            #         cmap=sns.cubehelix_palette(10, as_cmap=True)), ax=ax)
            #plt.gca().axis('off')
        if gs is None:
            gs = gridspec.GridSpec(ntrain + ntest, 1, figure=fig)
        if train_gs is None:
            train_gs = gridspec.GridSpecFromSubplotSpec(ntrain, ncols, subplot_spec=gs[:ntrain])
        if test_gs is None:
            test_gs = gridspec.GridSpecFromSubplotSpec(ntest, ncols, subplot_spec=gs[ntrain:])

        if train_gs is not None:
            for i in range(ntrain):
                ax0 = fig.add_subplot(train_gs[i, 0])
                ax1 = fig.add_subplot(train_gs[i, 1])
                self.train[i].show(fig=fig, axes=[ax0, ax1])
                if predictor is not None:
                    preds = islice(predictor.predict(self.train[i].input_field), npredictions)
                    for k, prediction in enumerate(preds):
                        ax = fig.add_subplot(train_gs[i, k + 2])
                        ax.set_xticks([])
                        ax.set_yticks([])
                        dice = Field.score(prediction, self.train[i].output_field)
                        prediction.show(ax, label=f"{dice:1.4f}")

        if test_gs is not None:
            for i in range(ntest):
                ax0 = fig.add_subplot(test_gs[i, 0])
                ax1 = fig.add_subplot(test_gs[i, 1])
                npredictions=1
                #pred_ax = [fig.add_subplot(test_gs[i, 2+k]) for k in range(npredictions)]
                self.test[i].show(fig=fig, axes=[ax0, ax1])
                #predictor=predictor, npredictions=npredictions)
                if predictor is not None:
                    preds = islice(predictor.predict(self.test[i].input_field), npredictions)
                    for k, prediction in enumerate(preds):
                        ax = fig.add_subplot(test_gs[i, k + 2])
                        ax.set_xticks([])
                        ax.set_yticks([])
                        if self.test[i].output_field is not None:
                            dice = Field.score(prediction, self.test[i].output_field)
                            dice = f"{dice:1.4f}"
                        else:
                            dice = "-"
                        prediction.show(ax, label=dice)
        
            
                
            