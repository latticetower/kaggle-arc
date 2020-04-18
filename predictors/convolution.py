import numpy as np
import torch
import torch.nn as nn
from itertools import product

from base.iodata import *
from base.field import *
from predictors.boosting_tree import get_moore_neighbours
from predictors.boosting_tree import get_tl_tr
from predictors.basic import Predictor, AvailableEqualShape


def binary_dice(a, b, eps=1.0):
    #print(a.shape)
    s = (torch.sum(a) + torch.sum(b)+ eps)
    if s != 0:
        return 2*torch.sum(a*b)/s
    return None#torch.tensor()

def dice_loss(pred, gt):
    #print(pred.shape, gt.shape)
    res = [
        binary_dice(pred[:, i], gt[:, i])
        for i in range(10)]
    res = [r for r in res if r is not None]
        
    return torch.sum(torch.stack(res))
        
        

def make_features(field, nfeat=13, local_neighb=5):
    nrows, ncols = field.shape
    #feat = np.zeros((nrows*ncols, nfeat))
    all_features = []
    cur_idx = 0
    for i in range(nrows):
        feature_list = []
        for j in range(ncols):
            color = field.data[i, j]
            features = [
                i,
                j,
                field.data[i, j]]
            features.extend(get_moore_neighbours(field, i, j, nrows, ncols))
            features.extend(get_tl_tr(field, i, j, nrows, ncols))
            features.extend([
                len(np.unique(field.data[i,:])),
                len(np.unique(field.data[:,j])),
                #next goes count of non-zero points
                np.sum(field.data[i, :] > 0),
                np.sum(field.data[:, j] > 0),
                (i+j),
                len(np.unique(field.data[
                    i-local_neighb:i+local_neighb,
                    j-local_neighb:j+local_neighb]))
            ])
            
            #feat[cur_idx,13]
            features.extend([
                (i + ncols - j - 1),
                (i + j) % 2,
                (i + j + 1) % 2,
                #(i + ncols - j - 1) % 2
                #(nrows - 1 - i + ncols - j - 1),
                #(nrows - 1 - i + j)
            ])
            features.extend([
                field.get(i + k, j + v)
                for k, v in product([-1, 0, 1], [-1, 0, 1])
            ])
            features.extend([
                field.data[nrows - 1 - i, j],
                field.data[nrows - 1 - i, ncols - 1 - j],
                field.data[i, ncols - 1 - j]
            ])
            features.extend([
                field.data[i, j] != 0,
                np.sum([ field.get(i+k, j+v) == color
                    for k, v in product([-1, 1], [-1, 1])]),
                np.sum([
                    field.get(i + 1, j) == color,
                    field.get(i - 1, j) == color,
                    field.get(i, j + 1) == color,
                    field.get(i, j - 1) == color
                ]),
                # np.sum([ field.get(i + k, j + v) == 0
                #     for k, v in product([-1, 1], [-1, 1])]),
                # np.sum([
                #     field.get(i + 1, j) == 0,
                #     field.get(i - 1, j) == 0,
                #     field.get(i, j + 1) == 0,
                #     field.get(i, j - 1) == 0
                # ])
            ])
            feature_list.append(features)
        all_features.append(feature_list)

    feat = np.asarray(all_features)
    return feat


def train_on_samples(iodata_list, cutoff=0.5, debug=False):
    model = nn.Sequential(
        nn.Conv2d(33, 128, 3, padding=1),
        nn.LeakyReLU(),
        nn.Conv2d(128, 64, 3, padding=1),
        nn.LeakyReLU(),
        nn.Conv2d(64, 32, 3, padding=1),
        nn.LeakyReLU(),
        #nn.Sigmoid(),
        nn.Conv2d(32, 10, 3, padding=1),
        #  nn.Sigmoid()
        nn.Softmax(dim=1)
    )
    loss_func = torch.nn.MSELoss()#dice_loss
    #print(net.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(20):
        model.train()
        if debug:
            print("Epoch", epoch)
        losses = []
        optimizer.zero_grad()
        #train_x, train_y, result = make_features(iodata_list)
        for iodata in iodata_list:
            features = make_features(iodata.input_field)#.reshape(iodata.input_field.shape+(-1,))
            features = np.moveaxis(features, -1, 0)
            features = features[np.newaxis, ...]
            i = torch.tensor(features).float()
            
            o = iodata.output_field.t_splitted()
            o = torch.unsqueeze(o, dim=0).float()
            p = model.forward(i)
            #print(i.is_leaf, p.is_leaf)
            #print(p.sum(1))
            #print(features.shape)
            #print(o.shape, p.shape)
            loss = loss_func(p, o)
            loss.backward()
            losses.append(loss.item())
        if debug:
            print(losses)
        #if epoch % 10 == 0:
        #    print("zero grad")   
        optimizer.step()
        
    if debug:
        print("Validation:")
    val_results = []
    model.eval()
    with torch.no_grad():
        scores = []
        for iodata in iodata_list:
            features = make_features(iodata.input_field).reshape(iodata.input_field.shape+(-1,))
            features = np.moveaxis(features, -1, 0)
            features = features[np.newaxis, ...]
            i = torch.tensor(features).float()
            
            o = iodata.output_field.t_splitted()
            o = torch.unsqueeze(o, dim=0).float()
            p = model.forward(i)
            p = torch.squeeze(p, dim=0)
            p = Field.from_splitted(p)
            score = Field.score(p, iodata.output_field)
            scores.append(score)
            val_results.append((p, iodata.input_field, iodata.output_field))
            if debug:
                print(score)
                p.show()
                iodata.output_field.show()
    scores = np.mean(scores)
    #print(scores)
    if scores < cutoff:
        return None
    return scores, model, val_results


class ConvolutionPredictor(Predictor, AvailableEqualShape):
    def __init__(self, nepochs=40, loss="mse"):
        #self.xgb =  XGBClassifier(n_estimators=25*2, booster="dart", n_jobs=-1)
        self.model = nn.Sequential(
            nn.Conv2d(33, 128, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(),
            #nn.Sigmoid(),
            nn.Conv2d(32, 10, 3, padding=1),
            #  nn.Sigmoid()
            nn.Softmax(dim=1)
        )
        if loss == "mse":
            self.loss_func = torch.nn.MSELoss()
        else:
            self.loss_func = dice_loss
        #print(net.parameters())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.nepochs = 20
        self.debug = False

    def train(self, iodata_list):
        all_losses = []
        for epoch in range(self.nepochs):
            self.model.train()
            if self.debug:
                print("Epoch", epoch)
            losses = []
            self.optimizer.zero_grad()
            #train_x, train_y, result = make_features(iodata_list)
            for iodata in iodata_list:
                features = make_features(iodata.input_field)#.reshape(iodata.input_field.shape+(-1,))
                features = np.moveaxis(features, -1, 0)
                features = features[np.newaxis, ...]
                i = torch.tensor(features).float()
                
                o = iodata.output_field.t_splitted()
                o = torch.unsqueeze(o, dim=0).float()
                p = self.model.forward(i)
                loss = self.loss_func(p, o)
                loss.backward()
                losses.append(loss.item())
            if self.debug:
                print(losses)
                
            losses = np.mean(losses)
            if len(all_losses) > 0:
                if np.mean(all_losses[-3:]) <= losses:
                    break
            all_losses.append(losses)
            
            #if epoch % 10 == 0:
            #    print("zero grad")   
            self.optimizer.step()
        
    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        self.model.eval()
        with torch.no_grad():
            features = make_features(field)
            features = np.moveaxis(features, -1, 0)
            features = features[np.newaxis, ...]
            i = torch.tensor(features).float()
            p = self.model.forward(i)
            p = torch.squeeze(p, dim=0).detach().cpu().numpy()
        yield Field.from_splitted(p)

    def __str__(self):
        return "ConvolutionPredictor()"
    
