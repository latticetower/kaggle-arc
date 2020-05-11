import numpy as np
import torch
import torch.nn as nn
from itertools import product

from skimage.measure import label

from base.iodata import *
from base.field import *
from predictors.boosting_tree import BTFeatureExtractor
from predictors.basic import *



def filter_ones(col, split_count=1):
    coords = np.argwhere(col > 0).flatten()
    if len(coords) == 0:
        return col, col
    split0 = []
    split1 = []
    last_seq = []
    for c in coords:
        if len(last_seq) < 1:
            last_seq.append(c)
            continue
        if last_seq[-1] + 1 == c:
            last_seq.append(c)
            continue
        if len(last_seq) <= split_count:
            split0.extend(last_seq)
        else:
            split1.extend(last_seq)
        last_seq = [c]
    if len(last_seq) > 0:
        if len(last_seq) <= split_count:
            split0.extend(last_seq)
        else:
            split1.extend(last_seq)
    s0 = np.zeros(col.shape)
    s0[split0] = 1
    s1 = np.zeros(col.shape)
    s1[split1] = 1
    return s0*col, s1*col

def split_coords(data, color, split_count=1):
    col = np.sum(d.data == color, 0)
    row = np.sum(d.data == color, 1)

    col0, col1 = filter_ones(col, split_count=split_count)
    row0, row1 = filter_ones(row, split_count=split_count)
    return col0*row0.reshape(-1, 1), col1*row1.reshape(-1, 1)
    

def dice_loss(pred, gt):
    def binary_dice(a, b, eps=1.0):
        #print(a.shape)
        s = (torch.sum(a) + torch.sum(b)+ eps)
        if s != 0:
            return 2*torch.sum(a*b)/s
        return None#torch.tensor()

    #print(pred.shape, gt.shape)
    res = [
        binary_dice(pred[:, i], gt[:, i])
        for i in range(10)]
    res = [r for r in res if r is not None]
        
    return torch.sum(torch.stack(res))
        
        

def make_conv_features(field, nfeat=13, local_neighb=5):
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
            features.extend(BTFeatureExtractor.get_moore_neighbours(field, i, j, nrows, ncols))
            features.extend(BTFeatureExtractor.get_tl_tr(field, i, j, nrows, ncols))
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
    # feat = np.concatenate([
    #     feat,
    #     np.stack([label(field.data==i) for i in range(10)], -1)
    #     ], -1)
    return feat


def make_conv_features2(field, nfeat=13, local_neighb=5):
    nrows, ncols = field.shape
    #feat = np.zeros((nrows*ncols, nfeat))
    all_features = []
    cur_idx = 0
    for i in range(nrows):
        feature_list = []
        for j in range(ncols):
            color = field.data[i, j]
            features = [
                # i,
                # j,
                field.data[i, j]]
            #features.extend(get_moore_neighbours(field, i, j, nrows, ncols))
            #features.extend(get_tl_tr(field, i, j, nrows, ncols))
            features.extend([
                len(np.unique(field.data[i,:])),
                len(np.unique(field.data[:,j])),
                #next goes count of non-zero points
                # np.sum(field.data[i, :] > 0),
                # np.sum(field.data[:, j] > 0),
                (i+j),
                #len(np.unique(field.data[
                #    i-local_neighb:i+local_neighb,
                #    j-local_neighb:j+local_neighb]))
            ])
            
            #feat[cur_idx,13]
            # features.extend([
            #     (i + ncols - j - 1),
            #     (i + j) % 2,
            #     (i + j + 1) % 2,
            #     (i + ncols - j - 1) % 2,
            #     (nrows - 1 - i + ncols - j - 1),
            #     (nrows - 1 - i + j)
            # ])
            features.extend([
                field.get(i + k, j + v)
                for k, v in product([-1, 0, 1], [-1, 0, 1])
                if k!= 0 or v!= 0
            ])
            features.extend([
                field.data[nrows - 1 - i, j],
                field.data[nrows - 1 - i, ncols - 1 - j],
                field.data[i, ncols - 1 - j],
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
                np.sum([ field.get(i + k, j + v) == 0
                     for k, v in product([-1, 1], [-1, 1])]),
                np.sum([
                     field.get(i + 1, j) == 0,
                     field.get(i - 1, j) == 0,
                     field.get(i, j + 1) == 0,
                     field.get(i, j - 1) == 0
                 ])
            ])
            features.extend([
                np.sum(field.data[i, :]==c)+np.sum(field.data[:, j] == c)
                for c in range(10)
            ])
            feature_list.append(features)
        all_features.append(feature_list)

    feat = np.asarray(all_features)
    feat = np.concatenate([
        feat,
        np.stack([label(field.data==i) for i in range(10)], -1)
        ], -1)
    masks = []
    for c in range(10):
        col = np.sum(field.data == i, 0)
        row = np.sum(field.data == i, 1)
        col0, col1 = filter_ones(col, split_count=1)
        row0, row1 = filter_ones(row, split_count=1)
        #return col0*row0.reshape(-1, 1), col1*row1.reshape(-1, 1)
        mask = col*row.reshape(-1, 1)
        masks.extend([
            col*row.reshape(-1, 1),
            col0*row0.reshape(-1, 1),
            col1*row1.reshape(-1, 1)
        ])
    
    masks = np.stack(masks, -1)
    #print(masks.shape)
    feat = np.concatenate([
        feat,
        masks
        ], -1)
    return feat
    

def get_nonzero_ids(iodata_list, make_conv_features=make_conv_features):
    zero_ids = dict()
    max_count = 0
    nfeatures = 0
    max_count += len(iodata_list)
    for iodata in iodata_list:#[sample.train, sample.test]:
        features = make_conv_features(iodata.input_field)
        nfeatures = max(nfeatures, features.shape[-1])
        features = features.reshape(-1, features.shape[-1])
        for i in np.argwhere(features.sum(0) > 0).flatten():
            if not i in zero_ids:
                zero_ids[i] = 0
            zero_ids[i] += 1
    return np.asarray([i for i in np.arange(nfeatures) if zero_ids.get(i, 0) < max_count])


def train_on_sample(sample, cutoff=0.5, debug=False, infeatures=70):
    feature_ids = get_nonzero_ids(sample.train+sample.test)
    model = nn.Sequential(
        nn.Conv2d(len(feature_ids), 128, 3, padding=1),
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
        for iodata in sample.train:
            features = make_conv_features(iodata.input_field)#.reshape(iodata.input_field.shape+(-1,))
            features = features[:, :, feature_ids]
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
        for iodata in sample.test:
            features = make_conv_features(iodata.input_field)#.reshape(iodata.input_field.shape+(-1,))
            features = features[:, :, feature_ids]
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
        if loss == "mse":
            self.loss_func = torch.nn.MSELoss()
        else:
            self.loss_func = dice_loss
        #print(net.parameters())
        self.nepochs = nepochs
        self.lr = 0.01
        self.debug = False
        
    def build_model(self, feature_ids):
        model = nn.Sequential(
            nn.Conv2d(len(feature_ids), 128, 3, padding=1),
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
        return model
        #
        
    def train(self, iodata_list):
        self.feature_ids = get_nonzero_ids(iodata_list,
            make_conv_features=make_conv_features2)
        self.model = self.build_model(self.feature_ids)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        all_losses = []
        for epoch in range(self.nepochs):
            self.model.train()
            if self.debug:
                print("Epoch", epoch)
            losses = []
            self.optimizer.zero_grad()
            #train_x, train_y, result = make_features(iodata_list)
            for iodata in iodata_list:
                features = make_conv_features2(iodata.input_field)#.reshape(iodata.input_field.shape+(-1,))
                features = features[:, :, self.feature_ids]
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
                if len(all_losses) > 10 and np.mean(all_losses[-10:]) <= losses:
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
            features = make_conv_features2(field)
            features = features[:, :, self.feature_ids]
            features = np.moveaxis(features, -1, 0)
            features = features[np.newaxis, ...]
            i = torch.tensor(features).float()
            p = self.model.forward(i)
            p = torch.squeeze(p, dim=0).detach().cpu().numpy()
        yield Field.from_splitted(p)

    def __str__(self):
        return "ConvolutionPredictor()"
    

class Convolution2PointPredictor(Predictor, AvailableShape2PointOrConstColor):
    def __init__(self, nepochs=40, loss="mse"):
        #self.xgb =  XGBClassifier(n_estimators=25*2, booster="dart", n_jobs=-1)
        if loss == "mse":
            self.loss_func = torch.nn.MSELoss()
        else:
            self.loss_func = dice_loss
        #print(net.parameters())
        self.nepochs = nepochs
        self.lr = 0.01
        self.debug = False
        
    def build_model(self, feature_ids):
        model = nn.Sequential(
            nn.Conv2d(len(feature_ids), 128, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(),
            #nn.Sigmoid(),
            nn.Conv2d(32, 10, 3, padding=1),
            nn.AvgPool2d(3), 
            nn.Sigmoid()
            
            #nn.Softmax(dim=1)
            
        )
        return model
        #
        
    def train(self, iodata_list):
        self.feature_ids = get_nonzero_ids(iodata_list,
            make_conv_features=make_conv_features2)
        self.model = self.build_model(self.feature_ids)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        all_losses = []
        for epoch in range(self.nepochs):
            self.model.train()
            if self.debug:
                print("Epoch", epoch)
            losses = []
            self.optimizer.zero_grad()
            #train_x, train_y, result = make_features(iodata_list)
            for iodata in iodata_list:
                features = make_conv_features2(iodata.input_field)#.reshape(iodata.input_field.shape+(-1,))
                features = features[:, :, self.feature_ids]
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
                if len(all_losses) > 10 and np.mean(all_losses[-10:]) <= losses:
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
            features = make_conv_features2(field)
            features = features[:, :, self.feature_ids]
            features = np.moveaxis(features, -1, 0)
            features = features[np.newaxis, ...]
            i = torch.tensor(features).float()
            p = self.model.forward(i)
            p = torch.squeeze(p, dim=0).detach().cpu().numpy()
        yield Field.from_splitted(p)

    def __str__(self):
        return "ConvolutionPredictor()"
