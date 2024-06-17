import torch
from torch import nn

from predictors.basic import Predictor, AvailableEqualShape
from base.field import *


class CAModel(nn.Module):
    def __init__(self, num_states):
        super(CAModel, self).__init__()
        self.transition = nn.Sequential(
            nn.Conv2d(num_states, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_states, kernel_size=1)
        )
        
    def forward(self, x, steps=1):
        for _ in range(steps):
            x = self.transition(torch.softmax(x, dim=1))
        return x
        
def solve_task(iodata_list, max_steps=10, num_epochs=100):
    model = CAModel(10).to(device)
    criterion = nn.CrossEntropyLoss()
    losses = np.zeros((max_steps - 1) * num_epochs)

    for num_steps in range(1, max_steps):
        optimizer = torch.optim.Adam(model.parameters(), lr=(0.1 / (num_steps * 2)))
        
        for e in range(num_epochs):
            optimizer.zero_grad()
            loss = 0.0

            for sample in task:
                # predict output from input
                x = torch.from_numpy(inp2img(sample["input"])).unsqueeze(0).float().to(device)
                y = torch.tensor(sample["output"]).long().unsqueeze(0).to(device)
                y_pred = model(x, num_steps)
                loss += criterion(y_pred, y)
                
                # predit output from output
                # enforces stability after solution is reached
                y_in = torch.from_numpy(inp2img(sample["output"])).unsqueeze(0).float().to(device)
                y_pred = model(y_in, 1) 
                loss += criterion(y_pred, y)

            loss.backward()
            optimizer.step()
            losses[(num_steps - 1) * num_epochs + e] = loss.item()
    return model, num_steps, losses
               


class CAMPredictor(Predictor, AvailableEqualShape):
    def __init__(self, max_steps=10, num_epochs=100):
        self.max_steps = max_steps
        self.num_epochs = num_epochs
        self.device="cpu"
        self.model = CAModel(10).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        #self.optimizer = torch.optim.Adam(model.parameters(), lr=(0.1 / (max_steps * 2)))
        pass
        
    def train(self, iodata_list):
        losses = np.zeros((self.max_steps - 1) * self.num_epochs)
        self.model.train()
        for num_steps in range(1, self.max_steps):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=(0.1 / (num_steps * 2)))
            
            for e in range(self.num_epochs):
                optimizer.zero_grad()
                loss = 0.0

                for iodata in iodata_list:
                    # predict output from input
                    x = torch.from_numpy(iodata.input_field.data_splitted).unsqueeze(0).float().to(self.device)
                    y = torch.from_numpy(iodata.output_field.data).long().unsqueeze(0).to(self.device)
                    y_pred = self.model(x, num_steps)
                    loss += self.criterion(y_pred, y)
                    
                    # predit output from output
                    # enforces stability after solution is reached
                    y_in = torch.from_numpy(iodata.output_field.data_splitted).unsqueeze(0).float().to(self.device)
                    y_pred = self.model(y_in, 1) 
                    loss += self.criterion(y_pred, y)

                loss.backward()
                optimizer.step()
                losses[(num_steps - 1) * self.num_epochs + e] = loss.item()
        self.losses = losses
        #model, num_steps, losses

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(field.data_splitted).unsqueeze(0).float().to(self.device)
            pred = self.model(x, 100).argmax(1).squeeze().detach().cpu().numpy()
        yield Field(pred)
        
        
from predictors.basic import Predictor, AvailableEqualShape

class MoverPredictor(Predictor, AvailableEqualShape):
    def __init__(self):
        pass

    def train(self, iodata_list):
        self.transitions = []
        h = []
        w = []
        for iodata in iodata_list:
            i = iodata.input_field
            o = iodata.output_field
            coords = np.argwhere(i.data != o.data)
            if coords.shape[0] > 0:
                xmin, ymin = np.min(coords, 0)
                xmax, ymax = np.max(coords, 0)
                start = i.data[xmin:xmax+1, ymin:ymax+1]
                end = o.data[xmin:xmax+1, ymin:ymax+1]
            else:
                start = i.data.copy()
                end = o.data.copy()
            self.transitions.append((start, end))
            h.append(start.shape[0])
            w.append(start.shape[1])
        if len(np.unique(h)) and len(np.unique(w))==1:
            self.single_step = True
        else:
            self.single_step = False
        self.minh = np.min(h)
        self.minw = np.min(w)

    def is_available(self, iodata_list):
        for iodata in iodata_list:
            if iodata.input_field.shape!= iodata.output_field.shape:
                return False

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        data = field.data.copy()
        offsets = np.ones(data.shape)
        offsets[-self.minh+1:] = 0
        offsets[:, -self.minw+1:] = 0
        for _ in range(100):
            something_changed = False
            #print(offsets)
            for offset0, offset1 in np.argwhere(offsets==1):
                no_changes_with_offset = True
                for start, end in self.transitions:
                    h, w = start.shape
                    if offset0 + h > data.shape[0] or offset1 + w > data.shape[1]:
                        #offsets[offset0:, offset1:] = 0
                        continue
                    if np.all(data[offset0:offset0+h, offset1:offset1+w] == start):
                        data[offset0:offset0+h, offset1:offset1+w] = end[:, :]
                        offsets[offset0:offset0+h, offset1:offset1+w] = 1
                        something_changed = True
                        no_changes_with_offset = False
                        if self.single_step:
                            yield Field(data)
                            return
                        break
                if no_changes_with_offset:
                    offsets[offset0, offset1] = 0
            if not something_changed:
                break
        yield Field(data)
