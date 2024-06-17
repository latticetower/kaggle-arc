from predictors.basic import *
from operations.field2point import SimpleSummarizeOperation

class SimpleSummarizePredictor(Predictor):
    def __init__(self):
        self.op = SimpleSummarizeOperation()

    def is_available(self, iodata_list):
        for iodata in iodata_list:
            if iodata.output_field.shape!= (1,1):
                return False
        return True
    def train(self, iodata_list):
        self.op.train(iodata_list)
        pass
    def predict(self, field):
        result = self.op.do(field, bg = self.op.bg)
        yield result
    pass