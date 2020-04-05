from base.field import Field
from predictors.basic import Predictor, AvailableAll, AvailableWithIntMultiplier


class RepeatingPredictor(Predictor, AvailableWithIntMultiplier):
    def __init__(self):
        self.m1 = 1
        self.m2 = 1

    def train(self, iodata_list):
        all_sizes = set()
        for iodata in iodata_list:
            m1 = iodata.output_field.height // iodata.input_field.height
            m2 = iodata.output_field.width // iodata.input_field.width
            all_sizes.append((m1, m2))
        self.m1, self.m2 = all_sizes.pop()

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        result = np.zeros((field.height * self.m1, field.width * self.m2), dtype=field.dtype)
        #for offset1 in range(self.m1):
        #    for offset2 in range(self.m2):
        for i in range(field.height):
            for j in range(field.width):
                result[i::field.height, j::field.width] = field.data[i, j]
        yield Field(result)

    def __str__(self):
        return f"RepeatingPredictor(m1={self.m1}, m2={self.m2})"



class ResizingPredictor(Predictor, AvailableWithMultiplier):
    def __init__(self):
        self.m1 = 1
        self.m2 = 1

    def train(self, iodata_list):
        all_sizes = set()
        for iodata in iodata_list:
            m1 = iodata.output_field.height // iodata.input_field.height
            m2 = iodata.output_field.width // iodata.input_field.width
            all_sizes.append((m1, m2))
        self.m1, self.m2 = all_sizes.pop()

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        result = np.zeros((field.height * self.m1, field.width * self.m2), dtype=field.dtype)
        for i in range(field.height):
            for j in range(field.width):
                result[
                    i * self.m1 : (i + 1) * self.m1,
                    j * self.m2 : (j + 1) * self.m2 ] = field.data[i, j]
        yield Field(result)

    def __str__(self):
        return f"ResizingPredictor(m1={self.m1}, m2={self.m2})"


class FractalPredictor(Predictor, AvailableWithMultiplier):
    def __init__(self):
        self.m1 = 1
        self.m2 = 1

    def train(self, iodata_list):
        all_sizes = set()
        for iodata in iodata_list:
            m1 = iodata.output_field.height // iodata.input_field.height
            m2 = iodata.output_field.width // iodata.input_field.width
            all_sizes.append((m1, m2))
        self.m1, self.m2 = all_sizes.pop()

    def predict(self, field):
        if isinstance(field, IOData):
            for v in self.predict(field.input_field):
                yield v
            return
        result = np.zeros((field.height * self.m1, field.width * self.m2), dtype=field.dtype)
        for i in range(field.height):
            for j in range(field.width):
                result[i::field.height, j::field.width] = field.data[i, j]
        for i in range(field.height):
            for j in range(field.width):
                if field.data[i, j] == 0:
                    result[
                        i * self.m1 : (i + 1) * self.m1,
                        j * self.m2 : (j + 1) * self.m2 ] = 0 # field.data[i, j]
        yield Field(result)

    def __str__(self):
        return f"FractalPredictor(m1={self.m1}, m2={self.m2})"