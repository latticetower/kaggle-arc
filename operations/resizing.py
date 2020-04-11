import numpy as np

from operations.basic import Operation


class Repeater(Operation):
    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2

    def __call__(self, data):
        height, width = data.shape
        result = np.zeros((height * self.m1, width * self.m2),
            dtype=data.dtype)
        #for offset1 in range(self.m1):
        #    for offset2 in range(self.m2):
        for i in range(height):
            for j in range(width):
                result[i::height, j::width] = data[i, j]
        return result


class Mirror(Operation):
    def __init__(self, m1, m2, horizontal=True, vertical=True):
        self.m1 = m1
        self.m2 = m2
        self.horizontal = horizontal
        self.vertical = vertical

    def __call__(self, data):
        height, width = data.shape
        result = np.zeros((height * self.m1, width * self.m2),
            dtype=data.dtype)
        #for offset1 in range(self.m1):
        #    for offset2 in range(self.m2):
        for i in range(height):
            for j in range(width):
                result[i::height, j::width] = data[i, j]
                if self.vertical:
                    result[height + i:: 2 * height, j::width] = data[height - 1 - i, j]
                if self.horizontal:
                    result[i:: height, width + j :: 2 * width] = data[i, width - 1 - j]
                if self.horizontal and self.vertical:
                    result[height + i :: 2 * height, width + j :: 2 * width] = data[height - 1 - i, width - 1 - j]
        return result


class Resizer(Operation):
    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2
        
    def __call__(self, data):
        height, width = data.shape
        result = np.zeros((height * self.m1, width * self.m2), dtype=data.dtype)
        for i in range(height):
            for j in range(width):
                result[
                    i * self.m1 : (i + 1) * self.m1,
                    j * self.m2 : (j + 1) * self.m2 ] = data[i, j]
        return result

class Fractal(Operation):
    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2

    def __call__(self, data):
        height, width = data.shape
        result = np.zeros((height * self.m1, width * self.m2),
            dtype=data.dtype)
        for i in range(height):
            for j in range(width):
                result[i::height, j::width] = data[i, j]
        for i in range(height):
            for j in range(width):
                if data[i, j] == 0:
                    result[
                        i * self.m1 : (i + 1) * self.m1,
                        j * self.m2 : (j + 1) * self.m2 ] = 0 # field.data[i, j]
        return result