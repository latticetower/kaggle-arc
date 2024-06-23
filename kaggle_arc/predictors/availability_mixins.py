import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import numpy as np
import skimage.measure as sk_measure
from fractions import Fraction

from utils import check_if_can_be_mirrored

class AvailableAll:
    def is_available(self, iodata_list):
        return True


class AvailableEqualShape:
    def is_available(self, iodata_list):
        for iodata in iodata_list:
            if iodata.input_field.shape != iodata.output_field.shape:
                return False
        return True


class AvailableShape2Point:
    def is_available(self, iodata_list):
        for iodata in iodata_list:
            if iodata.output_field.shape != (1, 1):
                return False
        return True


class AvailableShape2PointOrConstColor:
    def is_available(self, iodata_list):
        for iodata in iodata_list:
            if iodata.output_field.shape != (1, 1):
                if len(np.unique(iodata.output_field.data)) != 1:
                    return False
        return True


class AvailableEqualShapeAndMaxNColors:
    def is_available(self, iodata_list, n_colors=4):
        for iodata in iodata_list:
            if iodata.input_field.shape != iodata.output_field.shape:
                return False
            if len(np.unique(iodata.input_field.data)) > n_colors:
                return False
            if len(np.unique(iodata.output_field.data)) > n_colors:
                return False
        return True


class AvailableWithIntMultiplier:
    def is_available(self, iodata_list):
        all_sizes = set()
        for iodata in iodata_list:
            m1 = iodata.output_field.height // iodata.input_field.height
            m2 = iodata.output_field.width // iodata.input_field.width
            all_sizes.add((m1, m2))
        if len(all_sizes) == 1:
            h, w = all_sizes.pop()
            if w > 1 and h > 1:
                self.m1 = h
                self.m2 = w
                return True
        return False


class AvailableWithFractionalMultiplier:
    def is_available(self, iodata_list):
        all_sizes = set()
        for iodata in iodata_list:
            m1 = Fraction(iodata.output_field.height, iodata.input_field.height)
            m2 = Fraction(iodata.output_field.width, iodata.input_field.width)
            all_sizes.add((m1, m2))
        if len(all_sizes) == 1:
            h, w = all_sizes.pop()
            self.m1 = h
            self.m2 = w
            return True
        return False


class AvailableMirror(AvailableWithIntMultiplier):
    def is_available(self, iodata_list):
        availability_check = AvailableWithIntMultiplier()
        # print(isinstance(self, AvailableMirror))
        if not availability_check.is_available(iodata_list):
            # print(11)
            return False
        self.m1 = availability_check.m1
        self.m2 = availability_check.m2
        results = set()
        for iodata in iodata_list:
            h, w = iodata.input_field.shape
            res = check_if_can_be_mirrored(iodata.output_field.data, h=h, w=w)
            # print(res)
            if res is None:
                return False
            results.add(res)
        (vertical, horizontal) = results.pop()
        if len(results) > 0:
            return False
        self.vertical = vertical
        self.horizontal = horizontal
        return True


class AvailableEqualShapeAndLessThanNComponents:
    def is_available(self, iodata_list, n_components=10):
        for iodata in iodata_list:
            if iodata.input_field.shape != iodata.output_field.shape:
                return False
        for iodata in iodata_list:
            region_labels = sk_measure.label(iodata.input_field.data)
            max_region_id = np.max(region_labels)
            if max_region_id > n_components:
                return False
        return True