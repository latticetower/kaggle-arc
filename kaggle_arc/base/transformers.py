"""
Transformer functions for iodata
"""
import numpy as np

from base.field import Field
from base.iodata import IOData


def resize_output(iodata):
    if isinstance(iodata, list):
        return [ resize_output(data) for data in iodata ]
    h, w = iodata.input_field.shape
    if iodata.output_field is not None:
        output = iodata.output_field.data[:h, :w]
        output = Field(output)
    else:
        output = None
    return IOData(input_field=iodata.input_field, output_field=output)
    
def crop_data(data):
    h = np.argwhere(data.std(0) > 0).flatten()
    w = np.argwhere(data.std(1) > 0).flatten()
    if len(h) < 1 or len(w) < 1:
        return data
    return data[min(h) : max(h) + 1, min(w) : max(w) + 1]