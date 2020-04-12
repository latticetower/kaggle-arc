"""
Transformer functions for iodata
"""
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