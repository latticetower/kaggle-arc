from base.field import *


class IOData:
    def __init__(self, data, input_field=None, output_field=None):
        #data['train'][0]['input']
        self.input_field = input_field
        self.output_field = output_field
        if data is not None:
            if 'input' in data:
                self.input_field = Field(data['input'])
            if 'output' in data:
                self.output_field = Field(data['output'])

    def show(self):
        fig, axes = plt.subplots(nrows=1, ncols=2)
        ax0, ax1 = axes
        if self.input_field is not None:
            self.input_field.show(ax0, label="input")
        if self.output_field is not None:
            self.output_field.show(ax1, label="output")
        ax0.axis("off")
        ax1.axis("off")