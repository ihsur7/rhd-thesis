import os

class Directory():
    def __init__(self, input_str, output_str=None):
        self.input_str = input_str
        self.output_str = output_str

    def InputDIR(self):
        input_path = os.path.join(*self.input_str.split('/'))
        return input_path

    def OutputDIR(self):
        if not os.path.exists(os.path.join(*self.output_str.split('/'))):
            os.makedirs(os.path.join(*self.output_str.split('/')))
        output_path = os.path.join(*self.output_str.split('/'))
        return output_path