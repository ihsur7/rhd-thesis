import os

class Directory():
    def __init__(self, input_str, output_str):
        self.input_str = input_str
        self.output_str = output_str
    
    def InputDIR(self):
        if ',' in self.input_str:
            separator = ','
            working_path = os.path.join(*self.input_str.split(separator))
            # print("Working path is: " + working_path)
            # print('test')
            return working_path
        elif '/' in self.input_str:
            separator = '/'
            working_path = os.path.join(*self.input_str.split(separator))
            # print("Working path is: " + working_path)
            return working_path
        else:
            print('Please use a comma or forward-slash as a separator.')
            return
        # path = os.path.join(*self.input_str.split())
    
    def OutputDIR(self):
        if ',' in self.output_str:
            separator = ','
            working_path = os.path.join(*self.output_str.split(separator))
            # print("Working path is: " + working_path)
            # print('test')
            return working_path
        elif '/' in self.output_str:
            separator = '/'
            working_path = os.path.join(*self.output_str.split(separator))
            # print("Working path is: " + working_path)
            return working_path
        else:
            print('Please use a comma or forward-slash as a separator.')
            return
