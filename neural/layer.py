# Abstract implementation of layer
class Layer:

    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        raise NotImplementedError

    def back_propagation(self, output_error, learning_rate):
        raise NotImplementedError
