import numpy as np
class InputUtils:

    def __init__(self, channels, shape):
        self._channels = channels
        self._shape= shape

    def get_input_shape(self):
        return np.concatenate([[self._channels], self._shape])
       

    def reshape_input(self, input, add_size=1):
            return np.reshape(input, np.concatenate([[add_size], self.get_input_shape()]))

