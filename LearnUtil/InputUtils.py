import numpy as np
class InputUtils:

    def __init__(self, dim, channels, size1, size2):
        self._dim = dim
        self._channels = channels
        self._size1 = size1
        self._size2 = size2

    def get_input_shape(self):
        if self._dim == 1:
            return self._channels, self._size1
        if self._dim == 2:
            return self._channels, self._size1, self._size2

    def reshape_input(self, input, add_size=1):
            return np.reshape(input, [add_size] + list(self.get_input_shape()))

