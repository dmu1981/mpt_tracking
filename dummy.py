import numpy as np


class DummyFilter:
    def __init__(self, shape):
        self.shape = shape

    def reset(self, measurement):
        return np.zeros(self.shape)

    def update(self, dt, measurement):
        return np.zeros(self.shape)

