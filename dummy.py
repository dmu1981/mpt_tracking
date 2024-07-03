import numpy as np


class DummyFilter:
    def __init__(self, shape):
        self.shape = shape

# Wird mit jeder neuen Messreihen aufgerufen
    def reset(self, measurement):
        return np.zeros(self.shape)
    
# Wird mit jeder Messung aufgerufen
# dt = Zeit seit letzter Messung
    def update(self, dt, measurement):
        return np.zeros(self.shape)
