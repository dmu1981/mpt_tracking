import numpy as np

class RandomNoiseFilter():
    def __init__(self, measurement_size):
        # Initialisiere Zustandsvariablen
        self.measurement_size = measurement_size
        self.x = np.zeros(measurement_size)  # Zustand (Position)
        self.P = np.eye(measurement_size)  # Fehlerkovarianzmatrix

    def reset(self, measurement):
        # Initialisiere den Zustand mit der ersten Messung
        self.x = np.array(measurement[:self.measurement_size])
        self.P = np.array(measurement[self.measurement_size:]).reshape(self.measurement_size, self.measurement_size)
        return self.x
    
    def update(self, dt, measurement):
        # Messung (Measurement)
        z = np.array(measurement[:self.measurement_size])
        R = np.array(measurement[self.measurement_size:]).reshape(self.measurement_size, self.measurement_size)
        
        # Berechne die inversen Kovarianzmatrizen
        P_inv = np.linalg.inv(self.P)
        R_inv = np.linalg.inv(R)
        
        # Fusion der Kovarianzmatrizen
        P_fused = np.linalg.inv(P_inv + R_inv)
        
        # Fusion der Zustände
        self.x = P_fused @ (P_inv @ self.x + R_inv @ z)
        
        # Aktualisiere die Fehlerkovarianzmatrix
        self.P = P_fused
        
        return self.x
