import numpy as np


"""
Objekt bewegt sich statisch: x(t) = x(t-1)
Jede Messung hat individuelle Meßrauschen
"""

class KalmanFilter():
    def __init__(self, measurement_size):
        # Initialisierung der Zustandsvariablen
        self.measurement_size = measurement_size
        self.x = np.zeros(measurement_size)  # Zustand (Position)
        self.P = np.eye(measurement_size)  # Fehlerkovarianzmatrix
        self.R = np.eye(measurement_size) # Messrauschkovarianz
        self.Q = np.eye(measurement_size) * 0 # Prozessrauschkovarianz

    def reset(self, measurement):
        # Initialisierung des Zustands mit der ersten Messung
        self.x = np.array(measurement[:self.measurement_size])
        return self.x
    
    def update(self, dt, measurement):
        # Vorhersage (Predict)
        self.x = self.x  # Da das Objekt statisch ist, ändert sich der Zustand nicht
        self.P = self.P + self.Q  # Aktualisierung der Fehlerkovarianzmatrix
        
        # Messung (Measurement)
        z = np.array(measurement[:self.measurement_size])
        # Individuelle Meßrauschen
        self.R = np.array(measurement[self.measurement_size:].reshape(self.measurement_size,self.measurement_size))

        # Berechnung des Kalman-Gewinns
        S = self.P + self.R  # Innovationskovarianz
        K = np.dot(self.P, np.linalg.inv(S))  # Kalman-Gewinn
        
        # Update
        y = z - self.x  # Innovationsvektor
        self.x = self.x + np.dot(K, y)  # Aktualisierung des Zustands
        self.P = self.P - np.dot(K, self.P)  # Aktualisierung der Fehlerkovarianzmatrix
        return self.x
