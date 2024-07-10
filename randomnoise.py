import numpy as np


"""
Objekt bewegt sich statisch: x(t) = x(t-1)
Jede Messung hat individuelle Meßrauschen
"""

class Randomnoise_KalmanFilter():
    def __init__(self, measurement_size):
        # Initialisierung der Zustandsvariablen
        self.measurement_size = measurement_size
        self.x = np.zeros(measurement_size)  # Zustand (Position)
        self.P = np.eye(measurement_size)  # Fehlerkovarianzmatrix
        self.R = np.eye(measurement_size) # Messrauschkovarianz

    def reset(self, measurement):
        # Initialisierung des Zustands mit der ersten Messung
        self.x = np.array(measurement[:self.measurement_size])
        self.P = np.eye(self.measurement_size) * 0.1  # Ich habe experimentiert und den niedrigsten RMSE erreicht, indem ich die Einheitsmatrix mit dem Faktor 0,1 multipliziert habe.
        return self.x
    
    def update(self, dt, measurement):
        # Vorhersage (Predict)
        self.x = self.x  # Da das Objekt statisch ist, ändert sich der Zustand nicht
        
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
