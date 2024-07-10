import numpy as np

"""
Objekt bewegt sich mit konstanter Geschwindigkeit: 
x(t) = x(t-1) + dt * v
Unabhängiges, normalverteiltes Rauschen mit STD = 0.2/Achse
z(t) = x(t) + et; et ~ N(0, 0.04)
"""

class KalmanFilter():
    def __init__(self, measurement_size=2):
        # Initialisierung der Zustandsvariablen
        self.measurement_size = measurement_size
        self.state_size = 2 * measurement_size  # Zustand (Position + Geschwindigkeit)
        self.x = np.zeros(self.state_size)  # Zustand (Position und Geschwindigkeit)
        self.R = np.eye(self.measurement_size) * 0.04  # Messrauschkovarianz

        # Messmatrix
        self.H = np.zeros((self.measurement_size, self.state_size))
        self.H[:, :self.measurement_size] = np.eye(self.measurement_size)

    def reset(self, measurement):
        # Initialisierung des Zustands mit der ersten Messung
        self.x[:self.measurement_size] = measurement
        self.x[self.measurement_size:] = 0  # Anfangsgeschwindigkeit auf 0 setzen
        self.P = np.eye(self.state_size) * 1000  # Fehlerkovarianzmatrix zurücksetzen
        return self.x[:self.measurement_size]

    def update(self, dt, measurement):
        # Vorhersage (Predict)
        self.F = np.eye(self.state_size)
        for i in range(self.measurement_size):
            self.F[i, i + self.measurement_size] = dt
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T)
        
        # Messung (Measurement)
        z = np.array(measurement[:self.measurement_size])
        
        # Berechnung des Kalman-Gewinns
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # Innovationskovarianz
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman-Gewinn
        
        # Update (Correct)
        y = z - np.dot(self.H, self.x)  # Innovationsvektor
        self.x = self.x + np.dot(K, y)  # Aktualisierung des Zustands
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))  # Aktualisierung der Fehlerkovarianzmatrix
        return self.x[:self.measurement_size]