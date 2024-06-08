import numpy as np

"""
Was ist der Kalman Filter?
ist ein rekursiver Algorithmus, der zur Schätzung des Zustands eines dynamischen Systems verwendet

Wie funktioniert der Kalman-Filter?
Hauptsächlich in zwei Hauptschritten: dem Vorhersageschritt und dem Update-Schritt
1: Initialisierung: 
- initialen Zustandsschätzwert und initiale Kovarianzmatrix
2: Vorhersageschritt: 
Ziel: Vorhersage des nächsten Zustands des Systems basierend auf dem aktuellen Zustand und dem Modell des Systems
3: Update-Schritt: 
Ziel: korrigieren der Vorhersage basierend auf der neuen Messung
"""

state_dim = 2
measurement_dim = 2

class KalmanFilter:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.x = np.zeros((state_dim, 1))  # initialer Zustandsvektor
        self.P = np.eye(state_dim)  # initiale Kovarianzmatrix
        self.F = np.eye(state_dim)  # Zustandsübergangsmatrix
        self.H = np.eye(measurement_dim, state_dim)  # Beobachtungsmatrix
        self.Q = np.zeros((state_dim, state_dim))  # Prozessrauschen
        self.R = np.eye(measurement_dim) * 0.04  # Messrauschen

    def reset(self, measurement):
        self.x = measurement
        self.P = np.eye(self.state_dim)
        return self.x

    def update(self, dt, measurement):
        # Vorhersageschritt
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Update-Schritt
        K = P_pred @ self.H.T @ np.linalg.inv(self.H @ P_pred @ self.H.T + self.R)
        self.x = x_pred + K @ (measurement - self.H @ x_pred)
        self.P = (np.eye(self.state_dim) - K @ self.H) @ P_pred

        return self.x