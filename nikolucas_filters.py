import numpy as np
import pandas as pd

class AdaptiveKalmanFilter:
    def __init__(self):
        self.state_dim = 4
        self.measurement_dim = 2
        self.x = np.zeros((self.state_dim, 1))  # Initial state vector [x, y, vx, vy]
        self.P = np.eye(self.state_dim)  # Initial covariance matrix with some uncertainty
        self.F = np.eye(self.state_dim)  # State transition matrix (to be updated with dt)
        self.H = np.zeros((self.measurement_dim, self.state_dim))  # Observation matrix
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.Q = np.eye(self.state_dim) * 1e-6  # Initial process noise
        self.R = np.eye(self.measurement_dim) * 0.04  # Measurement noise covariance matrix
        self.alpha = 0.00001  # Adaptation rate for Q

    def reset(self, measurement):
        self.x[:2] = measurement[:2].reshape(2, 1)
        self.x[2:] = 0  # Initial velocities set to zero
        self.P = np.eye(self.state_dim) * 1  # Reinitialize P with some uncertainty
        return self.x[:2].flatten()

    def update(self, dt, measurement):
        # Update the state transition matrix with the new dt
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        # Prediction step
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Measurement update step
        z = measurement[:2].reshape(2, 1)
        y = z - self.H @ x_pred  # Innovation or residual
        S = self.H @ P_pred @ self.H.T + self.R  # Innovation covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = x_pred + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ P_pred

        # Adaptive process noise update
        self.Q = (1 - self.alpha) * self.Q + self.alpha * (K @ y @ y.T @ K.T)

        return self.x[:2].flatten()



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
        self.state_dim = 2
        self.measurement_dim = 2
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
    

class RandomNoise:
    def __init__(self, state_dim=2, measurement_dim=2):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.x = np.zeros((state_dim, 1))  # Initial state vector
        self.P = np.eye(state_dim) * 1.0  # Initial covariance matrix with some uncertainty
        self.F = np.eye(state_dim)  # State transition matrix
        self.H = np.eye(measurement_dim, state_dim)  # Observation matrix
        self.Q = np.eye(state_dim) * 1e-5  # Small process noise to avoid overconfidence

    def reset(self, measurement):
        self.x = measurement[:self.state_dim].reshape(self.state_dim, 1)
        self.P = np.eye(self.state_dim) * 1  # Reinitialize P with some uncertainty
        return self.x.flatten()

    def update(self, dt, measurement):
        # Extract the measured positions and the measurement noise covariance matrix
        z = measurement[:self.measurement_dim].reshape(self.measurement_dim, 1)
        Rt = measurement[2:].reshape(2, 2)
        
        # Prediction step
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Update step
        S = self.H @ P_pred @ self.H.T + Rt  # Innovation covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S + np.eye(S.shape[0]) * 1e-9)  # Kalman gain with numerical stability
        self.x = x_pred + K @ (z - self.H @ x_pred)
        self.P = (np.eye(self.state_dim) - K @ self.H) @ P_pred

        return self.x.flatten()
