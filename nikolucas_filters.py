import numpy as np
import pandas as pd

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


class KalmanFilter:
    def __init__(self, state_dim=2, measurement_dim=2):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.x = np.array([[1], [1]])  # initialer Zustandsvektor
        self.P = np.eye(state_dim)   # initiale Kovarianzmatrix (Anfangsunsicherheit)
        self.F = np.eye(state_dim)  # Zustandsübergangsmatrix (Vorgeschrieben: Objekt befindet sich an einem Ort)
        self.H = np.eye(measurement_dim, state_dim) * 1.006  # Beobachtungsmatrix (direkte Positionsmessung)
        self.Q = np.eye(state_dim) * 1e-6 # Prozessrauschen
        self.R = np.eye(measurement_dim) * 0.04  # vorgegebenes Messrauschen 

    def reset(self, measurement):
        self.x = measurement 
        self.P = np.eye(self.state_dim)
        return self.x

    def update(self, dt, measurement):
        
        # Vorhersageschritt
        x_pred = self.F @ self.x # Zustandsschätzung basierend auf aktuellem Zustand und dem Übergang (hier konstanter Übergang)
        P_pred = self.F @ self.P @ self.F.T + self.Q # Vorhersage des neuen P, basierend auf F, P und Q

        # Update-Schritt
        S = self.H @ self.P @ self.H.T + self.R # stellt die Unsicherheit in der Messung dar
        K = P_pred @ self.H.T @ np.linalg.inv(S) # Berechnung des Kalman-Gain, der bestimmt, wie stark die Schätzung basierend auf der neuen Messung korrigiert werden soll
        self.x = x_pred + K @ (measurement - self.H @ x_pred) # aktualisiert die Position basierend auf der Differenz zwischen der tatsächlichen Messung und der vorhergesagten Messung
        self.P = (np.eye(self.state_dim) - K @ self.H) @ P_pred # Aktualisiere die Unsicherheit im neuen Zustand, basierend auf dem Kalman-Gain

        return self.x



class ConstantVelocityKalmanFilter:
    def __init__(self):
        self.state_dim = 4
        self.measurement_dim = 2 # es wird nur die Position gemessen
        self.x = np.zeros((self.state_dim, 1))  # initialer Zustand
        self.P = np.eye(self.state_dim)  # Anfangsunsicherheit
        self.F = np.eye(self.state_dim)  # konstante Geschwindigkeit = kosante Änderung der Position (weiter unten mit dt aktualisiert)
        self.H = np.zeros((self.measurement_dim, self.state_dim))  # Position beschreibt das System gänzlich
        self.H[0, 0] = 1
        self.H[1, 1] = 1.005
        self.Q = np.eye(self.state_dim) * 1e-8  # kleines Prozessrauschen 
        self.R = np.eye(self.measurement_dim) * 0.04  # Messrauschen vorgegeben

    def reset(self, measurement):
        self.x[:2] = measurement[:2].reshape(2, 1)
        self.x[2:] = 0.0001  # Initial velocities set to zero
        self.P = np.eye(self.state_dim) * 1  # Reinitialize P with some uncertainty
        return self.x[:2].flatten()

    def update(self, dt, measurement):
        # Update the state transition matrix with the new dt
        self.F[0, 2] = dt # aktualisiere die Matrix
        self.F[1, 3] = dt

        # Prediction step
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Measurement update step
        z = measurement[:2].reshape(2, 1)
        S = self.H @ P_pred @ self.H.T + self.R  # Innovationskovarianz
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # Berechnung des Kalman-Gain (wie stark soll sich auf die Messung verlassen werden)
        self.x = x_pred + K @ (z - self.H @ x_pred)
        self.P = (np.eye(self.state_dim) - K @ self.H) @ P_pred

        return self.x[:2].flatten()


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
        self.Q = np.eye(self.state_dim) * 1e-7 # Initial process noise
        self.R = np.eye(self.measurement_dim) * 0.04  # Measurement noise covariance matrix
        self.alpha = 0.0000000001  # Adaptation rate for Q

    def reset(self, measurement):
        self.x[:2] = measurement[:2].reshape(2, 1)
        self.x[2:] = 0  # Initial velocities set to zero
        self.P = np.eye(self.state_dim)  # Reinitialize P with some uncertainty
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
    

class RandomNoise:
    def __init__(self, state_dim=2, measurement_dim=2):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.x = np.zeros((state_dim, 1))  # Ausgangszustand
        self.P = np.eye(state_dim)  # Anfangsunsicherheit 
        self.F = np.eye(state_dim)  # Position des Objektes bleibt gleich
        self.H = np.eye(measurement_dim, state_dim) * 1.001  # wieder direkte Positionsmessung
        self.Q = np.eye(state_dim) * 1e-9  # kleines Prozessrauschen

    def reset(self, measurement):
        self.x = measurement[:self.state_dim].reshape(self.state_dim, 1) # Position wird auf basierend auf der Messung initialisiert
        self.P = np.eye(self.state_dim) # Anfangsunsicherheit initialisieren
        return self.x.flatten()

    def update(self, dt, measurement):
        # Extract the measured positions and the measurement noise covariance matrix
        z = measurement[:self.measurement_dim].reshape(self.measurement_dim, 1) # extrahiert die Positionen der Messungen 
        Rt = measurement[2:].reshape(2, 2) # extrahiert das Messrauschen 
        
        # Prediction step
        x_pred = self.F @ self.x # Zustandsschätzung basierend auf aktuellem Zustand und dem Übergang (hier konstanter Übergang)
        P_pred = self.F @ self.P @ self.F.T + self.Q  # Vorhersage des neuen P, basierend auf F, P und Q

        # Update step
        S = self.H @ P_pred @ self.H.T + Rt  # berechnet die Innovationskovarianz
        K = P_pred @ self.H.T @ np.linalg.inv(S + np.eye(S.shape[0]) * 1e-9)  # berechnet das Kalman-Gain (bestimmt, wie stark die Messung korrigiert werden soll)
        self.x = x_pred + K @ (z - self.H @ x_pred) # neue Zustandsschätzung
        self.P = (np.eye(self.state_dim) - K @ self.H) @ P_pred # neue Kovarianzmatrix

        return self.x.flatten()
    
    
class ExtendedKalmanFilter:
    def __init__(self, process_noise=1e-5, measurement_noise_r=0.01, measurement_noise_phi=0.0025):
        self.state_dim = 2  # State dimension [x, y]
        self.measurement_dim = 2  # Measurement dimension [r, phi]
        self.x = np.zeros((self.state_dim, 1))  # Initial state vector [x, y]
        self.P = np.eye(self.state_dim)  # Initial covariance matrix
        self.Q = np.eye(self.state_dim) * process_noise  # Process noise covariance matrix
        self.R = np.array([[measurement_noise_r, 0], [0, measurement_noise_phi]])  # Measurement noise covariance matrix

    def reset(self, measurement):
        r, phi = measurement
        self.x[0] = r * np.cos(phi)
        self.x[1] = r * np.sin(phi)
        self.P = np.eye(self.state_dim)   # Reinitialize P with some uncertainty
        return self.x.flatten()

    def h(self, x):
        """Measurement function to convert state to measurement."""
        r = np.sqrt(x[0]**2 + x[1]**2)
        phi = np.arctan2(x[1], x[0])
        return np.array([r, phi]).reshape(-1, 1)

    def H_jacobian(self, x):
        """Jacobian of the measurement function."""
        r = np.sqrt(x[0]**2 + x[1]**2)
        H = np.zeros((self.measurement_dim, self.state_dim))
        if r != 0:
            H[0, 0] = x[0] / r
            H[0, 1] = x[1] / r
            H[1, 0] = -x[1] / (r**2)
            H[1, 1] = x[0] / (r**2)
        return H

    def update(self, dt, measurement):
        # Prediction step (no movement, so prediction is the same as previous state)
        x_pred = self.x
        P_pred = self.P + self.Q

        # Measurement update step
        z = measurement.reshape(self.measurement_dim, 1)
        h_x = self.h(x_pred)
        y = z - h_x
        H = self.H_jacobian(x_pred)
        S = H @ P_pred @ H.T + self.R  # Innovation covariance
        K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = x_pred + K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ P_pred

        return self.x.flatten()

    

