import numpy as np

class KalmanFilterConstantVelocity:
    def __init__(self):
        self.x = np.zeros(4)  
        self.P = np.eye(4)  
        self.R = np.eye(2) * 0.04  
        self.I = np.eye(4)  

    def reset(self, measurement):
        self.x = np.zeros(4)
        self.x[:2] = measurement[:2]  
        self.P = np.eye(4)
        return self.x[:2]

    def update(self, dt, measurement):
        # Zustandsübergangsmatrix
        self.F = np.eye(4)
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        # Vorhersage
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T

        # Messmatrix
        H = np.eye(2, 4)

        # Korrektur
        z = measurement[:2]
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self.I - K @ H) @ self.P

        return self.x[:2]