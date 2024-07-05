import numpy as np

class KalmanFilterConstantVelocityMultiple:
    def __init__(self):
        self.x = np.zeros(4)  # Zustandsvektor [x, y, vx, vy]
        self.P = np.eye(4)  # Kovarianzmatrix
        self.I = np.eye(4)  # Einheitsmatrix

    def reset(self, measurement):
        self.x = np.zeros(4)
        self.x[:2] = np.mean(measurement[:10].reshape(5, 2), axis=0)  # Durchschnittliche Position als Initialisierung
        self.P = np.eye(4)
        return self.x[:2]

    def update(self, dt, measurement):
        z = measurement[:10].reshape(5, 2)  # Extrahiere die fünf Positionsmessungen
        R_values = measurement[10:]  # Extrahiere die Standardabweichungen
        R = np.diag(R_values)  # Erstelle die Diagonalmatrix der Messrauschkovarianzen

        # Zustandsübergangsmatrix
        self.F = np.eye(4)
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        # Vorhersage
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T

        # Berechnung der Residuen und Jacobian
        H = np.zeros((10, 4))
        for i in range(5):
            H[2*i:2*i+2, :2] = np.eye(2)

        # Korrektur
        z_mean = np.mean(z, axis=0)  # Durchschnittliche Position
        y = z_mean - self.x[:2]  # Residuum basierend auf der durchschnittlichen Position

        # Skalieren der Residuen entsprechend der Standardabweichungen
        y = np.concatenate([y for _ in range(5)])

        # Kalman-Gewinn
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Aktualisieren der schätzung
        self.x = self.x + K @ y
        self.P = (self.I - K @ H) @ self.P

        return self.x[:2]