import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, state_size, measurement_size, turn_rate = 0.01):
        self.state_size = state_size
        self.measurement_size = measurement_size
        self.x = np.zeros(state_size)  # Zustand [x, y, Standardabweichung_x, Standardabweichung _y]
        self.P = np.eye(state_size)   # Große Anfangskovarianzmatrix
        self.Q = np.eye(state_size) * 0.01  # Prozessrauschkovarianz
        self.H = np.zeros((self.measurement_size//2, self.state_size)) # Beobachtungsmodell
        self.turn_rate = turn_rate  # Drehgeschwindigkeit
    
    def reset(self, measurement):
        # Initialisierung des Zustands mit der ersten Messung
        self.x[:2] = measurement[:2]  # Initialisieren der Position
        return self.x[:2]

    def update(self, dt, measurement):
        # Extrahiere Messungen und Messrauschen
        z = measurement[:10].reshape(5, 2)

        # Zustandsübergangsmatrix F für konstante Drehrate
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, np.cos(self.turn_rate * dt), -np.sin(self.turn_rate * dt)],
            [0, 0, np.sin(self.turn_rate * dt), np.cos(self.turn_rate * dt)]
        ])

        # Vorhersage
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

        # Messmatrix H - nur für Positionsmessungen
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Kalman-Gewinnberechnung und Zustandsupdate für alle 5 Messungen
        for i in range(5):
            z_i = z[i]
            R_values = measurement[10:]  # Standardabweichungen
            R_i = np.diag(R_values[i*2:i*2+2])  # Quadrieren, um Varianzen zu erhalten

            # Korrektur
            y = z_i - H @ self.x
            S = H @ self.P @ H.T + R_i
            K = self.P @ H.T @ np.linalg.inv(S)

            # Aktualisierung
            self.x = self.x + K @ y
            self.P = self.P - K @ H @ self.P

        return self.x[:2]  # Rückgabe der 2-dimensionalen Position

    