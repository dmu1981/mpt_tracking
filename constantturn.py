import numpy as np

class Constantturn_KalmanFilter:
    def __init__(self, state_size, measurement_size, turn_rate = 0.001):
        self.state_size = state_size  # Zustand: 4-dimensionaler Vektor aus Position und Geschwindigkeit (x, y, vx, vy)
        self.measurement_size = measurement_size  # Messungen: 2-dimensionaler Vektor (x, y)

        # Initialisierung des Zustands und der Kovarianzmatrix
        self.x = np.zeros(self.state_size)
        self.P = np.eye(self.state_size)

        # Prozessrauschkovarianz Q
        self.Q = np.eye(self.state_size) * 0.01 # Das Prozessrauschen wird beispielhaft modelliert, um den RMSE niedriger zu machen
        
        #Drehgeschwendigkeitsrate
        self.turn_rate = turn_rate

    def reset(self, measurement):
        z = measurement[:10].reshape(5, 2) 
        self.x[:2] = np.mean(z, axis=0)  # Durchschnitt der fünf Positionsmessungen
        self.x[2:] = 0  # Anfangsgeschwindigkeit auf 0 setzen
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

    