import numpy as np

class ConstantVelocityKalmanFilter2:
    # Ein Objekt befindet sich an einem unbekannten Ort und bewegt sich mit einer unbekannten (aber konstanten) Geschwindigkeit fort
    # Es gibt 5 unabhängige Messungen mit jeweils unkorreliertem Messrauschen
    # Die Standardabweichung für jede Messung ist in jeder Phase zufällig
    # Die letzte 10 Dimensionen geben die jeweilige Standardabweichung des Meßrauschens an
    # ZIEL: Schätzung der 2-dimensionalen Position des Objekts

    def __init__(self, state_size, measurement_size):
        self.state_size = state_size  # Zustand: 4-dimensionaler Vektor aus Position und Geschwindigkeit (x, y, vx, vy)
        self.measurement_size = measurement_size  # Messungen: 2-dimensionaler Vektor (x, y)

        # Initialisierung des Zustands und der Kovarianzmatrix
        self.x = np.zeros(self.state_size)
        self.P = np.eye(self.state_size) * 1000 # Eine hohe Kovarianz = hohe Unsicherheit aufgrund mangelnder Infos

        # Prozessrauschkovarianz Q
        self.Q = np.eye(self.state_size) * 0.01 # Das Prozessrauschen wird modelliert

    def reset(self, measurement):
        # Initialisierung des Zustands mit dem Mittelwert der ersten fünf unabhängigen Messungen
        z = measurement[:10]
        self.x[:2] = np.mean(z, axis=0)  # Durchschnitt der Positionen
        self.x[2:] = 0  # Anfangsgeschwindigkeit auf 0 setzen
        self.P = np.eye(self.state_size) * 1000  # Kovarianz zurücksetzen
        return self.x[:2]

    def update(self, dt, measurement):
        # Extrahiere Messungen und Messrauschen
        z = measurement[:10]
        R_values = measurement[10:]

        # Zustandsübergangsmatrix F
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Vorhersage
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

        # Kalman-Gewinnberechnung und Zustandsupdate für alle 5 Messungen (for schleife)
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        for i in range(5): 
            z_i = z[i]
            R_i = np.diag(R_values[i*2:i*2+2])

            # Korrektur
            y = z_i - H @ self.x
            S = H @ self.P @ H.T + R_i
            K = self.P @ H.T @ np.linalg.inv(S)

            # Aktualisierung
            self.x = self.x + K @ y
            self.P = self.P - K @ H @ self.P

        return self.x[:2]