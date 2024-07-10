import numpy as np

class ConstantVelocityKalmanFilter2:
    # Ein Objekt befindet sich an einem unbekannten Ort und bewegt sich mit einer unbekannten (aber konstanten) Geschwindigkeit fort.
    # Es gibt 5 unabhängige Messungen mit jeweils unkorreliertem Messrauschen
    # Die Standardabweichung für jede Messung ist in jeder Phase zufällig
    # Die letzte 10 Dimensionen geben die jeweilige Standardabweichung des Meßrauschens an
    # ZIEL: Schätzung der 2-dimensionalen Position des Objekts

    def __init__(self, measurement_size):
        # Initialisierung des Zustands und der Kovarianzmatrix
        self.measurement_size = measurement_size 
        self.x = np.zeros(self.measurement_size)
        self.P = np.eye(measurement_size)
        self.Q = np.eye(self.measurement_size) * 0.01 # Prozessrauschkovarianz Q

    def reset(self, measurement):
        # Initialisierung des Zustands mit dem Mittelwert der ersten fünf Messungen
        z = measurement[:10].reshape(5, 2)
        self.x[:2] = np.mean(z, axis=0)  # Durchschnitt der Positionen
        self.x[2:] = 0  # Anfangsgeschwindigkeit auf 0 setzen
        self.P = np.eye(self.measurement_size) * 1000  # Kovarianz zurücksetzen
        return self.x[:2]

    def update(self, dt, measurement):
        # Extrahiere Messungen und Messrauschen
        z = measurement[:10].reshape(5, 2)
        R_values = measurement[10:]
        
        # Zustandsübergangsmatrix F
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Vorhersage (Predict)
        self.x = np.dot(F, self.x)
        self.P = np.dot(F, np.dot(self.P, F.T)) + self.Q
        
        # Kalman-Gewinnberechnung und Zustandsupdate für jede Messung
        for i in range(5):
            z_i = z[i]
            R_i = np.diag(R_values[i*2:i*2+2])
            
            # Messmatrix H
            H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])
            
            # Innovation/Korrektur
            y = z_i - np.dot(H, self.x)
            S = np.dot(H, np.dot(self.P, H.T)) + R_i
            K = np.dot(self.P, np.dot(H.T, np.linalg.inv(S)))
            
            # Aktualisierung
            self.x = self.x + np.dot(K, y)
            self.P = self.P - np.dot(K, np.dot(H, self.P))
        
        return self.x[:2]