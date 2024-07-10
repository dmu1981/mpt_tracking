import numpy as np

class ConstantVelocityKalmanFilter:
    # Ein Objekt befindet sich an einem unbekannten Ort und bewegt sich mit einer unbekannten (aber konstanten) Geschwindigkeit fort.
    # Es gibt 5 unabhängige Messungen mit jeweils unkorreliertem Messrauschen
    # Die Standardabweichung für jede Messung ist in jeder Phase zufällig
    # Die letzte 10 Dimensionen geben die jeweilige Standardabweichung des Meßrauschens an
    # Wir schätzen die 2-dimensionale Position des Objekts. 

    def __init__(self):
        self.state_size = 4  # [x, y, vx, vy]
        self.measurement_size = 2  # [x, y]
        
        # Zustand und Kovarianzinitialisierung
        self.x = np.zeros(self.state_size)
        self.P = np.eye(self.state_size) * 1000  # Initial große Unsicherheit
        
        # Prozessrauschkovarianz Q
        self.Q = np.eye(self.state_size) * 0.01
        
    def reset(self, measurement):
        # Initialisierung des Zustands x mit der ersten Messung (Durchschnitt der ersten 5 Messungen)
        z = measurement[:10].reshape(5, 2)
        self.x[:2] = np.mean(z, axis=0)  # Mittelwert der Positionsmessungen
        self.x[2:] = 0  # Anfangsgeschwindigkeit auf 0 setzen
        self.P = np.eye(self.state_size) * 1000  # Kovarianz zurücksetzen
        return self.x[:2]
    
    def update(self, dt, measurement):
        # Rekonstruierung von Positionsmessungen und Messunsicherheiten anhand des Messvektors
        z = measurement[:10].reshape(5, 2)
        R = np.diag(measurement[10:])
        
        # Zustandsübergangsmatrix F
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Vorhersage
        self.x = np.dot(F, self.x)
        self.P = np.dot(F, np.dot(self.P, F.T)) + self.Q
        
        # Kalman-Gewinnberechnung und Zustandsupdate für jede Messung
        for i in range(5):
            z_i = z[i]
            R_i = np.diag(R[i*2:i*2+2])
            
            # Messmatrix H
            H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])
            
            # Korrektur
            y = z_i - np.dot(H, self.x)
            S = np.dot(H, np.dot(self.P, H.T)) + R_i
            K = np.dot(self.P, np.dot(H.T, np.linalg.inv(S)))
            
            # Update
            self.x = self.x + np.dot(K, y)
            self.P = self.P - np.dot(K, np.dot(H, self.P))
        
        return self.x[:2]

