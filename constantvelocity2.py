import numpy as np

class ConstantVelocityKalmanFilter2:
    # Ein Objekt befindet sich an einem unbekannten Ort und bewegt sich mit einer unbekannten (aber konstanten) Geschwindigkeit fort
    # Es gibt 5 unabhängige Messungen mit jeweils unkorreliertem Messrauschen
    # Die Standardabweichung für jede Messung ist in jeder Phase zufällig
    # Die letzte 10 Dimensionen geben die jeweilige Standardabweichung des Meßrauschens an
    # ZIEL: Schätzung der 2-dimensionalen Position des Objekts

    def __init__(self):
        self.x = np.zeros(4)  # Zustandsvektor: 4 dimensional bestehend aus Position und Geschwindigkeit [x, y, vx, vy]
        self.P = np.eye(4)    # Fehlerkovarianzmatrix: beschreibt Unsicherheit über x

    def reset(self, measurement):
        # gibt die schätzung für die position von x zurück
        z = measurement[:10].reshape(5, 2)  # reshape teilt den Vektor in eine 5x2 Matrix auf = jede Zeile ist eine Positionsmessung
        self.x = np.mean(z, axis=0)         # mittlerer Punkt aller Messungen
        self.x = np.append(self.x, [0, 0])  # geschwindigkeitsvektor wird zu x hinzugefügt
        self.P = np.eye(4)                  # hohe kovarianz = hohe unsicherheit 
        return self.x[:2]

    def update(self, dt, measurement):
        z = measurement[:10].reshape(5, 2)
        R_diag = measurement[10:]
        R = np.diag(R_diag)
        
        # F = Zustandsübergangsmatrix
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        self.x = F @ self.x                # Berechnung des "neuen" Zustands
        self.P = F @ self.P @ F.T          # Berechnung der neuen Fehlerkovarianz 

    
        z_mean = np.mean(z, axis=0)            # z_mean = Messvektor ist der Durschnitt der Positionsmessungen
        H = np.array([[1, 0, 0, 0],            # H = Messmatrix
                      [0, 1, 0, 0]])
        S = H @ self.P @ H.T + R               # S = Innovationskovarianzmatrix
        K = self.P @ H.T @ np.linalg.inv(S)    # K = Der Kalman Gain
        
        y = z_mean - H @ self.x[:2]              # y = Residualvektor
        self.x = self.x + K @ y                  # finaler Zustand (Kalman Gain wird mit Residualvektor multipliziert)
        self.P = (np.eye(4) - K @ H) @ self.P    # finale Fehlerkovarianz

        return self.x[:2]