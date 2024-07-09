import numpy as np

# Problem 2
class RandomNoise:
    def __init__(self, shape):
        self.shape = shape
        self.x = np.zeros(shape)  # Zustandsschätzung   
        self.P = np.eye(shape)  # Fehlerkovarianzmatrix

    def reset(self, measurement):
        self.x = measurement[:self.shape]  # Initialisierung der Zustände mit den ersten Messungen
        self.P = np.eye(self.shape)  # Initialisierung der Fehlerkovarianzmatrix
        return self.x

    def update(self, dt, measurement): 
        z = measurement[:self.shape]
        Rt = measurement[self.shape:].reshape(self.shape, self.shape)
        
        # Predict Schritt
        # x bleibt gleich, da wir ein statisches Objekt annehmen
        # P bleibt gleich, da kein Prozessrauschen angenommen wird

        # Update Schritt
        H = np.eye(self.shape)  # Messmatrix
        y = z - H @ self.x  # Innovationsvektor
        S = H @ self.P @ H.T + Rt  # Innovationskovarianz
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman Gain

        self.x = self.x + K @ y  # Aktualisierung der Zustandschätzung
        self.P = (np.eye(self.shape) - K @ H) @ self.P  # Aktualisierung der Fehlerkovarianzmatrix

        return self.x
    

# Problem 5
class ConstantVelocity:
    def __init__(self, shape):
        self.shape = shape
        self.state_dim = shape * 2  # Zustandsvektor (Position + Geschwindigkeit)
        self.x = np.zeros(self.state_dim)  # Zustandsschätzung (Position und Geschwindigkeit)
        self.P = np.eye(self.state_dim)  # Fehlerkovarianzmatrix

    def reset(self, measurement):
        self.x[:self.shape] = np.mean(measurement[:10].reshape(5, 2), axis=0)  # Initialisierung der Position mit dem Mittelwert der ersten Messungen
        self.P = np.eye(self.state_dim)  # Initialisierung der Fehlerkovarianzmatrix
        return self.x[:self.shape]  # Rückgabe der Positionsschätzung

    def update(self, dt, measurement):
        # Positionen und Messrauschen extrahieren
        z = measurement[:10].reshape(5, 2)
        R = np.diag(measurement[10:])

        # Predict Schritt
        F = np.eye(self.state_dim)
        F[:self.shape, self.shape:] = np.eye(self.shape) * dt  # Übergangsmatrix für konstante Geschwindigkeit

        self.x = F @ self.x
        self.P = F @ self.P @ F.T

        # Update Schritt
        H = np.zeros((10, self.state_dim))
        for i in range(5):
            H[2*i:2*i+2, :self.shape] = np.eye(self.shape)

        z_hat = H @ self.x
        y = z.flatten() - z_hat  # Innovationsvektor
        S = H @ self.P @ H.T + R  # Innovationskovarianz
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman Gain

        self.x = self.x + K @ y  # Aktualisierung der Zustandschätzung
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P  # Aktualisierung der Fehlerkovarianzmatrix

        return self.x[:self.shape]  # Rückgabe der Positionsschätzung
    
# Problem 6
class KalmanFilterConstantTurn:
    def __init__(self, shape, turn_rate=0.000001):
        self.shape = shape
        self.state_dim = shape * 2  # Zustandsvektor (Position + Geschwindigkeit)
        self.turn_rate = turn_rate  # Konstante Drehgeschwindigkeit
        self.x = np.zeros(self.state_dim)  # Zustandsschätzung (Position und Geschwindigkeit)
        self.P = np.eye(self.state_dim)  # Fehlerkovarianzmatrix

    def reset(self, measurement):
        self.x[:self.shape] = np.mean(measurement[:10].reshape(5, 2), axis=0)  # Initialisierung der Position mit dem Mittelwert der ersten Messungen
        self.P = np.eye(self.state_dim)  # Initialisierung der Fehlerkovarianzmatrix
        return self.x[:self.shape]  # Rückgabe nur der Positionsschätzung

    def update(self, dt, measurement):
        # Positionen und Messrauschen extrahieren
        z = measurement[:10].reshape(5, 2)
        R = np.diag(measurement[10:])

        # Predict Schritt
        F = np.eye(self.state_dim)
        F[:self.shape, self.shape:] = np.eye(self.shape) * dt  # Übergangsmatrix für konstante Geschwindigkeit

        # Rotation für die konstante Drehgeschwindigkeit
        cos_a_dt = np.cos(self.turn_rate * dt)
        sin_a_dt = np.sin(self.turn_rate * dt)
        rotation_matrix = np.array([
            [cos_a_dt, -sin_a_dt],
            [sin_a_dt, cos_a_dt]
        ])
        F[self.shape:, self.shape:] = rotation_matrix

        self.x = F @ self.x
        self.P = F @ self.P @ F.T

        # Update Schritt
        H = np.zeros((10, self.state_dim))
        for i in range(5):
            H[2*i:2*i+2, :self.shape] = np.eye(self.shape)

        z_hat = H @ self.x
        y = z.flatten() - z_hat  # Innovationsvektor
        S = H @ self.P @ H.T + R  # Innovationskovarianz
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman Gain

        self.x = self.x + K @ y  # Aktualisierung der Zustandschätzung
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P  # Aktualisierung der Fehlerkovarianzmatrix

        return self.x[:self.shape]  # Rückgabe der Positionsschätzung
    