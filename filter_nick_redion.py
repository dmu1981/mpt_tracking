import numpy as np

# Problem 1
class KalmanFilter():
    def __init__(self, shape):
        self.shape = shape
        self.A = np.eye(shape)  # Zustandsübergangsmatrix
        self.H = np.eye(shape)  # Beobachtungsmatrix
        self.Q = np.eye(shape) * 0.0001  # Prozessrauschen
        self.R = np.eye(shape) * 0.04  # Messrauschen
        self.x = np.zeros(shape)  # Zustandsvektor
        self.P = np.eye(shape)  # Kovarianzmatrix

    def reset(self, measurement):
        self.x = measurement[:self.shape]
        self.P = np.eye(self.shape)
        return self.x

    def update(self, dt, measurement):
        # Prediction
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q

        # Update
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))
        y = measurement[:self.shape] - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))

        return self.x

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


# Problem 3
class ExtendedKalmanFilter():
    def __init__(self, shape):
        self.shape = shape
        self.A = np.eye(shape)  # Zustandsübergangsmatrix
        self.Q = np.eye(shape) * 0.0001  # Prozessrauschen
        self.R = np.array([[0.0100, 0.0000], [0.0000, 0.0025]])  # Messrauschen
        self.x = np.zeros(shape)  # Zustandsvektor
        self.P = np.eye(shape)  # Kovarianzmatrix

    def h(self, x):
        """Konvertiert kartesische in polare Koordinaten"""
        r = np.sqrt(x[0]**2 + x[1]**2)
        phi = np.arctan2(x[1], x[0])
        return np.array([r, phi])

    def H_jacobian(self, x):
        """Berechnet die Jacobi-Matrix der h-Funktion"""
        r = np.sqrt(x[0]**2 + x[1]**2)
        H = np.array([
            [x[0] / r, x[1] / r],
            [-x[1] / (r**2), x[0] / (r**2)]
        ])
        return H

    def reset(self, measurement):
        r, phi = measurement
        self.x = np.array([r * np.cos(phi), r * np.sin(phi)])
        self.P = np.eye(self.shape)
        return self.x

    def update(self, dt, measurement):
        # Prediction
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q

        # Update
        H = self.H_jacobian(self.x)
        z_pred = self.h(self.x)
        y = measurement - z_pred

        # Normalisiere Winkel
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi

        S = np.dot(H, np.dot(self.P, H.T)) + self.R
        K = np.dot(self.P, np.dot(H.T, np.linalg.inv(S)))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(H, self.P))

        return self.x

# Problem 4
class KalmanFilterConstantVelocity():
    def __init__(self, shape):
        self.shape = shape
        self.dim = shape // 2
        self.A = np.eye(shape)
        self.A[:self.dim, self.dim:] = np.eye(self.dim)
        self.Q = np.eye(shape) * 0.0001  # Prozessrauschen
        self.R = np.eye(self.dim) * 0.04  # Messrauschen
        self.x = np.zeros(shape)  # Zustandsvektor (Position und Geschwindigkeit)
        self.P = np.eye(shape)  # Kovarianzmatrix

    def reset(self, measurement):
        self.x[:self.dim] = measurement[:self.dim]
        self.x[self.dim:] = 0
        self.P = np.eye(self.shape)
        return self.x[:self.dim]

    def update(self, dt, measurement):
        # Update the state transition matrix A with the new dt
        self.A[:self.dim, self.dim:] = np.eye(self.dim) * dt

        # Prediction
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q

        # Update
        H = np.eye(self.dim, self.shape)
        z_pred = np.dot(H, self.x)
        y = measurement[:self.dim] - z_pred

        S = np.dot(H, np.dot(self.P, H.T)) + self.R
        K = np.dot(self.P, np.dot(H.T, np.linalg.inv(S)))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(H, self.P))

        return self.x[:self.dim]


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
    
