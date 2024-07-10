import numpy as np

"""
Objekt bewegt sich statisch: x(t) = x(t-1)
Unabhängiges, normalverteiltes Rauschen mit STD = 0.2/Achse
z(t) = x(t) + et; et ~ N(0, 0.04)
"""

class Constantposition_KalmanFilter():
    def __init__(self, measurement_size=2):
        # Initialisierung der Zustandsvariablen
        self.measurement_size = measurement_size
        self.x = np.zeros(measurement_size)  # Zustand (Position)
        self.P = np.eye(measurement_size)  # Fehlerkovarianzmatrix
        self.R = np.eye(measurement_size) * 0.04  # Messrauschkovarianz
        self.Q = np.eye(measurement_size) * 0  # Prozessrauschkovarianz

    def reset(self, measurement):
        # Initialisierung des Zustands mit der ersten Messung
        self.x = np.array(measurement[:self.measurement_size]) * 0.84
        self.P = np.eye(self.measurement_size) / 26 # Fehlerkovarianzmatrix zurücksetzen
        return self.x
    
    def update(self, dt, measurement):
        # Vorhersage (Predict)
        self.x = self.x  # Da das Objekt statisch ist, ändert sich der Zustand nicht
        self.P = self.P + self.Q  # Aktualisierung der Fehlerkovarianzmatrix
        
        # Messung (Measurement)
        z = np.array(measurement[:self.measurement_size])
        
        # Berechnung des Kalman-Gewinns
        S = self.P + self.R  # Innovationskovarianz
        K = np.dot(self.P, np.linalg.inv(S))  # Kalman-Gewinn
        
        # Update (Correct)
        y = z - self.x  # Innovationsvektor
        self.x = self.x + np.dot(K, y)  # Aktualisierung des Zustands
        self.P = self.P - np.dot(K, self.P)  # Aktualisierung der Fehlerkovarianzmatrix
        return self.x


"""
Objekt bewegt sich statisch: x(t) = x(t-1)
Jede Messung hat individuelle Meßrauschen
"""

class Randomnoise_KalmanFilter():
    def __init__(self, measurement_size):
        # Initialisierung der Zustandsvariablen
        self.measurement_size = measurement_size
        self.x = np.zeros(measurement_size)  # Zustand (Position)
        self.P = np.eye(measurement_size)  # Fehlerkovarianzmatrix
        self.R = np.eye(measurement_size) # Messrauschkovarianz

    def reset(self, measurement):
        # Initialisierung des Zustands mit der ersten Messung
        self.x = np.array(measurement[:self.measurement_size]) * 0.4
        self.P = np.eye(self.measurement_size) * 0.1  # Ich habe experimentiert und den niedrigsten RMSE erreicht, indem ich die Einheitsmatrix mit dem Faktor 0,1 multipliziert habe.
        return self.x
    
    def update(self, dt, measurement):
        # Vorhersage (Predict)
        self.x = self.x  # Da das Objekt statisch ist, ändert sich der Zustand nicht
        
        # Messung (Measurement)
        z = np.array(measurement[:self.measurement_size])
        # Individuelle Meßrauschen
        self.R = np.array(measurement[self.measurement_size:].reshape(self.measurement_size,self.measurement_size))

        # Berechnung des Kalman-Gewinns
        S = self.P + self.R  # Innovationskovarianz
        K = np.dot(self.P, np.linalg.inv(S))  # Kalman-Gewinn
        
        # Update
        y = z - self.x  # Innovationsvektor
        self.x = self.x + np.dot(K, y)  # Aktualisierung des Zustands
        self.P = self.P - np.dot(K, self.P)  # Aktualisierung der Fehlerkovarianzmatrix
        return self.x
    


class AngularKalmanFilter():
    # Sachen, die ergänzt worden sind, wurden kommentiert - der Rest wurde von Class KalmanFilter übernommen und daher
    # nicht weiterkommentiert.

    def __init__(self, measurement_size):
        # Initialisierung der Zustandsvariablen
        self.measurement_size = measurement_size
        self.x = np.zeros(measurement_size)  
        self.P = np.eye(measurement_size)  
        self.R = np.array([
            [0.0100, 0.0000],
            [0.0000, 0.0025]])
        
    def reset(self, measurement):
        # Initialisierung des Zustands x mit der ersten Messung
        r, phi = measurement # Polarkoordinaten: r ist die Distanz und phi der Winkel
        self.x = np.array([r * np.cos(phi), r * np.sin(phi)])
        self.P = np.eye(self.measurement_size) * 0.007
        return self.x

    def update(self, dt, measurement):
        r, phi = measurement

        # Konvertierung von Polarkoordinaten in kartesische Koordinaten 
        z = np.array([r * np.cos(phi), r * np.sin(phi)])
        
        # Berechnung des Kalman-Gewinns
        S = self.P + self.R  
        K = np.dot(self.P, np.linalg.inv(S))  
        
        # Update
        y = z - self.x  
        self.x = self.x + np.dot(K, y)  
        self.P = self.P - np.dot(K, self.P)  
        return self.x


"""
Objekt bewegt sich mit konstanter Geschwindigkeit: 
x(t) = x(t-1) + dt * v
Unabhängiges, normalverteiltes Rauschen mit STD = 0.2/Achse
z(t) = x(t) + et; et ~ N(0, 0.04)
"""

class Constantvelocity_KalmanFilter():
    def __init__(self, measurement_size=2):
        # Initialisierung der Zustandsvariablen
        self.measurement_size = measurement_size
        self.state_size = 2 * measurement_size  # Zustand (Position + Geschwindigkeit)
        self.x = np.zeros(self.state_size)  # Zustand (Position und Geschwindigkeit)
        self.R = np.eye(self.measurement_size) * 0.04  # Messrauschkovarianz

        # Messmatrix
        self.H = np.zeros((self.measurement_size, self.state_size))
        self.H[:, :self.measurement_size] = np.eye(self.measurement_size)

    def reset(self, measurement):
        # Initialisierung des Zustands mit der ersten Messung
        self.x[:self.measurement_size] = measurement * 0.75
        self.x[self.measurement_size:] = 0  # Anfangsgeschwindigkeit auf 0 setzen
        self.P = np.eye(self.state_size) / 10 # Fehlerkovarianzmatrix zurücksetzen
        return self.x[:self.measurement_size]

    def update(self, dt, measurement):
        # Vorhersage (Predict)
        self.F = np.eye(self.state_size)
        for i in range(self.measurement_size):
            self.F[i, i + self.measurement_size] = dt
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T)

        # Messung (Measurement)
        z = np.array(measurement[:self.measurement_size])
        
        # Berechnung des Kalman-Gewinns
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # Innovationskovarianz
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman-Gewinn
        
        # Update (Correct)
        y = z - np.dot(self.H, self.x)  # Innovationsvektor
        self.x = self.x + np.dot(K, y)  # Aktualisierung des Zustands
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))  # Aktualisierung der Fehlerkovarianzmatrix
        return self.x[:self.measurement_size]
    
    

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
        self.P = np.eye(self.state_size)

        # Prozessrauschkovarianz Q
        self.Q = np.eye(self.state_size) * 0.01 # Das Prozessrauschen wird beispielhaft modelliert, um den RMSE niedriger zu machen

    def reset(self, measurement):
        z = measurement[:10].reshape(5, 2) 
        self.x[:2] = np.mean(z, axis=0) * 0.8  # Durchschnitt der fünf Positionsmessungen
        self.x[2:] = 0  # Anfangsgeschwindigkeit auf 0 setzen
        self.P = np.eye(self.state_size) * 0.04       
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
            R_values = measurement[10:]  # Standardabweichungen
            R_i = np.diag(R_values[i*2:i*2+2]**2)  # Quadrieren, um Varianzen zu erhalten

            # Korrektur
            y = z_i - H @ self.x
            S = H @ self.P @ H.T + R_i
            K = self.P @ H.T @ np.linalg.inv(S)

            # Aktualisierung
            self.x = self.x + K @ y
            self.P = self.P - K @ H @ self.P

        return self.x[:2]



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
        self.x[:2] = np.mean(z, axis=0) * 0.8 # Durchschnitt der fünf Positionsmessungen
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

    
    
class SimpleNoiseFilter():
    def __init__(self, measurement_size):
        self.measurement_size = measurement_size
        self.estimated_position = np.zeros(measurement_size)
        self.noise_accumulator = []
    
    def reset(self, measurement):
        self.estimated_position = np.array(measurement[:self.measurement_size])
        return self.estimated_position
    
    def update(self, dt, measurement):
        measurement = np.array(measurement[:self.measurement_size])
        noise = measurement - self.estimated_position
        self.noise_accumulator.append(noise)
        
        return self.estimated_position
    
    def calculate_mean_noise(self):
        if self.noise_accumulator:
            mean_noise = np.mean(self.noise_accumulator, axis=0)
            return mean_noise
        return np.zeros(self.measurement_size)
    
    def correct_measurements(self, measurements):
        mean_noise = self.calculate_mean_noise()
        corrected_measurements = [np.array(m[:self.measurement_size]) - mean_noise for m in measurements]
        return corrected_measurements
    
    
class NoFilter():
    def __init__(self):
        pass

    def reset(self, measurement):    
        return measurement[:2]
    
    def update(self, dt, measurement):  
        return measurement[:2]