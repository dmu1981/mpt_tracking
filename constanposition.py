import numpy as np

"""
Objekt bewegt sich statisch: x(t) = x(t-1)
Unabhängiges, normalverteiltes Rauschen mit STD = 0.2/Achse
z(t) = x(t) + et; et ~ N(0, 0.04)
"""

class KalmanFilter():
    def __init__(self, measurement_size):
        # Initialisierung der Zustandsvariablen
        self.measurement_size = measurement_size
        self.x = np.zeros(measurement_size)  # Zustand (Position)
        self.P = np.eye(measurement_size)  # Fehlerkovarianzmatrix
        self.R = np.eye(measurement_size) * 0.04  # Messrauschkovarianz
        self.Q = np.eye(measurement_size) * 0.00  # Prozessrauschkovarianz

    def reset(self, measurement):
        # Initialisierung des Zustands mit der ersten Messung
        self.x = np.array(measurement[:self.measurement_size])
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