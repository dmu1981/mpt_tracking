import numpy as np

class AngularKalmanFilter():
    # Sachen, die ergänzt worden sind, wurden kommentiert - der Rest wurde von Class KalmanFilter übernommen und daher
    # nicht weiterkommentiert.

    def __init__(self, measurement_size):
        # Initialisierung der Zustandsvariablen
        self.measurement_size = measurement_size
        self.x = np.zeros(measurement_size)  
        self.P = np.eye(measurement_size)  
        self.R = np.eye(measurement_size) * 0.04  
        #self.Q = np.eye(measurement_size) * 0.00  

    def reset(self, measurement):
        # Initialisierung des Zustands x mit der ersten Messung
        r, phi = measurement # Polarkoordinaten: r ist die Distanz und phi der Winkel
        self.x = np.array([r * np.cos(phi), r * np.sin(phi)])
        self.P = np.eye(self.measurement_size)  # Zurücksetzen der Fehlerkovarianzmatrix P
        return self.x

    def update(self, dt, measurement):
        r, phi = measurement

        # Konvertierung von Polarkoordinaten in kortesische Koordinaten 
        z = np.array([r * np.cos(phi), r * np.sin(phi)])
        
        # Vorhersage (Predict)
        self.P = self.P #+ self.Q  # Aktualisierung der Fehlerkovarianzmatrix
        
        # Berechnung des Kalman-Gewinns
        S = self.P + self.R  
        K = np.dot(self.P, np.linalg.inv(S))  
        
        # Update
        y = z - self.x  
        self.x = self.x + np.dot(K, y)  
        self.P = self.P - np.dot(K, self.P)  
        return self.x
