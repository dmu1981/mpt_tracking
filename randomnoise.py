import numpy as np

class RandomNoiseFilter():
    def __init__(self, measurement_size):
        # Initialize state variables
        self.measurement_size = measurement_size
        self.x = np.zeros(measurement_size)  # State (position)
        self.P = np.eye(measurement_size)  # Error covariance matrix

    def reset(self, measurement):
        # Initialize the state with the first measurement
        self.x = np.array(measurement[:self.measurement_size])
        self.P = np.array(measurement[self.measurement_size:]).reshape(self.measurement_size, self.measurement_size)
        return self.x
    
    def update(self, measurement):
        # Measurement (Measurement)
        z = np.array(measurement[:self.measurement_size])
        R = np.array(measurement[self.measurement_size:]).reshape(self.measurement_size, self.measurement_size)
        
        # Compute the inverse covariance matrices
        P_inv = np.linalg.inv(self.P)
        R_inv = np.linalg.inv(R)
        
        # Fusion of the covariance matrices
        P_fused = np.linalg.inv(P_inv + R_inv)
        
        # Fusion of the states
        self.x = P_fused @ (P_inv @ self.x + R_inv @ z)
        
        # Update the error covariance matrix
        self.P = P_fused
        
        return self.x
