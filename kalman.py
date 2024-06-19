import numpy as np

class KalmanFilter:
    def __init__(self, shape):
        self.shape = shape
        self.x = np.zeros(shape)  
        self.uc = np.eye(shape[0])
        self.st = np.eye(shape[0])
        self.H = np.eye(shape[0]) 
        self.R = np.eye(shape[0]) * 0.04
        self.I = np.eye(shape[0])
        
    def reset(self, measurement):
        self.x = measurement[:2]
        self.uc = np.eye(2)
        return self.x
    
    def update(self, dt, measurement):
        Z = measurement[:2]
        y = Z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.uc, self.H.T)) + self.R
        K = np.dot(np.dot(self.uc, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.uc = np.dot((self.I - np.dot(K, self.H)), self.uc)
        return self.x
