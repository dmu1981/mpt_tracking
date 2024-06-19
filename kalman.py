import numpy as np

"""
The Kalman filter is a recursive algorithm that estimates the state of a dynamic system over time.
It combines measurements (which are often noisy and inaccurate) with a model of the system to provide a more accurate estimate of the state.
"""

class KalmanFilter:
    def __init__(self, shape):
        self.shape = shape
        self.x = np.zeros(shape)  # initial state (location and velocity)
        self.P = np.eye(shape[0])  # uncertainty covariance
        self.F = np.eye(shape[0])  # state transition matrix
        self.H = np.eye(shape[0])  # measurement matrix
        self.R = np.eye(shape[0]) * 0.04  # measurement uncertainty (noise) matrix
        self.I = np.eye(shape[0])  # identity matrix
        
    def reset(self, measurement):
        self.x = measurement[:2]  # initialize state with first measurement
        self.P = np.eye(2)  # reset uncertainty covariance
        return self.x
    
    def update(self, dt, measurement):
        Z = measurement[:2]  # measurement vector / measured value
        y = Z - np.dot(self.H, self.x)  # measurement residual: difference between measured and estimated values
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # residual covariance: uncertainty of measurement residual
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman gain: weighting of measurement residual to update stategit
        self.x = self.x + np.dot(K, y)  # update of state estimate: addition of weighted measurement residual
        self.P = np.dot((self.I - np.dot(K, self.H)), self.P)  # updated estimate covariance: reduction of uncertainty
        return self.x
