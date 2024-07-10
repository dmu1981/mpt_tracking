# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 03:38:31 2024

@author: marko
"""
import numpy as np
from math import sin, cos

class ConstantVelocity2:
    def __init__(self, guess_H = 1, guess_P = 1, Q_noise = 0.0005):
        self.dt = 1
        self.state_estimate = np.zeros(4)
        self.guess_P = guess_P
        self.guess_H = guess_H
        self.Q_noise = Q_noise
        self.P = np.eye(4) * guess_P
        self.H = np.eye(2,4) * guess_H #measurement_matrix
        self.Q = np.eye(4) * self.Q_noise
        self.I = np.eye(4) #einheitsmatrix
        
        
        """in den folien:
             state = F@state + Ga
             mit:
                 Ga = np.array([[0.5*dt**2, 0],
                                [0, 0.5*dt**2], 
                                [dt, 0],
                                [0, dt]])
                 Q = np.ndarray.var(Ga)"""
            

    def reset(self, measurement):
        self.state_estimate[:2] = np.mean(measurement[:10].reshape(-1, 2), axis=0) #positions
        self.state_estimate[2:] = 0  # velocity: unknown
        self.P = np.eye(4) # uncertainty covariance
        return self.state_estimate[:2]
    
    def update(self, dt, measurement):
        #Prediction
        self.dt = dt
        
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]) # process model
        
        G = np.array([[0.5*dt**2, 0],
                       [0, 0.5*dt**2], 
                       [dt, 0],
                       [0, dt]])
        
        Q = np.ndarray.var(G) * self.Q_noise
        
        self.state_estimate = F @ self.state_estimate 
        
        self.P = F @ self.P @ F.T + Q #has shape (4,4)
        
        #getting values
        measured_values = measurement[:10].reshape(-1, 2)
        R = measurement[10:].reshape(-1, 2) #measurement_noise_covariance/measurement_noise
        
        #avg because we have multiple measurements
        avg_value = np.mean(measured_values, axis = 0)
        avg_R = np.ones((2,2))
        
        n_measured_values = len(measured_values)
        
        for i in range(n_measured_values):
            avg_R *= np.diag(R[i]) **2
        avg_R = avg_R/ len(measured_values)

        #-------calculating----------
        
        #Innovation
        residual_covariance = self.H @ self.P @ self.H.T + avg_R # means S
        #KalmanGain
        kalman_gain = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(residual_covariance))
        
        measurement_residual = avg_value - self.H @ self.state_estimate # innovation
        
        #Update
        self.state_estimate = self.state_estimate + kalman_gain @ measurement_residual
        self.P = np.dot((self.I - np.dot(kalman_gain, self.H)), self.P)
        
        return self.state_estimate[:2]
    
class ConstantVelocity():
    
    def __init__(self, guess_H = 1, guess_P = 1, Q_noise = 0.0005):
        self.dt = 1
        self.state_estimate = np.zeros(4)
        self.guess_P = guess_P
        self.guess_H = guess_H
        self.Q_noise = Q_noise
        self.H = np.eye(2,4)
        self.Q = np.eye(2) * self.Q_noise            

    def reset(self, measurement):
        self.state_estimate[:2] = np.mean(measurement[:10].reshape(-1, 2), axis=0) #positions
        self.state_estimate[2:] = 0  # velocity: unknown
        self.P = np.eye(4)/2 # uncertainty covariance
        return self.state_estimate[:2]
    
    def update(self, dt, measurement):
        
        #Prediciton
        self.dt = dt
        
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        
        Q = np.eye(4) * self.Q_noise * dt
        
        self.state_estimate = F @ self.state_estimate
        
        self.P = F @ self.P @ F.T + Q 
        
        #getting values
        measured_values = np.mean(measurement[:10].reshape(-1, 2), axis=0)
        R = np.eye(2) * 0.5**2 #measurement_noise_covariance/measurement_noise
        
        #Innovation
        residual_covariance = self.H @ self.P @ self.H.T + R # means S
        
        #KalmanGain
        kalman_gain = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(residual_covariance))
        
        measurement_residual = measured_values - self.H @ self.state_estimate # innovation
        
        #Update
        self.state_estimate = self.state_estimate + kalman_gain @ measurement_residual
        
        I = np.eye(4)
        
        self.P = np.dot((I - np.dot(kalman_gain, self.H)), self.P)
        
        
        return self.state_estimate[:2]
    