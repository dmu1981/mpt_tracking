# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 03:38:31 2024

@author: marko
"""
import numpy as np


class ConstantVelocity2:
    """
    This class works more or less the same as the normal Kalman filter.
    Only change here was the addition of the measurement noise covariance,
    which quantifies the level of confidence in the measurements
    and therefore is needed to update the state estimate and its uncertainty/noise.
    """
    def __init__(self, guess_H = 1, guess_P = 1):
        #self.state_dim = state_dim
        self.dt = 0
        self.state_estimate = np.zeros(4)
        self.P = np.eye(4) #* guess_P        
        #self.state_transition = np.eye(4)
        self.H = np.eye(2,4) #* guess_H #measurement_matrix
        self.Q = np.eye(4) 
        self.guess_P = guess_P
        self.guess_H = guess_H
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
        self.P = np.eye(4) # uncartainty covariance
        return self.state_estimate[:2]
    
    def update(self, dt, measurement):
        #Prediciton
        self.dt = dt
        
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]) #Prozessmodell
        
        self.state_estimate = self.F @ self.state_estimate
        
        Q = self.Q #* self.guess_Q
        
        self.P =F @ self.P @ F.T + Q #hat shape (4,4)
        
        #getting values
        measured_values = measurement[:10].reshape(-1, 2)
        R = measurement[10:].reshape(-1, 2) #measurement_noise_covariance/measurement_noise
        #R = np.diag(np.concatenate([R_values[:, 0]**2, R_values[:, 1]**2]))
        
        #avg because we have multiple measurements
        avg_value = np.mean(measured_values, axis = 0)
        avg_R = np.ones((2,2))
        
        n_measured_values = len(measured_values)
        
        for i in range(n_measured_values):
            avg_R *= np.diag(R[i]) **2
        avg_R /= len(measured_values)

        #-------calculating----------
        
        #Innovation
        #H = np.eye(2,4)
        residual_covariance = self.H @ self.P @ self.H.T + avg_R # means S
        #KalmanGain
        kalman_gain = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(residual_covariance))
        
        measurement_residual = avg_value - self.H @ self.state_estimate # innovation
        
        #Update
        self.state_estimate = self.state_estimate + kalman_gain @ measurement_residual
        self.P = np.dot((self.I - np.dot(kalman_gain, self.H)), self.P)
        
        return self.state_estimate[:2]#.reshape(1, 2)
    
    

    