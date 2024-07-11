import numpy as np


class ConstantPositionFilter:
    def __init__(self):
        pass

    def reset(self, measurement):
        self.X = measurement
        self.P = np.eye(2)

        return self.X

    def update(self, dt, measurement):
        F = np.eye(2)  # Constant process model
        Q = np.zeros(2)  # No process uncertainty

        # Prediction
        self.X = F @ self.X
        self.P = F @ self.P @ F.T + Q

        # Update
        H = np.eye(2)  # Direct measurement
        y = measurement - H @ self.X

        R = np.eye(2) * 0.04  # Given measurement noise from task description

        # Innovation variance and kalman gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Updated state and uncertainty
        self.X += K @ y
        self.P = (np.eye(2) - K @ H) @ self.P

        return self.X


class RandomNoiseFilter:
    def __init__(self):
        pass

    def reset(self, measurement):
        self.X = measurement[:2]
        self.P = measurement[2:].reshape(2, 2)

        return self.X

    def update(self, dt, measurement):
        F = np.eye(2)  # Constant process model
        Q = np.zeros(2)  # No process uncertainty

        # Prediction
        self.X = F @ self.X
        self.P = F @ self.P @ F.T + Q

        # Update
        H = np.eye(2)  # Direct measurement
        y = measurement[:2] - H @ self.X
        R = measurement[2:].reshape(2, 2)

        # Innovation variance and kalman gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Updated state and uncertainty
        self.X += K @ y
        self.P = (np.eye(2) - K @ H) @ self.P

        return self.X


class AngularFilter:
    def __init__(self):
        pass

    def reset(self, measurement):
        self.X = measurement[:2]
        x = self.X[0] * np.cos(self.X[1])
        y = self.X[0] * np.sin(self.X[1])
        self.X = np.array([x, y])
        self.P = np.eye(2, 2) * 0.0005

        return self.X

    def update(self, dt, measurement):
        F = np.eye(2)  # Constant process model
        Q = np.zeros(2)  # No process uncertainty

        # Prediction
        self.X = F @ self.X
        self.P = F @ self.P @ F.T + Q

        # Update
        ## r = sqrt(x^2+y^2)
        ## phi = arctan2(x, y)
        ## dr/dx = x/sqrt(x^2+y^2)
        ## dr/dy = y/sqrt(x^2+y^2)
        ## dphi/dx = -y/(x^2+y^2)
        ## dphi/dy = x/(x^2+y^2)

        rSquared = np.sum(np.power(self.X, 2))
        r = np.sqrt(rSquared)
        H = np.array(
            [
                [self.X[0] / r, self.X[1] / r],
                [-self.X[1] / rSquared, self.X[0] / rSquared],
            ]
        )

        phi = np.arctan2(self.X[1], self.X[0])
        yPred = np.array([r, phi])
        y = measurement[:2] - yPred
        R = np.array([[0.01, 0.0], [0.0, 0.0025]])

        # Innovation variance and kalman gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Updated state and uncertainty
        self.X += K @ y
        self.P = (np.eye(2) - K @ H) @ self.P

        return self.X


class ConstantVelocityFilter:
    def __init__(self):
        self.firstMeasurement = None
        self.secondMeasurement = None
        pass

    def reset(self, measurement):
        self.P = np.eye(4) * 0.04
        self.X = np.array([measurement[0], measurement[1], 0.0, 0.0])

        return measurement[:2]

    def update(self, dt, measurement):
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        Q = np.eye(4) * 1e-6

        # Prediction
        self.X = F @ self.X
        self.P = F @ self.P @ F.T + Q

        # Update
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        y = measurement - H @ self.X

        R = np.eye(2) * 0.04  # Given measurement noise from task description

        # Innovation variance and kalman gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Updated state and uncertainty
        self.X += K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        return self.X[:2]


class ConstantVelocityFilter2:
    def __init__(self):
        self.firstMeasurement = None
        self.secondMeasurement = None
        pass

    def reset(self, measurement):
        self.P = np.eye(4)
        self.X = np.array(
            [
                (
                    measurement[0]
                    + measurement[2]
                    + measurement[4]
                    + measurement[6]
                    + measurement[8]
                )
                / 5.0,
                (
                    measurement[1]
                    + measurement[3]
                    + measurement[5]
                    + measurement[7]
                    + measurement[9]
                )
                / 5.0,
                0.0,
                0.0,
            ]
        )

        return self.X[:2]

    def update(self, dt, measurement):
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        Q = np.eye(4) * 1e-6

        # Prediction
        self.X = F @ self.X
        self.P = F @ self.P @ F.T + Q

        # Update
        H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        y = measurement[:10] - H @ self.X
        R = np.diag(measurement[10:])

        # R = np.eye(10) * 0.04 # Given measurement noise from task description
        # print(R)

        # Innovation variance and kalman gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Updated state and uncertainty
        self.X += K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        return self.X[:2]


class ConstantTurnRate:
    def __init__(self):
        self.firstMeasurement = None
        self.secondMeasurement = None
        pass

    def reset(self, measurement):
        self.P = np.eye(7)
        self.X = np.array(
            [
                (
                    measurement[0]
                    + measurement[2]
                    + measurement[4]
                    + measurement[6]
                    + measurement[8]
                )
                / 5.0,
                (
                    measurement[1]
                    + measurement[3]
                    + measurement[5]
                    + measurement[7]
                    + measurement[9]
                )
                / 5.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        return self.X[:2]

    def update(self, dt, measurement):
        a = self.X[6]

        F = np.array(
            [
                [1.0, 0.0, dt, 0.0, 0.5 * dt**2, 0.0, 0.0],
                [0.0, 1.0, 0.0, dt, 0.0, 0.5 * dt**2, 0.0],
                [0.0, 0.0, 1.0, 0.0, dt, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, dt, 0.0],
                [0.0, 0.0, 0.0, 0.0, np.cos(a * dt), -np.sin(a * dt), 0.0],
                [0.0, 0.0, 0.0, 0.0, np.sin(a * dt), np.cos(a * dt), 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

        Q = np.eye(7) * 1e-2

        # Prediction
        self.X = F @ self.X
        self.P = F @ self.P @ F.T + Q

        # Update
        H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        y = measurement[:10] - H @ self.X
        R = np.diag(measurement[10:])

        # R = np.eye(10) * 0.04 # Given measurement noise from task description
        # print(R)

        # Innovation variance and kalman gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Updated state and uncertainty
        self.X += K @ y
        self.P = (np.eye(7) - K @ H) @ self.P
        return self.X[:2]

