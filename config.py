import dummy
from filter_nick_redion import RandomNoise, ConstantVelocity, ConstantTurn, KalmanFilterConstantTurn, EKF

# TODO: Add your filters here
filters = {
    "Redion_Nick": {
        "color": [0.2, 0.2, 0.4],
        "constantposition": dummy.DummyFilter(2),
        "constantvelocity": dummy.DummyFilter(2),
        "constantvelocity2": ConstantVelocity(2),
        "constantturn": KalmanFilterConstantTurn(2),
        "randomnoise": RandomNoise(2),
        "angular": dummy.DummyFilter(2),
    },
    "Test": {
        "color": [0.5, 0.5, 0.1],
        "consantturn": ConstantTurn(2)
    },
    "Extended Kalman Filter": {
        "color": [0.7, 0.2, 0.3],
        "constantturn": ConstantVelocity(2)
    }
}

