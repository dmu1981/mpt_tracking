import dummy
from filter_nick_redion import RandomNoise, ConstantVelocity, KalmanFilterConstantTurn, KalmanFilter, ExtendedKalmanFilter, KalmanFilterConstantVelocity

# TODO: Add your filters here
filters = {
    "Redion_Nick": {
        "color": [0.2, 0.2, 0.4],
        "constantposition": KalmanFilter(2),
        "constantvelocity": KalmanFilterConstantVelocity(2),
        "constantvelocity2": ConstantVelocity(2),
        "constantturn": KalmanFilterConstantTurn(2),
        "randomnoise": RandomNoise(2),
        "angular": ExtendedKalmanFilter(2),
    }
}

