import dummy
import kalman

# TODO: Add your filters here
filters = {
    "Dummy": {
        "color": [0.2, 0.2, 0.4],
        "constantposition": dummy.DummyFilter(2),
        "constantvelocity": dummy.DummyFilter(2),
        "constantvelocity2": dummy.DummyFilter(2),
        "constantturn": dummy.DummyFilter(2),
        "randomnoise": dummy.DummyFilter(2),
        "angular": dummy.DummyFilter(2),
    },
    "Nicolucas": {
        "color": [0.6, 0.6, 0.2],
        "constantposition": kalman.KalmanFilter(2),
        "constantvelocity": kalman.KalmanFilter(2),
        "constantvelocity2": kalman.KalmanFilter(2),
        "constantturn": kalman.KalmanFilter(2),
        "randomnoise": kalman.KalmanFilter(2),
        "angular": kalman.KalmanFilter(2)
    }
}
