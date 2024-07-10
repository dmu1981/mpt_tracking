import dummy
import constanposition
import randomnoise

# TODO: Add your filters here

filters = {
    "Dummy":{
        "color": [0.2, 0.2, 0.4],
        "constantposition": dummy.DummyFilter(2),
        "constantvelocity": dummy.DummyFilter(2),
        "constantvelocity2": dummy.DummyFilter(2),
        "constantturn": dummy.DummyFilter(2),
        "randomnoise": dummy.DummyFilter(2),
        "angular": dummy.DummyFilter(2),
    },
    "NoFilter": {
        "color": [0.3, 0.3, 0.4],
        "constantposition": constanposition.NoFilter(),
        "randomnoise": dummy.DummyFilter(2),
    },
    "KalmanFilter": {
        "color": [0.3, 0.3, 0.4],
        "constantposition": constanposition.KalmanFilter(2),
        "randomnoise": dummy.DummyFilter(2),
    },
    "SimpleNoiseFilter":{
        "color": [0.1, 0.2, 0.3],
        "constantposition": constanposition.SimpleNoiseFilter(2),
        "randomnoise": dummy.DummyFilter(2),
    },
    "RandomNoiseFilter": {
        "color": [0.1, 0.2, 0.3],
        "constantposition": dummy.DummyFilter(2),
        "randomnoise": randomnoise.RandomNoiseFilter(2),
    },
    "AngularKalmanFilter": {
        "color": [0.1, 0.2, 0.3],
        "constantposition": constanposition.AngularKalmanFilter(2),
        "randomnoise": randomnoise.RandomNoiseFilter(2),
    }
}
# Aufruf mit python main.py --mode=constantposition --index=5
# Visualisierung python main.py --mode=constantposition --debug