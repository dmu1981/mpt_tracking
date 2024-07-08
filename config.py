import numpy as np
import dummy
import random_noise
import angular 
import constantvelocity
import constantvelocity2
import nam
import constant_turn


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
    "NAMTeam": {
        "color": [0.5, 0.1, 0.9],
        "constantposition": nam.NamFilter(),
        "randomnoise": nam.RandomNoiseFilter(),
        "angular": nam.AngularFilter() 
        "constantvelocity": nam.KalmanFilterConstantVelocity(),
        "constantvelocity2": nam.KalmanFilterConstantVelocityMultiple(),
        "constantturn": nam.ConstantTurnFilter(),
  

    },
}
if __name__ == "__main__":
    print("This is the config file. Run main.py to start the simulation.")
