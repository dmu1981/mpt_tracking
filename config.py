import numpy as np
import dummy
#from nam import NamFilter
import constant_position
import random_noise
import nam
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
        #"constantposition": constant_position.ConstantPositionFilter(),
        "randomnoise": nam.RandomNoiseFilter(),
        "angular": nam.AngularFilter() ,
        "constantvelocity": dummy.DummyFilter(2),
        "constantvelocity2": dummy.DummyFilter(2),
        "constantturn": nam.ConstantTurnFilter(turn_rate=0.1),

    },
    
    
}
if __name__ == "__main__":
    print("This is the config file. Run main.py to start the simulation.")
