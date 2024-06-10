import dummy
from nam import NamFilter
import numpy as np
from dummy import ConstantPositionFilter

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
        "constantposition": NamFilter(),
    },
    "ConstantPositionFilter": {
        "color": [0.1, 0.6, 0.1],
        "constantposition": ConstantPositionFilter(),
    },
}
