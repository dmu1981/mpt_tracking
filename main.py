import pickle
import dummy
from matplotlib import pyplot as plt
from replay import replay
from evaluate import evaluate
from config import filters
from scorestatistics import print_score_statistics
import numpy as np
import argparse


# No need to touch anything below this line
# ---------------------------------------------------


def run_mode(mode):
    file = mode + ".pk"

    # Make sure index is an integer
    args.index = int(args.index)

    # Load the timeseries data from a pickle file
    with open(file, "rb") as f:
        timeseries = pickle.loads(f.read())

    #args.all is a boolean, if its True --> process all time series, 
    # initialize the RMSE scores and compute summirized RMSE & RMSE per run
    # if args.all is False, check the index, process the given time series 
    # and calculate their RMSE.
    
    if args.all:
        scores = {}
        for teams in filters.keys():
            scores[teams] = {}
            scores[teams]["rmse"] = 0.0
            scores[teams]["rmse_per_run"] = []

        for series in timeseries:
            results = replay(filters, mode, series)
            intermediate = evaluate(series["targets"], results)

            for teams in filters.keys():
                scores[teams]["rmse_per_run"].append(intermediate[teams]["rmse"])
                scores[teams]["rmse"] += intermediate[teams]["rmse"]

        # Turn scores_per_run into numpy array
        for teams in filters.keys():
            scores[teams]["rmse_per_run"] = np.array(scores[teams]["rmse_per_run"])

        for teams in filters.keys():
            scores[teams]["rmse"] /= len(timeseries)

        ordered = print_score_statistics(scores, args, multiruns=True)
    else:
        if args.index >= len(timeseries) or args.index < 0:
            print("Invalid index")
            exit()

        results = replay(filters, mode, timeseries[args.index])
        scores = evaluate(timeseries[args.index]["targets"], results)

        for teams in filters.keys():
            rmse = scores[teams]["rmse"]
            scores[teams]["rmse_per_run"] = [rmse]

        ordered = print_score_statistics(scores, args, multiruns=False)

    if args.debug and not args.all:
        import visualize

        visualize.draw(timeseries, args, filters, results, scores)

    return scores, ordered


list_of_modes = {
    "constantposition": 0,
    "constantvelocity": 0,
    "constantvelocity2": 0,
    "constantturn": 0,
    "randomnoise": 0,
    "angular": 0,
}

# Sanity check filters
for team in filters.keys():
    res = filters[team]
    if "color" not in res:
        print(f"Team {team}: You must specify a color in config.py")
        exit()

    if type(res["color"]) != list or len(res["color"]) != 3:
        print(f"Team {team}: Color must be a 3-element list like [1.0, 0.0, 0.0]")
        exit()

    for mode in list_of_modes.keys():
        if mode not in res:
            print(
                f"Team {team}: You did not specify a filter for mode {mode}... replacing with Dummy Filter"
            )
            filters[team][mode] = (dummy.DummyFilter(2),)


# Create an argument parser

parser = argparse.ArgumentParser(description="MPT Replay Tool")
parser.add_argument("--mode", action="store")
parser.add_argument("--index", action="store", default=0)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--all", action="store_true")

# Parse command line
args = parser.parse_args()

# Is there a file given?
if args.mode is None:
    print("Must specify mode")
    exit()

if args.mode not in list_of_modes.keys():
    print("Unknown mode: ", args.mode)
    exit()

if args.mode == "all":
    s = set()
    for team in filters.keys():
        for k in filters[team].keys():
            if k != "color":
                s.add(k)
    modes = list(s)

    scores = {}
    for teams in filters.keys():
        scores[teams] = {}
        scores[teams]["rmse"] = 0.0
        scores[teams]["rmse_per_run"] = [0]

    for mode in modes:
        print(mode)
        score, ordered = run_mode(mode)

        # Do a ranked statistic for the overall score
        for index, (team, score) in enumerate(ordered):
            scores[team]["rmse"] += index

        print("")

    print("overall")
    print_score_statistics(scores, args, multiruns=False)
    # print(scores)
else:
    scores, ordered = run_mode(args.mode)
