import numpy as np
from matplotlib import pyplot as plt
import pandas


def print_score_statistics(scores, args, multiruns):
    # Create an ordered list of teams
    ordered = []
    for team in scores.keys():
        rmse = scores[team]["rmse"]
        ordered.append((team, rmse))

    # Sort them
    ordered.sort(key=lambda x: x[1])

    # Print (best team on top)
    rmse_per_run = None
    df = {}
    for team, rmse in ordered:
        bestRunIndex = np.argmin(scores[team]["rmse_per_run"])
        worstRunIndex = np.argmax(scores[team]["rmse_per_run"])
        if multiruns:
            print(
                f"   {team:10}: {rmse:10.4} (Best run was index {bestRunIndex}, Worst run was index {worstRunIndex})"
            )
        else:
            print(f"   {team:10}: {rmse:10.4}")

        df[team] = scores[team]["rmse_per_run"]

    if args.debug and args.all is True:
        df["indices"] = range(20)
        df = pandas.DataFrame(df)
        df.plot(x="indices", y=scores.keys(), kind="bar")
        plt.show()

    return ordered
