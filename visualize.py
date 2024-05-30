from matplotlib import pyplot as plt
import matplotlib.collections as mcoll
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def generate_cmap(targetColor):
    cdict = {
        "red": [
            [0.0, 1.0, 1.0],
            [0.5, 0.9, 0.9],
            [1.0, targetColor[0], targetColor[0]],
        ],
        "green": [
            [0.0, 1.0, 1.0],
            [0.5, 0.9, 0.9],
            [1.0, targetColor[1], targetColor[1]],
        ],
        "blue": [
            [0.0, 1.0, 1.0],
            [0.5, 0.9, 0.9],
            [1.0, targetColor[2], targetColor[2]],
        ],
    }

    return LinearSegmentedColormap("custommap", cdict)


def colorline(
    x,
    y,
    z=None,
    cmap=plt.get_cmap("copper"),
    norm=plt.Normalize(0.0, 1.0),
    linewidth=3,
    alpha=1.0,
):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(
        segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha
    )

    #    ax = plt.gca()
    #   ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def getBoundaries(states, results):
    xMin, xMax = np.min(states[:, 0]), np.max(states[:, 0])
    yMin, yMax = np.min(states[:, 1]), np.max(states[:, 1])

    return xMin, yMin, xMax, yMax


def draw(timeseries, args, filters, results, scores):
    def draw_till(axs, timestep):
        states = timeseries[args.index]["targets"]
        measurements = timeseries[args.index]["measurements"]
        axs[0].set_title("States, Predictions, Measurements")

        axs[0].plot(states[timestep, 0], states[timestep, 1], "k*")
        legend_entries = ["True States"]
        for teams in filters.keys():
            legend_entries.append(teams)
            axs[0].plot(
                results[teams][timestep, 0],
                results[teams][timestep, 1],
                "*",
                color=filters[teams]["color"],
            )

        for teams in filters.keys():
            axs[0].add_collection(
                colorline(
                    results[teams][:timestep, 0],
                    results[teams][:timestep, 1],
                    cmap=generate_cmap(filters[teams]["color"]),
                )
            )

        axs[0].legend(legend_entries)

        axs[0].add_collection(
            colorline(
                states[:timestep, 0],
                states[:timestep, 1],
                cmap=generate_cmap([0.0, 0.0, 0.0]),
            )
        )

        xMin, yMin, xMax, yMax = getBoundaries(states, results)
        axs[0].set_xlim([xMin - 0.2, xMax + 0.2])
        axs[0].set_ylim([yMin - 0.2, yMax + 0.2])

        axs[1].set_title("RMSE")
        legend_entries = []
        for teams in filters.keys():
            if "errors" in scores[teams]:
                axs[1].plot(
                    scores[teams]["errors"][:timestep], color=filters[teams]["color"]
                )
                legend_entries.append(teams)

        if "errors" in scores[teams]:
            l = scores[teams]["errors"].shape[0]
            for teams in filters.keys():
                axs[1].plot(
                    [0, l],
                    [scores[teams]["rmse"], scores[teams]["rmse"]],
                    "--",
                    color=filters[teams]["color"],
                )

            axs[1].legend(legend_entries)

    total = timeseries[args.index]["targets"].shape[0]
    fig, axs = plt.subplots(1, 2)
    # print(results)
    for i in range(total):
        axs[0].clear()
        axs[1].clear()
        draw_till(axs, i)
        plt.pause(0.01)
