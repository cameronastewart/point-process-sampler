import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar

# config
FIGURE_SCALE = 0.53371286247998576765700053371286
FIGURE_ESS = 1000
BATCH_SIZE = 3000
ESS_MULTIPLIER = FIGURE_ESS / BATCH_SIZE ** 2
RUNS_PER_SETTING = 10

# t-critical value for 95% confidence interval with RUNS_PER_SETTING - 1 degrees of freedom
T_VALUE = 2.26215716

# sampler properties (file prefix, plot colour, marker shape, plot zorder)
SAMPLERS = [("point", "#000000", "d", 6),
            ("birthdeath", "#EE7733", "s", 2),
            ("sqrt", "#009988", "o", 3),
            ("metropolis", "#0077BB", "^", 5),
            ("barker", "#EE3377", "v", 4)]

# plot setup
plt.rcParams.update({"text.usetex": True,
                     "font.family": "serif",
                     "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
                     "font.size": 7 / FIGURE_SCALE})
plt.rc("axes", labelsize=9 / FIGURE_SCALE)
plt.rc("legend", fontsize=9 / FIGURE_SCALE)

fig, axes = plt.subplots(2, 3, sharex="col")
fig.set_size_inches(13, 8)
fig.subplots_adjust(hspace=0.05, wspace=0.12)

# loads data from CSV files and plots ESS and ESS/Second
def loadAndPlot(model, scales, axTop, axBottom, xLabel):
    for sampler, colour, marker, zorder in SAMPLERS:
        # data shape (scaling parameter values = 21, runs per parameter = 10, data points per run 2)
        try:
            data = np.array([np.genfromtxt(f"data/{sampler}_{model}_{scale}.csv", delimiter=",") for scale in scales])
        except OSError:
            print(f"Error: at least one missing file of form {sampler}_{model}_*.csv")
            sys.exit(1)
        except ValueError:
            print(f"Error: encountered unexpected data in file of form {sampler}_{model}_*.csv (expected 10 rows and 2 columns of floats)")
            sys.exit(1)

        # top row (ESS)
        ess = ESS_MULTIPLIER * data[:, :, 0]
        meanEss = ess.mean(axis=1)
        errEss = T_VALUE * ess.std(axis=1, ddof=1) / RUNS_PER_SETTING ** 0.5
        axTop.errorbar(scales, meanEss, errEss, color=colour, marker=marker, markersize=5, zorder=zorder)

        # bottom row (ESS/Second)
        essPerSec = data[:, :, 0] / data[:, :, 1]
        meanEssPerSec = essPerSec.mean(axis=1)
        errEssPerSec = T_VALUE * essPerSec.std(axis=1, ddof=1) / RUNS_PER_SETTING ** 0.5
        axBottom.errorbar(scales, meanEssPerSec, errEssPerSec, color=colour, marker=marker, markersize=5, zorder=zorder)

    # plot formatting
    for ax in (axTop, axBottom):
        ax.set_yscale("log")
        if model == "poisson":
            ax.set_xscale("log")
            ax.grid(linestyle="--", which="major")
        else:
            ax.grid(linestyle="--", which="both")
            ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
            ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%d'))
            ax.yaxis.set_minor_formatter(tck.FormatStrFormatter('%d'))

    axBottom.set_xlabel(xLabel)

# draw plots
scalesPoisson = [0.1, 0.125893, 0.158489, 0.199526, 0.251189, 0.316228, 0.398107,
                 0.501187, 0.630957, 0.794328, 1.0, 1.25893, 1.58489, 1.99526,
                 2.51189, 3.16228, 3.98107, 5.01187, 6.30957, 7.94328, 10.0]
scalesIsingNeural = np.linspace(0, 2.5, 21)

loadAndPlot("poisson", scalesPoisson, axes[0, 0], axes[1, 0], r"Rate \(\Lambda\)")
loadAndPlot("ising", scalesIsingNeural, axes[0, 1], axes[1, 1], r"Inverse temperature \(\beta\)")
loadAndPlot("neural", scalesIsingNeural, axes[0, 2], axes[1, 2], r"Weight strength \(\alpha\)")

# finalising
axes[0, 0].set_ylabel(r"\textsc{ess}")
axes[1, 0].set_ylabel(r"\textsc{ess/second}")
legendLabels = [r"Point process",
                r"Birth-death process",
                r"Zanella process - \(z^{1/2}\)",
                r"Zanella process - \(\min\left(1, z\right)\)",
                r"Zanella process - \(z/\left(1 + z\right)\)"]
fig.legend(legendLabels, handler_map={ErrorbarContainer: HandlerErrorbar(yerr_size=0)}, markerscale=2, bbox_to_anchor=(0, -0.08, 1, 1), loc="lower center", ncol=3)

fig.savefig("plots.eps", format="eps", bbox_inches="tight", transparent=True)