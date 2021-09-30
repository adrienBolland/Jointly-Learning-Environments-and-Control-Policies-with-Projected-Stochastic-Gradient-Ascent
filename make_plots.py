import argparse
import os
from copy import deepcopy
from itertools import zip_longest

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import uniform_filter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.style.use("classic")

import utils

from matplotlib import scale as mscale

mscale.register_scale(utils.CustomScale)

# Rendering options for the plots
PLOT_CONTEXT = {
    "font.family": "serif",
    "font.serif": "Computer Modern Sans",
    "text.usetex": True,
    "font.size": 28,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": "small",
    "legend.loc": "lower right",
    "axes.formatter.useoffset": False
}

# Plots that can be made from tensorboard logs for each systems
# NOTE: "tb_fields" is a dict "tensorboard scalar" -> "legend"
# NOTE: first-level key is the systems name
TB_PLOTS = {
    "MSDSystem": {
            "expected_return": {"xlabel": "Iteration $k$",
                                "ylabel": "Expected return $V(\\psi_k, \\theta_k)$",
                                "tb_fields": {"performance/agent": "expected return"}
                                },
            "return": {"xlabel": "Iteration $k$",
                       "ylabel": "return",
                       "tb_fields": {"control-performance/return": "Performance controller"}
                       },
            "parameter_omega":
                {"xlabel": "Iteration $k$",
                 "ylabel": "Parameter $\\omega$",
                 "tb_fields": {"systems-params/omega": "$\\omega$"}
                 },
            "parameter_zeta":
                {"xlabel": "Iteration $k$",
                 "ylabel": "Parameter $\\zeta$",
                 "tb_fields": {"systems-params/zeta": "$\\zeta$"}
                 },
            "parameter_phi_0":
                {"xlabel": "Iteration $k$",
                 "ylabel": "Parameters $\\phi_{0}$",
                 "tb_fields": {"systems-params/phi-0": "$\\phi_{0}$"}
                 },
            "parameter_phi_1":
                {"xlabel": "Iteration $k$",
                 "ylabel": "Parameters $\\phi_{1}$",
                 "tb_fields": {"systems-params/phi-1": "$\\phi_{1}$"}
                 },
            "parameter_phi_2":
                {"xlabel": "Iteration $k$",
                 "ylabel": "Parameters $\\phi_{2}$",
                 "tb_fields": {"systems-params/phi-2": "$\\phi_{2}$"}
                 }
        },
    "MGSystem": {
        "expected_return": {"xlabel": "Iteration $k$ (in thousands)",
                             "ylabel": "Expected return $V(\\psi_k, \\theta_k)$",
                             "tb_fields": {"performance/agent": "expected return"},
                             "yscale": {"value": "custom",
                                        "scale_factor": 2.0,
                                        "locations": [-400, -300, -200, -100, 0, 25, 50, 75, 100],
                                        "max_limit": 110},
                             "y_scale_factor": 1./1000.
                            },
        "return": {"xlabel": "Iteration $k$",
                   "ylabel": "return",
                   "tb_fields": {"control-performance/return": "Performance controller"}
                   },
        "parameter_battery":
            {"xlabel": "Iteration $k$",
             "ylabel": "Parameter $\\bar{P}^B$",
             "tb_fields": {"systems-params/bat_size": "$\\bar{P}^B$"},
             "clipping_vals": {"a_min": 20. / 100., "a_max": 200. / 100.}
             },
        "parameter_genco":
            {"xlabel": "Iteration $k$",
             "ylabel": "Parameter $\\bar{P}^G$",
             "tb_fields": {"systems-params/gen_size": "$\\bar{P}^G$"},
             "clipping_vals": {"a_min": 20. / 100., "a_max": 200. / 100.}
             },
        "parameter_pv":
            {"xlabel": "Iteration $k$",
             "ylabel": "Parameter $\\bar{P}^{PV}$",
             "tb_fields": {"systems-params/pv_size": "$\\bar{P}^{PV}$"},
             "clipping_vals": {"a_min": 1.6 / 8., "a_max": 16. / 8.}
             }
        },
    "Drone": {
        "expected_return": {"xlabel": "Iteration $k$ (in ten thousands)",
                            "ylabel": "Expected return $V(\\psi_k, \\theta_k)$",
                            "tb_fields": {"performance/agent": "expected return"},
                            "yscale": {"value": "custom",
                                       "scale_factor": 2.0,
                                       "locations": [-200, -150, -100, -50, 0, 10, 20, 30],
                                       "min_limit": -250,
                                       "max_limit": 35},
                            "y_scale_factor": 1. / 10000.
                            },
        "return": {"xlabel": "Iteration $k$",
                   "ylabel": "return",
                   "tb_fields": {"control-performance/return": "Performance controller"}
                   },
        "parameter_psi_0":
            {"xlabel": "Iteration $k$",
             "ylabel": "Parameters $\\psi_{0}$",
             "tb_fields": {"systems-params/arm": "$\\psi_{0}$"}
             },
        "parameter_psi_1":
            {"xlabel": "Iteration $k$",
             "ylabel": "Parameters $\\psi_{1}$",
             "tb_fields": {"systems-params/radius": "$\\psi_{1}$"}
             },
        "parameter_psi_2":
            {"xlabel": "Iteration $k$",
             "ylabel": "Parameters $\\psi_{2}$",
             "tb_fields": {"systems-params/thickness": "$\\psi_{2}$"}
             },
        "parameter_psi_3":
            {"xlabel": "Iteration $k$",
             "ylabel": "Parameters $\\psi_{3}$",
             "tb_fields": {"systems-params/width": "$\\psi_{3}$"}
             }
        }
    }

TB_PLOTS.update({"ComplexMGSystem": deepcopy(TB_PLOTS["MGSystem"])})
TB_PLOTS.update({"DroneTrajectory": deepcopy(TB_PLOTS["Drone"])})


def plot_tb_logs(plots, event_accs_list, plots_path, plot_context=None):
    if plot_context is None:
        plot_context = PLOT_CONTEXT

    with mpl.rc_context(plot_context):
        for pname, pdict in plots.items():

            # get the fields in the event file
            f = plt.figure()
            plt.xlabel(pdict["xlabel"])
            plt.ylabel(pdict["ylabel"])
            plt.yscale(**pdict.get("yscale", {"value": "linear"}))

            labels = []

            for field, field_name in pdict["tb_fields"].items():

                ymin, ymax = float("inf"), -float("inf")

                for event_accs_name, event_accs in event_accs_list.items():

                    steps, vals = [], []

                    for event_acc in event_accs:
                        _, step, val = zip(*event_acc.Scalars(field))
                        steps.append(step), vals.append(list(val))  # NOTE: steps should be the same

                    lens = []
                    for i in vals:
                        lens.append(len(i))
                    min_len = min(lens)
                    steps = [y[:min_len] for y in steps]
                    vals = [y[:min_len] for y in vals]

                    if len(steps) == 1 and len(vals) == 1:  # Not mean / std
                        plt.plot(steps[0], vals[0])
                        ymin, ymax = min(ymin, min(vals[0])), max(ymax, max(vals[0]))
                    else:

                        if pdict.get("clipping_vals", None) is not None:
                            vals_ = np.clip(np.array(vals), **pdict["clipping_vals"])
                        else:
                            vals_ = np.array(vals)

                        mean_vals = vals_.mean(axis=0)
                        std_vals_plus = vals_.std(axis=0, where=(vals >= mean_vals))
                        std_vals_min = vals_.std(axis=0, where=(vals < mean_vals))
                        l = plt.plot(steps[0], mean_vals)
                        color = l[-1].get_color()
                        y0, y1 = mean_vals - std_vals_min, mean_vals + std_vals_plus
                        plt.fill_between(steps[0], y0, y1, alpha=.2, color=color)
                        ymin, ymax = min(ymin, min(y0)), max(ymax, max(y1))

                        print(field, mean_vals[-1])

                    plt.ylim(ymin - .05 * abs(ymax - ymin), ymax + .05 * abs(ymax - ymin))
                    # plt.ylim(0., 2.)

                    # add label
                    if len(pdict["tb_fields"]) > 1 and len(event_accs_list) > 1:
                        labels.append(f"{field_name}/{event_accs_name}")
                    elif len(pdict["tb_fields"]) > 1:
                        labels.append(field_name)
                    elif len(event_accs_list) > 1:
                        labels.append(event_accs_name)

                if len(labels) > 1:  # legend only if multiple lines
                    plt.legend(labels)

            # rescale the x axis if required
            y_scale_factor = pdict.get("y_scale_factor", 1.0)
            tcks = np.array(plt.xticks()[0][1:-1])
            plt.xticks(tcks, tcks*y_scale_factor)
            # plt.xlim(0, 100000)

            plt.tight_layout()
            f.savefig(f"{plots_path}/{pname}.pdf")
            plt.close(f)


def plot_grids(grids_dict, parameters, path, plot_context=None):
    assert "xy" in grids_dict and "z" in grids_dict
    assert len(parameters) == 2

    if plot_context is None:
        plot_context = PLOT_CONTEXT

    # retrieve desired parameters from results
    xname = parameters[0]
    x = grids_dict["xy"][xname]["values"]
    xlabel = grids_dict["xy"][xname]["name"]
    xidx = list(grids_dict["xy"].keys()).index(xname)  # dict are ordered as of python 3.7

    yname = parameters[1]
    y = grids_dict["xy"][yname]["values"]
    ylabel = grids_dict["xy"][yname]["name"]
    yidx = list(grids_dict["xy"].keys()).index(yname)

    # create meshgrid
    xx, yy = np.meshgrid(x, y)

    with mpl.rc_context(plot_context):
        for zname in grids_dict["z"]:
            # retrieve grid
            zz = np.array(grids_dict["z"][zname])

            # if greater dimension than 2, aggregate over other dimensions
            if len(zz.shape) > 2:
                dims = [i for i in range(len(zz.shape)) if (i != xidx and i != yidx)]
                zz = np.max(zz, dims)

            f = plt.figure()
            plt.contourf(xx*100., yy*100., uniform_filter(zz, size=3, mode="mirror"), 100)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            cbar = plt.colorbar()
            cbar.ax.set_ylabel(zname)
            plt.tight_layout()
            f.savefig(f"{path}/{zname.replace(' ', '-')}.pdf")
            plt.close(f)


def parse_args():
    parser = argparse.ArgumentParser(description="script making plots.")
    parser.add_argument("-m", "--mode", type=str, required=True, choices=("run", "avg_run", "grid"), default="avg_run")
    parser.add_argument("-l", "--logdir", type=str, required=True, nargs="+",
                        help="path to the directory containing the necessary files for plotting, "
                             "('agent': pickled agent and the tensorboard event file., 'grid': json file with results")
    parser.add_argument("-n", "--logname", type=str, required=False, nargs="+",
                        help="names of the directories")
    parser.add_argument("-gp", "--grid_parameters", type=str, nargs="+", default=["omega", "zeta"],
                        help="parameters than should be displayed in the grid plot, other dimensions are aggregated using a max operation.")
    return parser.parse_args()


def main_run(plots_path, args, config):
    # plot tensorboard logs
    event_acc = EventAccumulator(args.logdir[0])
    event_acc.Reload()
    plot_tb_logs(TB_PLOTS[config["system"]["name"]], {event_acc: "simulation"}, plots_path=plots_path)


def main_avg_run(plots_path, args, config):
    event_accs_list = dict()
    for logdir, name in zip_longest(args.logdir, args.logname):
        dirs = utils.list_run_dirs(logdir)
        # get all the tb event accumulators
        event_accs = []
        for dir in dirs:
            event_acc = EventAccumulator(dir)
            event_acc.Reload()
            event_accs.append(event_acc)

        event_accs_list[name] = event_accs

    # plot tensorboard logs
    plot_tb_logs(TB_PLOTS[config["system"]["name"]], event_accs_list=event_accs_list, plots_path=plots_path)


def main_grid(plots_path, args, config):
    results = utils.load_json(os.path.join(args.logdir, "results.json"))
    plot_grids(results, args.grid_parameters, plots_path)


if __name__ == "__main__":
    utils.set_seeds(42)  # necessary for trajectory sampling
    args = parse_args()

    # In every case, create destinatio, directry for plots
    config = utils.load_json(os.path.join(args.logdir[0], "config.json"))
    plots_path = os.path.join(args.logdir[0], "plots")
    os.makedirs(plots_path, exist_ok=True)

    if args.mode == "run":
        main_run(plots_path, args, config)
    elif args.mode == "avg_run":
        main_avg_run(plots_path, args, config)
    elif args.mode == "grid":
        main_grid(plots_path, args, config)
    else:
        raise ValueError("Allowed modes are: ")
