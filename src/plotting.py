# imports
import numpy as np
import seaborn as sns
import os, sys, pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patheffects as PathEffects

from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection

# this file's directory
file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_dir)

# custom import
from theory import depth
from viz import get_colours
from utils import load_experiment
from theory import critical_point, fixed_point, c_map, c_map_slope, depth_scale

def set_plot_params():
    sns.set_context("paper", rc={
    # sns.set_context("paper", font_scale=2, rc={
        "image.cmap": "viridis",
        "lines.linewidth": 2,
        "lines.markersize": 4,
    })
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Computer Modern Roman']
    rcParams['text.usetex'] = True
    rcParams['pdf.fonttype'] = 42

def plot_tanh():
    # Dictionary for data that needs to be extracted
    tests = [{
            "distributions": [{"dist": "none"}],
            "activations": ["tanh"],
            "inits": ["xavier"]
        }, {
            "distributions": [{"dist": "add gauss", "std": 1}],
            "activations": ["tanh"],
            "inits": ["xavier"]
        }, {
            "distributions": [{"dist": "none"}],
            "activations": ["relu"],
            "inits": ["he"]
        }, {
            "distributions": [{"dist": "add gauss", "std": 1}],
            "activations": ["relu"],
            "inits": ["he"]
        }]

    ############################################################################
    # q - length / variance plots
    ############################################################################
    nq = 30
    qmax = 15
    qrange = np.linspace(0, qmax, nq)
    widxs = [0]
    bidxs = [0]
    n_hidden_layers = 16

    n_tests = len(tests)
    pal = get_colours(6, 7)[2:]
    test_data = []
    for i, test in enumerate(tests):
        test_data.append(load_experiment(test, ["q_maps", "single_layer_qmap_sim", "multi_layer_qmap_sim"], results_dir))

    fig = plt.figure(figsize=(8, 3))

    gs = plt.GridSpec(1, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])

    # Add unity line
    ax1.plot((0, qmax), (0, qmax), '--', color='k', zorder=900, dashes=(12, 24), label="Identity line")
    ax1.set_xlim(0, qmax)
    ax1.set_ylim(0, qmax)
    ax1.set_xlabel(r'Input variance ($\nu^{l-1})$')
    ax1.set_ylabel(r'Output variance ($\nu^l$)')
    ax1.set_title("Iterative variance map")

    # axis 2
    ax2.set_xlim(0, qmax-1)
    ax2.set_ylim(0, qmax)
    ax2.set_xlabel(r'Layer ($l$)')
    ax2.set_ylabel(r'Variance ($\nu^{l})$')
    ax2.set_title(r"Dynamics of $\nu$")

    nn = len(test_data)
    col_i = 0
    shade_i = 5
    for test, attr in zip(test_data, tests):
        for dist in attr["distributions"]:
            for act in attr["activations"]:

                if dist['dist'] == "none":
                    if act == "tanh":
                        label = "Tanh - none"
                        col_i = 0
                    else:
                        label = "ReLU - none"
                        col_i = 2

                elif "gauss" in dist['dist']:
                    if act == "tanh":
                        label = "Tanh - add Gauss $(\sigma^2_\epsilon = $ " + str(dist['std']) + ")"
                        col_i = 1
                    else:
                        label = "ReLU - add Gauss $(\sigma^2_\epsilon = $ " + str(dist['std']) + ")"
                        col_i = 3

                for init in attr["inits"]:
                    dashes = (None, None)


                    # extract test data
                    qmaps = test[dist['dist']][act][init]['q_maps']['qmaps']
                    single_layer_sims = test[dist['dist']][act][init]['single_layer_qmap_sim']
                    multi_layer_sims = test[dist['dist']][act][init]['multi_layer_qmap_sim']['qmaps_sim']
                    multi_layer_theory = test[dist['dist']][act][init]['multi_layer_qmap_sim']['theory']


                    ############################################################################
                    # left
                    ############################################################################
                    for w, b in zip(widxs, bidxs):

                        # plot means of simulation (as dots)
                        mu = single_layer_sims[w, b].mean(-1).mean(-2)
                        ax1.plot(qrange, mu, w, b, marker='o', ls='none', alpha=0.9, zorder=0, c=pal[col_i][shade_i])

                        # add confidence interval around simulation
                        std = single_layer_sims[w, b].mean(-1).std(-2)
                        ax1.fill_between(qrange, mu-std, mu+std, alpha=0.4, label='_nolegend_', color=pal[col_i][3])

                        # theory line
                        ax1.plot(qrange, qmaps[0, 0, :, 1], c=pal[col_i][shade_i], label=label, dashes=dashes)


                    ############################################################################
                    # right
                    ############################################################################
                    q = 1
                    xx = np.arange(multi_layer_sims.shape[-2])
                    for w, b in zip(widxs, bidxs):
                        # confidence intervals
                        mu = multi_layer_sims[w, b].mean(axis=-1).mean(axis=0)
                        std = multi_layer_sims[w, b].mean(axis=-1).std(axis=0)

                        # plot theory
                        ax2.plot(multi_layer_theory, c=pal[col_i][shade_i], label="Theory")

                        # plot the simulation
                        ax2.fill_between(xx, mu-std, mu+std, alpha=0.2, label='_nolegend_', color=pal[col_i][3])

                        # dots for mean
                        ax2.plot(xx, mu, 'o', alpha=0.9, color=pal[col_i][shade_i], label="Simulation")

    ##############
    # add labels #
    ##############
    fig.text(0.02, 0.95, "(a)")
    fig.text(0.52, 0.95, "(b)")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5,-0.15)
    )

    plt.gcf().tight_layout()
    plt.savefig(os.path.join(figures_dir, "tanh.pdf"), bbox_inches='tight')

def shared_legend(fig, ncol=3):
    handles_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    handles, labels = [sum(lol, []) for lol in zip(*handles_labels)]

    # remove clutter from legend elements:
    #   * gather lines by label
    #   * keep the darkest line within each group
    #       (the darkest line is the one with the smallest sum of its elements)
    dup_labels = {}
    for index, label in enumerate(labels):
        if label in dup_labels:
            dup_labels[label].append(index)
        else:
            dup_labels[label] = [index,]

    final_labels = []
    final_handles = []
    for label, indices in dup_labels.items():
        min_index = indices[0]

        if len(indices) > 1:
            min_shade = np.inf

            for index in indices:
                shade = np.sum(handles[index].get_color())

                if shade < min_shade:
                    min_index = index
                    min_shade = shade

        final_labels.append(labels[min_index])
        final_handles.append(handles[min_index])

    fig.legend(
        final_handles, final_labels, loc='lower center', ncol=ncol, bbox_to_anchor=(0.5, -0.1)
    )

def plot_variance():
    # Dictionary for data that needs to be extracted
    tests = [{
        "distributions": [{"dist": "mult gauss", "std": 0.25}],
        "activations": ["relu"],
        "inits": ["underflow"]
    },{
        "distributions": [{"dist": "mult gauss", "std": 0.25}],
        "activations": ["relu"],
        "inits": ["overflow"]
    },{
        "distributions": [{"dist": "mult gauss", "std": 0.25}],
        "activations": ["relu"],
        "inits": ["crit"]
    },{
        "distributions": [{"dist": "bern", "prob_1": 0.6}],
        "activations": ["relu"],
        "inits": ["underflow"]
    },{
        "distributions": [{"dist": "bern", "prob_1": 0.6}],
        "activations": ["relu"],
        "inits": ["overflow"]
    },{
        "distributions": [{"dist": "bern", "prob_1": 0.6}],
        "activations": ["relu"],
        "inits": ["crit"]
    }]

    ############################################################################
    # q - length / variance plots
    ############################################################################
    nq = 30
    qmax = 15
    qrange = np.linspace(0, qmax, nq)
    widxs = [0]
    bidxs = [0]
    n_hidden_layers = 16

    n_tests = len(tests)
    pal = get_colours(10, 7)
    test_data = []
    for i, test in enumerate(tests):
        test_data.append(load_experiment(test, ["q_maps", "single_layer_qmap_sim", "multi_layer_qmap_sim"], results_dir))

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 3))

    # Add unity line
    ax1.plot((0, qmax), (0, qmax), '--', color='k', zorder=900, dashes=(12, 24), label="Identity line")
    ax1.set_xlim(0, qmax)
    ax1.set_ylim(0, qmax)
    ax1.set_xlabel(r'Input variance ($\nu^{l-1})$')
    ax1.set_ylabel(r'Output variance ($\nu^l$)')
    ax1.set_title("Iterative variance map")
    ax1.text(2, 10, r'$\sigma^2_w > \frac{2}{\mu_2}$')
    ax1.text(10, 1, r'$\sigma^2_w < \frac{2}{\mu_2}$')
    ax1.text(11, 10, r'$\sigma^2_w = \frac{2}{\mu_2}$')

    # axis 2
    ax2.set_xlim(0, qmax)
    ax2.set_ylim(0, qmax)
    ax2.set_xlabel(r'Layer ($l$)')
    ax2.set_ylabel(r'Variance ($\nu^{l})$')
    ax2.set_title(r"Dynamics of $\nu$")

    nn = len(test_data)
    col_i = 0

    bern_label = False
    gauss_label = False
    for test, attr in zip(test_data, tests):
        for dist in attr["distributions"]:
            label = ""
            if dist['dist'] == "none":
                col_i = 0
            elif dist['dist'] == "bern":
                col_i = 1
                label = "Dropout"
            elif "gauss" in dist['dist']:
                col_i = 3
                label = "Mult Gauss"

            for act in attr["activations"]:
                for init in attr["inits"]:
                    dashes = (None, None)
                    if "under" in init:
                        shade_i = 2
                    elif "crit" in init:
                        shade_i = 4
                        dashes = (24, 12) if dist['dist'] == "bern" else (None, None)
                    else:
                        shade_i = 6

                    # extract test data
                    qmaps = test[dist['dist']][act][init]['q_maps']['qmaps']
                    single_layer_sims = test[dist['dist']][act][init]['single_layer_qmap_sim']
                    multi_layer_sims = test[dist['dist']][act][init]['multi_layer_qmap_sim']['qmaps_sim']
                    multi_layer_theory = test[dist['dist']][act][init]['multi_layer_qmap_sim']['theory']

                    ############################################################################
                    # left
                    ############################################################################
                    for w, b in zip(widxs, bidxs):

                        # plot means of simulation (as dots)
                        mu = single_layer_sims[w, b].mean(-1).mean(-2)
                        ax1.plot(qrange, mu, w, b, marker='o', ls='none', alpha=0.9, zorder=0, c=pal[col_i][shade_i])

                        # add confidence interval around simulation
                        std = single_layer_sims[w, b].mean(-1).std(-2)
                        ax1.fill_between(qrange, mu-std, mu+std, alpha=0.4, label='_nolegend_', color=pal[col_i][shade_i])

                        # theory line
                        ax1.plot(qrange, qmaps[0, 0, :, 1], c=pal[col_i][shade_i], label=label, dashes=dashes)

                    ############################################################################
                    # right
                    ############################################################################
                    q = 1
                    xx = np.arange(multi_layer_sims.shape[-2])
                    for w, b in zip(widxs, bidxs):
                        # confidence intervals
                        mu = multi_layer_sims[w, b].mean(axis=-1).mean(axis=0)
                        std = multi_layer_sims[w, b].mean(axis=-1).std(axis=0)

                        # plot theory
                        if "Dropout" in label and not bern_label:
                            bern_label = True
                            ax2.plot(multi_layer_theory, c=pal[col_i][shade_i], label=label)
                        elif "Gauss" in label and not gauss_label:
                            gauss_label = True
                            ax2.plot(multi_layer_theory, c=pal[col_i][shade_i], label=label)
                        else:
                            ax2.plot(multi_layer_theory, c=pal[col_i][shade_i])

                        # plot the simulation
                        ax2.fill_between(xx, mu-std, mu+std, alpha=0.2, label='_nolegend_', color=pal[col_i][shade_i])

                        # dots for mean
                        ax2.plot(xx, mu, 'o', alpha=0.9, color=pal[col_i][shade_i])

    shared_legend(fig)

    fig.text(0.02, 0.95, "(a)")
    fig.text(0.52, 0.95, "(b)")

    plt.gcf().tight_layout()
    plt.savefig(os.path.join(figures_dir, "variance.pdf"), bbox_inches='tight')

def plot_variance_edge():
    mu21 = np.linspace(1, 2, 100)
    sigma1 = 2/mu21

    fig, ax3 = plt.subplots(1, 1, figsize=(4, 3))

    ax3.plot(mu21, sigma1, c="purple", label="Variance critical boundary", linestyle='--')
    ax3.fill_between(mu21, 1, sigma1, facecolor='blue', alpha=0.2)
    ax3.fill_between(mu21, 2, sigma1, facecolor='red', alpha=0.2)
    ax3.text(1.5, 1.6, 'Overflow')
    ax3.text(1.55, 1.5, r'($\sigma^2_w > \frac{2}{\mu_2}$)')
    ax3.text(1.1, 1.2, 'Underflow')
    ax3.text(1.15, 1.1, r'($\sigma^2_w < \frac{2}{\mu_2}$)')
    ax3.text(1.2, 1.7, r'$\sigma^2_w = \frac{2}{\mu_2}$')
    ax3.set_xlim(1, 2)
    ax3.set_ylim(1, 2)
    ax3.set_xlabel('Weight initialisation ($\sigma^2_w$)')
    ax3.set_ylabel('Second moment of noise dist. ($\mu_2$)')
    ax3.set_title('Variance propagation dynamics')

    ax3.legend()

    plt.gcf().tight_layout()
    plt.savefig(os.path.join(figures_dir, "vairance_edge_theory.pdf"), bbox_inches='tight')

def plot_correlation():
    tests = [{
        "distributions": [{"dist": "none"}],
        "activations": ["relu"],
        "inits": ["crit"]
    }, {
        "distributions": [{"dist": "bern", "prob_1": 0.6}],
        "activations": ["relu"],
        "inits": ["crit"]
    }, {
        "distributions": [{"dist": "bern", "prob_1": 0.8}],
        "activations": ["relu"],
        "inits": ["crit"]
    }, {
        "distributions": [{"dist": "mult gauss", "std": 0.25}],
        "activations": ["relu"],
        "inits": ["crit"]
    }, {
        "distributions": [{"dist": "mult gauss", "std": 2}],
        "activations": ["relu"],
        "inits": ["crit"]
    }]

    pal = get_colours(10, 7)
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 3))
    test_data = []
    for i, test in enumerate(tests):
        test_data.append(load_experiment(test,
                                        ["multi_layer_cmap_sim", "cmap", "ctrajs", "chi1", "cmap_sim"], results_dir))

    for test, attr in zip(test_data, tests):
        for dist in attr["distributions"]:
            if dist['dist'] == "none":
                col_i = 0
                shade_i = 6
            elif dist['dist'] == "bern":
                col_i = 1

                if float(dist['prob_1']) < 0.7:
                    shade_i = 6
                else:
                    shade_i = 2

            elif "gauss" in dist['dist']:
                col_i = 3

                if float(dist['std']) < 0.5:
                    shade_i = 2
                else:
                    shade_i = 6

            for act in attr["activations"]:
                for init in attr["inits"]:
                    correlations = test[dist['dist']][act][init]['multi_layer_cmap_sim']
                    cmap_data = test[dist['dist']][act][init]['cmap']
                    cmaps = cmap_data["cmaps"]
                    cstars = cmap_data["cstars"]
                    ctrajs = test[dist['dist']][act][init]['ctrajs']
                    cmap_sim_data = test[dist['dist']][act][init]['cmap_sim']
                    cmap_sim_input = cmap_sim_data["input_correlations"]
                    cmap_sim_output = cmap_sim_data["output_correlations"]

                    num_trials = cmap_sim_input.shape[0]
                    num_networks = cmap_sim_output.shape[0] // num_trials

                    crange = np.linspace(0, 1.0, 51)

                    # create label
                    label = ""

                    x = crange[-1] - 0.03
                    y = cmaps[0, 0, -10] - 0.015
                    if "std" in dist:
                        label = "Mult Gauss"

                        ax1.text(
                            x, y, f"($\sigma_\epsilon = {dist['std']}$)",
                            horizontalalignment='right',
                            verticalalignment='top',
                        )
                    elif "prob_1" in dist:
                        label = "Dropout"

                        ax1.text(
                            x, y, f"($p = {dist['prob_1']}$)",
                            horizontalalignment='right',
                            verticalalignment='top',
                        )
                    else:
                        label = "None"

                    ############################################################
                    # left - Correlation map
                    ############################################################
                    # Theory
                    ax1.plot(crange, cmaps[0, 0], c=pal[col_i][shade_i], label=label)

                    # Simulation
                    mu_x = cmap_sim_input.mean(axis=0)
                    mu_y = cmap_sim_output.mean(axis=0).mean(axis=0)
                    std_y = cmap_sim_output.mean(axis=1).std(axis=0)

                    ax1.scatter(mu_x, mu_y, marker="o", alpha=0.9, color=pal[col_i][shade_i])
                    ax1.fill_between(mu_x, mu_y - std_y, mu_y + std_y, alpha=0.2, label='_nolegend_', color=pal[col_i][shade_i])

                    ax1.plot(cstars[0, 0], cstars[0, 0], markersize=15, marker='x', mew=3, c=pal[col_i][shade_i], zorder=999, alpha=1.0, clip_on=False)

                    ############################################################
                    # right - dynamics of convergence
                    ############################################################
                    # Theory
                    for j in range(correlations.shape[0]):
                        ax2.plot(ctrajs[0, 0, j, :].T, c=pal[col_i][shade_i])

                    # Simulations
                    x_axis = np.arange(0, correlations.shape[-1])

                    x = x_axis[-1] - 1
                    y = ctrajs[0, 0, 0, -1] + 0.1
                    if "std" in dist:
                        ax2.text(
                            x, y, f"($\sigma_\epsilon = {dist['std']}$)",
                            horizontalalignment='right',
                            verticalalignment='top',
                        )
                    elif "prob_1" in dist:
                        ax2.text(
                            x, y, f"($p = {dist['prob_1']}$)",
                            horizontalalignment='right',
                            verticalalignment='top',
                        )

                    means = correlations.mean(axis=-2).mean(axis=-2)
                    std = correlations.mean(axis=-2).std(axis=-2)

                    for i in np.arange(means.shape[0]):
                        ax2.scatter(np.arange(means[i].shape[0]), means[i], marker='o', alpha=0.9, color=pal[col_i][shade_i])
                        ax2.fill_between(x_axis, means[i] - std[i], means[i] + std[i], alpha=0.2, label='_nolegend_', color=pal[col_i][shade_i])

    # Add unity line
    ax1.plot((0, 1), (0, 1), '--', color='k', zorder=900, label="Identity line")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title(r'Iterative correlation map')

    ax1.set_xticks([0, 0.5, 1.0])
    ax1.set_yticks([0, 0.5, 1.0])
    ax1.set_xlabel(r'Input correlation ($c^{l-1})$')
    ax1.set_ylabel(r'Output correlation ($c^{l}$)')

    ax2.set_xlabel(r'Layer ($l$)')
    ax2.set_ylabel(r'Correlation ($c^{l})$')
    ax2.set_title(r'Dynamics of $c$')
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, correlations.shape[-1] - 1)

    shared_legend(fig, ncol=4)

    fig.text(0.02, 0.95, "(a)")
    fig.text(0.52, 0.95, "(b)")

    plt.gcf().tight_layout()
    plt.savefig(os.path.join(figures_dir, "correlation.pdf"), bbox_inches='tight')

def plot_correlation_edge_theory():
    rates = np.linspace(0.1, 1, 100)
    mu2s = 1/rates
    fps = []
    fp_slopes = []
    for p in rates:
        mu2 = 1/p
        fpoint = fixed_point(c_map, p, p*2, mu2)
        fps.append(fpoint)

        slope = c_map_slope(fpoint, p*2)
        fp_slopes.append(slope)

    fig, ax3 = plt.subplots(1, 1, figsize=(4, 3))

    ############################################################
    # right - phase diagram
    ############################################################
    ax3.plot(mu2s, fp_slopes, c='purple', label="Critical")
    ax3.set_xlabel(r"Second moment of noise distribution ($\mu_2$)")
    ax3.set_ylabel(r"Slope at fixed point ($\chi (c^*)$)")
    ax3.scatter(1, 1, c="red", marker='*', s=100, label='Edge of chaos')
    ax3.fill_between(mu2s, 0, 1, facecolor='cyan', alpha=0.2)
    ax3.text(3, 0.5, 'Ordered regime \n (vanishing gradients)')
    leg = ax3.legend()

    ax3.set_xlim(0.8, 10)
    ax3.set_ylim(0, 1.05)
    ax3.set_title('Phase diagram')

    plt.gcf().tight_layout()
    plt.savefig(os.path.join(figures_dir, "correlation_edge_theory.pdf"), bbox_inches='tight')

def plot_rate_of_convergence(data):
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 3))

    for ax, (noise_type, results) in zip([ax1, ax2], data.items()):
        for noise_label, colour, theory, imperical in zip(
            results["noise"],
            results["colours"],
            results["convergence_theory"],
            results["convergence_imperical"],
        ):
            ax.plot(theory, color=colour)
            ax.plot(imperical, color=colour, marker="o", linestyle="")

        ax.set_yscale('log')
        ax.set_ylabel(r'$|c^l - c^*|$')
        ax.set_xlabel('Layer ($l$)')
        ax.set_title(noise_type)

    fig.suptitle("Rate of convergence to fixed point")
    # fig.legend(...)

    ##############
    # add labels #
    ##############
    fig.text(0.02, 0.85, "(a)")
    fig.text(0.52, 0.85, "(b)")

    plt.gcf().tight_layout()
    plt.savefig(os.path.join(figures_dir, "correlation_rate_of_convergence.pdf"), bbox_inches='tight')

def plot_theoretical_vs_measured_depth_scale(data):
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 3))

    for ax, (noise_type, results) in zip([ax1, ax2], data.items()):
        # plot theory
        ax.plot(results["mu"], results["xi_theory"], color=results["colour"])

        # plot empirical results
        ax.plot(results["mu"], results["xi_imperical"], color=results["colour"], marker="o", linestyle="")

        ax.set_ylabel(r'$\xi_c$')
        ax.set_xlabel(r'$\mu_2$')
        ax.set_title(noise_type)

    fig.suptitle("Depth scales")
    # fig.legend(...)

    ##############
    # add labels #
    ##############
    fig.text(0.02, 0.85, "(a)")
    fig.text(0.52, 0.85, "(b)")

    plt.gcf().tight_layout()
    plt.savefig(os.path.join(figures_dir, "correlation_depth_scale.pdf"), bbox_inches='tight')

def load_and_process_correlation_convergence(tests):
    mu2s = []
    noises = []
    c_stars = []
    inferred_xi = []
    depth_colour = None
    convergence_colours = []
    imperical_convergence = []
    theoretical_convergence = []

    pal = pal = get_colours(4, 13)

    test_data = []
    for i, test in enumerate(tests):
        test_data.append(load_experiment(test,
                                        ["multi_layer_cmap_sim", "cmap", "ctrajs", "chi1", "cmap_sim"], results_dir))

    for i, (test, attr) in enumerate(zip(test_data, tests)):
        for dist in attr["distributions"]:
            if dist['dist'] == "none":
                raise ValueError("There should not be tests with no noise in this section")

            elif dist['dist'] == "bern":
                col_i = 1
                depth_colour = pal[col_i, 12]
                shade_i = int((10 - i)/10 * 12)
                mu2s.append(1/float(dist['prob_1']))
                noises.append(f"($p = {dist['prob_1']}$)")
                convergence_colours.append(pal[col_i, shade_i])

            elif "gauss" in dist['dist']:
                col_i = 3
                shade_i = i
                depth_colour = pal[col_i, 12]
                mu2s.append(float(dist['std'])**2 + 1)
                noises.append(f"($\sigma_\epsilon = {dist['std']}$)")
                convergence_colours.append(pal[col_i, shade_i])

            for act in attr["activations"]:
                for init in attr["inits"]:

                    correlations = test[dist['dist']][act][init]['multi_layer_cmap_sim']
                    ctrajs = test[dist['dist']][act][init]['ctrajs']

                    c_stars.append(correlations.mean(axis=(0, 1, 2))[-1])
                    rate_trajectories = np.abs(ctrajs - ctrajs[:, :, :, -1, np.newaxis]).mean(axis=(0, 1))

                    xi_array = []
                    rates_array = []
                    theory_array = []

                    for rates in rate_trajectories:
                        # mask zero values so that they don't break the plot
                        non_zero_rates = rates != 0

                        # get a axis that corresponds to the non-zero elements
                        x = np.arange(rates.shape[0])
                        z, b = np.polyfit(x[non_zero_rates], np.log(rates[non_zero_rates]), 1)
                        theory = np.exp(x*z + b)

                        xi_array.append(-1/z)
                        rates_array.append(rates)
                        theory_array.append(theory)

                    inferred_xi.append(np.mean(xi_array))
                    imperical_convergence.append(np.mean(rates_array, axis=0))
                    theoretical_convergence.append(np.mean(theory_array, axis=0))

    mu2s = np.array(mu2s)
    c_stars = np.array(c_stars)
    inferred_xi = np.array(inferred_xi)
    imperical_convergence = np.array(imperical_convergence)
    theoretical_convergence = np.array(theoretical_convergence)

    # mask zero values in imperical convergence because it ruins the plot
    zeros = imperical_convergence == 0
    imperical_convergence = np.ma.array(imperical_convergence, mask=zeros)
    theoretical_convergence = np.ma.array(theoretical_convergence, mask=zeros)

    # compare theory depth scales with inferred ones from simulations
    xi_c = -1/np.log(np.arcsin(c_stars)/(np.pi*mu2s) + 1/(2*mu2s))

    index_order = np.argsort(mu2s)

    mu2s_line = mu2s[index_order]
    xi_c_line = xi_c[index_order]
    inf_xi_line = inferred_xi[index_order]

    return {
        "convergence": {
            "noise": noises,
            "colours": convergence_colours,
            "convergence_theory": theoretical_convergence,
            "convergence_imperical": imperical_convergence
        },
        "depth_scale": {
            "mu": mu2s_line,
            "colour": depth_colour,
            "xi_theory": xi_c_line,
            "xi_imperical": inf_xi_line,
        }
    }

def plot_fixed_point_convergence():
    dropout_tests = [{
        "distributions": [{"dist": "bern", "prob_1": 0.1}],
        "activations": ["relu"],
        "inits": ["crit"]
    }, {
        "distributions": [{"dist": "bern", "prob_1": 0.2}],
        "activations": ["relu"],
        "inits": ["crit"]
    }, {
        "distributions": [{"dist": "bern", "prob_1": 0.3}],
        "activations": ["relu"],
        "inits": ["crit"]
    }, {
        "distributions": [{"dist": "bern", "prob_1": 0.4}],
        "activations": ["relu"],
        "inits": ["crit"]
    }, {
        "distributions": [{"dist": "bern", "prob_1": 0.5}],
        "activations": ["relu"],
        "inits": ["crit"]
    }, {
        "distributions": [{"dist": "bern", "prob_1": 0.6}],
        "activations": ["relu"],
        "inits": ["crit"]
    }, {
        "distributions": [{"dist": "bern", "prob_1": 0.7}],
        "activations": ["relu"],
        "inits": ["crit"]
    }, {
        "distributions": [{"dist": "bern", "prob_1": 0.8}],
        "activations": ["relu"],
        "inits": ["crit"]
    }, {
        "distributions": [{"dist": "bern", "prob_1": 0.9}],
        "activations": ["relu"],
        "inits": ["crit"]
    }]

    gauss_tests = [{
        "distributions": [{"dist": "mult gauss", "std": 0.1}],
        "activations": ["relu"],
        "inits": ["crit"]
    }, {
        "distributions": [{"dist": "mult gauss", "std": 0.25}],
        "activations": ["relu"],
        "inits": ["crit"]
    }, {
        "distributions": [{"dist": "mult gauss", "std": 0.4}],
        "activations": ["relu"],
        "inits": ["crit"]
    }, {
        "distributions": [{"dist": "mult gauss", "std": 0.55}],
        "activations": ["relu"],
        "inits": ["crit"]
    }, {
        "distributions": [{"dist": "mult gauss", "std": 0.7}],
        "activations": ["relu"],
        "inits": ["crit"]
    }, {
        "distributions": [{"dist": "mult gauss", "std": 0.85}],
        "activations": ["relu"],
        "inits": ["crit"]
    }, {
        "distributions": [{"dist": "mult gauss", "std": 1.0}],
        "activations": ["relu"],
        "inits": ["crit"]
    }, {
        "distributions": [{"dist": "mult gauss", "std": 1.15}],
        "activations": ["relu"],
        "inits": ["crit"]
    }, {
        "distributions": [{"dist": "mult gauss", "std": 1.3}],
        "activations": ["relu"],
        "inits": ["crit"]
    }, {
        "distributions": [{"dist": "mult gauss", "std": 1.45}],
        "activations": ["relu"],
        "inits": ["crit"]
    }, {
        "distributions": [{"dist": "mult gauss", "std": 1.6}],
        "activations": ["relu"],
        "inits": ["crit"]
    }, {
        "distributions": [{"dist": "mult gauss", "std": 1.75}],
        "activations": ["relu"],
        "inits": ["crit"]
    }, {
        "distributions": [{"dist": "mult gauss", "std": 1.9}],
        "activations": ["relu"],
        "inits": ["crit"]
    }]

    # process the data and pass it to the plotting functions
    dropout_data = load_and_process_correlation_convergence(dropout_tests)
    gauss_data = load_and_process_correlation_convergence(gauss_tests)

    convergence_data = {
        "Dropout": dropout_data["convergence"],
        "Gaussian": gauss_data["convergence"]
    }

    depth_scale_data = {
        "Dropout": dropout_data["depth_scale"],
        "Gaussian": gauss_data["depth_scale"]
    }

    plot_rate_of_convergence(convergence_data)
    plot_theoretical_vs_measured_depth_scale(depth_scale_data)

def plot_variance_depth_scale(data, shading="gouraud"):
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 3))

    for ax, (dataset, results) in zip([ax1, ax2], data.items()):
        init_axis = results["imperical"]["init_axis"]
        depth_axis = results["imperical"]["depth_axis"]
        variance = results["imperical"]["variance"]
        init_axis_theory = results["theory"]["init_axis"]
        max_depth = results["theory"]["max_depth"]
        critical = results["theory"]["critical"]
        xavier = results["theory"]["xavier_init"]
        xavier_depth = results["theory"]["xavier_depth"]
        he = results["theory"]["he_init"]
        he_depth = results["theory"]["he_depth"]
        num_layers = results["imperical"]["num_layers"]

        cmap = mpl.cm.get_cmap(name="Spectral_r")
        cmap.set_bad('black')

        pcm = ax.pcolormesh(init_axis, depth_axis, variance.T, cmap=cmap, shading=shading, linewidth=0)
        ax.plot(init_axis_theory, max_depth, label="Theoretical maximum depth", c='cyan')
        ax.plot([critical,]*2, [0, num_layers], color="black", linestyle="--", label="Criticality", dashes=(2, 2))
        ax.plot([xavier,]*2, [0, xavier_depth], color="orange", linestyle="--", label="Xavier", dashes=(2, 2))
        ax.plot([he,]*2, [0, he_depth], color="dodgerblue", linestyle="--", label="He", dashes=(2, 2))

        cbar = fig.colorbar(pcm, ax=ax, extend='max')
        cbar.ax.set_title(r'$log(\nu^l)$')

        ax.set_xlabel('Weight initialisation ($\sigma^2_w$)')
        ax.set_ylabel("Number of layers ($l$)")
        ax.set_title(dataset.upper())

        ax.set_ylim(0, num_layers)
        ax.set_xlim(np.min(init_axis), np.max(init_axis))
        ax.text(0.2, 400, 'Underflow', color="white")
        ax.text(1.7, 400, 'Overflow', color="white")

    fig.suptitle("Variance propagation depth for dropout with $p$ = 0.6, critical initialisation at $\sigma^2_w = 1.2$")
    fig.legend(*ax1.get_legend_handles_labels(), loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1))

    ##############
    # add labels #
    ##############
    fig.text(0.02, 0.85, "(a)")
    fig.text(0.52, 0.85, "(b)")

    plt.gcf().tight_layout()
    plt.savefig(os.path.join(figures_dir, "variance_depth.pdf"), bbox_inches='tight')

def plot_imperical_depth(fig, ax, dataset, dropout_rates, loss, depth_axis, init_axis, depth_scale_theory, type_loss, shading="gouraud"):
    lc = LineCollection(depth_scale_theory, cmap=plt.get_cmap('Greens'))
    lc.set_array(dropout_rates)
    ax.add_collection(lc)

    cbar_p = fig.colorbar(lc, ax=ax)
    cbar_p.ax.set_title('$p$')
    ax.set_xlabel("Critical initialisation for $p$ ($\sigma^2_w$)")
    ax.set_ylabel("Number of layers ($l$)")
    txt = ax.text(1.5, 15, r'$6\xi_c$', fontsize=15, color="black", weight="bold", horizontalalignment="center", verticalalignment="center")
    txt = ax.text(1.5, 15, r'$6\xi_c$', fontsize=15, color="white", horizontalalignment="center", verticalalignment="center")
    txt.set_path_effects([PathEffects.withStroke(foreground='black')])
    ax.set_title(dataset.upper())
    ax.set_xlim(0.3, 2.1)
    ax.set_ylim(2, 40)

    pcm = ax.pcolormesh(init_axis+0.1, depth_axis+0.1, loss, cmap='Spectral_r', shading=shading, linewidth=0)
    cbar_loss = fig.colorbar(pcm, ax=ax, extend='max')
    cbar_loss.ax.set_title(type_loss)

def plot_trainable_depth(data):
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 3))

    for ax, (dataset, results) in zip([ax1, ax2], data.items()):
        dropout_rates = results["imperical"]["training_dropout_rates"]
        loss = results["imperical"]["train_loss"]
        depth_axis = results["imperical"]["train_depth_axis"]
        init_axis = results["imperical"]["train_init_axis"]
        depth_scale_theory = results["theory"]["correlation_depth_scale"]

        plot_imperical_depth(fig, ax, dataset, dropout_rates, loss, depth_axis, init_axis, depth_scale_theory, "Train loss")

    fig.suptitle("Trainable depth")

    ##############
    # add labels #
    ##############
    fig.text(0.02, 0.85, "(a)")
    fig.text(0.52, 0.85, "(b)")

    plt.gcf().tight_layout()
    plt.savefig(os.path.join(figures_dir, "trainable_depth.pdf"), bbox_inches='tight')

def plot_generalisation_depth(data, shading="gouraud"):
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 3))

    for ax, (dataset, results) in zip([ax1, ax2], data.items()):
        dropout_rates = results["imperical"]["training_dropout_rates"]
        loss = results["imperical"]["test_loss"]
        depth_axis = results["imperical"]["train_depth_axis"]
        init_axis = results["imperical"]["train_init_axis"]
        depth_scale_theory = results["theory"]["correlation_depth_scale"]

        plot_imperical_depth(fig, ax, dataset, dropout_rates, loss, depth_axis, init_axis, depth_scale_theory, "Test loss")

    fig.suptitle("Generalisation depth")

    ##############
    # add labels #
    ##############
    fig.text(0.02, 0.85, "(a)")
    fig.text(0.52, 0.85, "(b)")

    plt.gcf().tight_layout()
    plt.savefig(os.path.join(figures_dir, "generalisation_depth.pdf"), bbox_inches='tight')

def load_depth_scale():
    p = 0.6
    data = {}

    # variance depth plot theory
    init_theory = np.linspace(0, 2.5, 1000)
    depth_per_p_theory = depth("Dropout", init_theory, p)
    crit_point = critical_point("Dropout", p)

    # calculate what depth Xavier and He inits would terminate at and put that on the figure
    xavier_init = 1
    xavier_index = np.argmin(np.abs(init_theory - xavier_init))
    xavier_depth = depth_per_p_theory[xavier_index]

    he_init = 2
    he_index = np.argmin(np.abs(init_theory - he_init))
    he_depth = depth_per_p_theory[he_index]

    # trainable depth plot theory
    train_depth_axis = np.linspace(2, 40, 10, dtype=int)
    critical_inits = 2 * np.linspace(0.1, 1, 10)
    train_init_matrix, train_depth_matrix = np.meshgrid(critical_inits, train_depth_axis, indexing='ij')

    dropout_rates = np.linspace(0.1, 1, 100)
    theory_critical_inits = 2 * dropout_rates

    fp_slopes = []
    for p in dropout_rates:
        mu2 = 1/p
        fpoint = fixed_point(c_map, p, p*2, mu2)
        slope = c_map_slope(fpoint, p*2)
        fp_slopes.append(slope)

    xi_c = 6 * depth_scale(fp_slopes)
    points = np.array([theory_critical_inits, xi_c]).T.reshape(-1, 1, 2)
    critical_init_vs_xi = np.concatenate([points[:-1], points[1:]], axis=1)

    for dataset in ["mnist", "cifar-10"]:
        dataset_dir_name = dataset.replace("-", "")
        variance_path = os.path.join(results_dir, "variance_depth", dataset_dir_name)
        trainable_path = os.path.join(results_dir, "trainable_depth", dataset_dir_name)

        inits = np.load(os.path.join(variance_path, "variance_depth_sigma.npy"))
        variance_per_layer = np.load(os.path.join(variance_path, "variance_depth.npy"))
        training_data = np.load(os.path.join(trainable_path, "trainable_depth.npy"))

        # # shape: (num_dropout_rates, num_depths, epoch, train / test loss)
        final_train_loss = training_data[:, :, -1, 0]
        final_test_loss = training_data[:, :, -1, 1]

        num_layers = variance_per_layer.shape[-1]
        depth_axis = np.linspace(1, num_layers, num_layers)
        depth_matrix, init_matrix = np.meshgrid(depth_axis, inits, indexing='ij')

        log_variance = np.log(variance_per_layer)

        bad_indices = np.isnan(log_variance) + np.isinf(log_variance)
        # bad_indices = np.isnan(Z1) or np.isinf(Z1)
        log_variance = np.ma.array(log_variance, mask=bad_indices)

        data[dataset] = {
            "imperical": {
                # "inits": inits,
                "variance": log_variance,
                "depth_axis": depth_matrix,
                "init_axis": init_matrix,
                "num_layers": num_layers,
                "training_dropout_rates": dropout_rates,
                "train_depth_axis": train_depth_matrix,
                "train_init_axis": train_init_matrix,
                "train_loss": final_train_loss,
                "test_loss": final_test_loss
            },
            "theory": {
                "init_axis": init_theory,
                "critical": crit_point,
                "max_depth": depth_per_p_theory,
                "p": p,
                "correlation_depth_scale": critical_init_vs_xi,
                "xavier_init": xavier_init,
                "xavier_depth": xavier_depth,
                "he_init": he_init,
                "he_depth": he_depth,
            }
        }

    return data

def plot_depth_scales():
    # load data
    data = load_depth_scale()

    plot_variance_depth_scale(data)
    plot_trainable_depth(data)
    plot_generalisation_depth(data)

if __name__ == "__main__":
    # plot settings
    set_plot_params()

    # results directory
    results_dir = os.path.join(file_dir, "../results")

    # figures directory
    figures_dir = os.path.join(file_dir, "../figures")
    os.makedirs(figures_dir, exist_ok=True)

    plot_tanh()
    plot_variance()
    plot_variance_edge()
    plot_correlation()
    plot_correlation_edge_theory()
    plot_fixed_point_convergence()
    plot_depth_scales()
