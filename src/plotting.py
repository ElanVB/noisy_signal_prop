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
# from src.numpy_simulation import *
from utils import load_experiment
from theory import critical_point, fixed_point, c_map, c_map_slope, depth_scale

def set_plot_params():
    # sns.set_context("paper", font_scale=2)

    sns.set_context("paper", rc={
    # sns.set_context("paper", font_scale=2, rc={
        # "axes.labelsize": 9,
        # "xtick.labelsize": 9,
        # "ytick.labelsize": 9,
        # "legend.fontsize": 9,
        # "font.size": 9,
        # "font.family": "serif",
        # "font.serif": "Computer Modern Roman",
        # "text.usetex": True,
        # "pdf.fonttype": 42,
        "image.cmap": "viridis",
        "lines.linewidth": 2,
        "lines.markersize": 4,
    })
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Computer Modern Roman']
    rcParams['text.usetex'] = True
    rcParams['pdf.fonttype'] = 42

    # rcParams['axes.labelsize'] = 9
    # rcParams['xtick.labelsize'] = 9
    # rcParams['ytick.labelsize'] = 9
    # rcParams['legend.fontsize'] = 9
    # rcParams['font.size'] = 9
    # # rcParams['font.size'] = 100

    # rcParams['image.cmap'] = 'viridis'

    # rcParams["lines.linewidth"] = 10
    # rcParams["lines.markersize"] = 3

    # sns.set_style({
    #     "legend.frameon": True,
    #     # "axes.facecolor": "white"
    #     "axes.facecolor": (0, 0, 0, 0)
    # })

    # sns.set_context("paper", font_scale=2)

# def plot_iterative_map():
#     pal = get_colours(6, 7)[2:]
#     test_data = []
#     for i, test in enumerate(tests):
#         test_data.append(load_experiment(test, ["q_maps", "single_layer_qmap_sim", "multi_layer_qmap_sim"], results_dir))

    # fig = plt.figure(figsize=(16, 6))

#     # gs = plt.GridSpec(1, 2)
#     # ax1 = plt.subplot(gs[0, 0])
#     # ax2 = plt.subplot(gs[0, 1])

#     # Add unity line
#     plt.plot((0, qmax), (0, qmax), '--', color='k', zorder=900, dashes=(4, 8), label="Identity line")
#     plt.xlim(0, qmax)
#     plt.ylim(0, qmax)
#     plt.xlabel('Input variance ($\nu^{l-1}$)')
#     plt.ylabel('Output variance ($\nu^l$)')
#     plt.title("Iterative variance map")

#     nn = len(test_data)
#     col_i = 0
#     shade_i = 5
#     for test, attr in zip(test_data, tests):
#         for dist in attr["distributions"]:
#             for act in attr["activations"]:
#                 if dist['dist'] == "none":
#                     if act == "tanh":
#                         label = "tanh - None"
#                         col_i = 0
#                     else:
#                         label = "ReLU - None"
#                         col_i = 2
#                 elif "gauss" in dist['dist']:
#                     if act == "tanh":
#                         label = "tanh - Add Gauss $(\sigma^2_\epsilon = $ " + str(dist['std']) + ")"
#                         col_i = 1
#                     else:
#                         label = "ReLU - Add Gauss $(\sigma^2_\epsilon = $ " + str(dist['std']) + ")"
#                         col_i = 3
#                 for init in attr["inits"]:
#                     dashes = (None, None)

#                     # extract test data
#                     qmaps = test[dist['dist']][act][init]['q_maps']['qmaps']
#                     single_layer_sims = test[dist['dist']][act][init]['single_layer_qmap_sim']

#                     ############################################################################
#                     # left
#                     ############################################################################
#                     for w, b in zip(widxs, bidxs):

#                         # plot means of simulation (as dots)
#                         mu = single_layer_sims[w, b].mean(-1).mean(-2)
#                         # ax1.plot(qrange, mu, w, b, marker='o', ls='none', markersize=1, alpha=0.9, zorder=0, c=pal[col_i][shade_i])
#                         ax1.plot(qrange, mu, w, b, marker='o', markersize=4, alpha=0.9, color=pal[col_i][shade_i])

#                         # add confidence interval around simulation
#                         std = single_layer_sims[w, b].mean(-1).std(-2)
#                         ax1.fill_between(qrange, mu-std, mu+std, alpha=0.4, label='_nolegend_', color=pal[col_i][3])

#                         # theory line
#                         ax1.plot(qrange, qmaps[0, 0, :, 1], c=pal[col_i][shade_i], label=label, dashes=dashes)

#     leg = ax1.legend(prop={'size': 15})

#     # set the linewidth of each legend object
#     for legobj in leg.legendHandles:
#         legobj.set_linewidth(3.0)

#     plt.gcf().tight_layout()
#     plt.savefig(os.path.join(figures_dir, "{name}_iterative_map.pdf"))

# def plot_dynamics():
#     pass

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
    #pal = get_colours(10, 7)
    pal = get_colours(6, 7)[2:]
    test_data = []
    for i, test in enumerate(tests):
        test_data.append(load_experiment(test, ["q_maps", "single_layer_qmap_sim", "multi_layer_qmap_sim"], results_dir))

    fig = plt.figure(figsize=(8, 3))
    # fig = plt.figure(figsize=(12, 4.5))
    # fig = plt.figure(figsize=(16, 6))

    gs = plt.GridSpec(1, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])

    # Add unity line
    ax1.plot((0, qmax), (0, qmax), '--', color='k', zorder=900, dashes=(12, 24), label="Identity line")
    # ax1.plot((0, qmax), (0, qmax), '--', color='k', zorder=900, dashes=(4, 8), label="Identity line")
    ax1.set_xlim(0, qmax)
    ax1.set_ylim(0, qmax)
    ax1.set_xlabel(r'Input variance ($\nu^{l-1})$')
    ax1.set_ylabel(r'Output variance ($\nu^l$)')
    ax1.set_title("Iterative variance map")

    # axis 2
    ax2.set_xlim(0, qmax-1) #n_hidden_layers-1)
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
                        label = "tanh - None"
                        col_i = 0
                    else:
                        label = "ReLU - None"
                        col_i = 2

                elif "gauss" in dist['dist']:
                    if act == "tanh":
                        label = "tanh - Add Gauss $(\sigma^2_\epsilon = $ " + str(dist['std']) + ")"
                        col_i = 1
                    else:
                        label = "ReLU - Add Gauss $(\sigma^2_\epsilon = $ " + str(dist['std']) + ")"
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
                        # ax1.plot(qrange, mu, w, b, marker='o', ls='none', markersize=1, alpha=0.9, zorder=0, c=pal[col_i][shade_i])

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
                        # ax2.plot(xx, mu, 'o', markersize=4, alpha=0.9, color=pal[col_i][shade_i], label="Simulation")

    ##############
    # add labels #
    ##############
    fig.text(0.02, 0.95, "(a)")
    # fig.text(0.02, 0.95, "(a)", fontsize=20)
    fig.text(0.52, 0.95, "(b)")
    # fig.text(0.52, 0.95, "(b)", fontsize=20)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5,-0.15)
    )

    plt.gcf().tight_layout()
    plt.savefig(os.path.join(figures_dir, "tanh.pdf"), bbox_inches='tight')
    # plt.savefig("tanh.pdf", dpi=200)

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


    # fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 3))
    # fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(8, 2))
    # fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(12, 3))
    # fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(16, 5))

    # Add unity line
    ax1.plot((0, qmax), (0, qmax), '--', color='k', zorder=900, dashes=(12, 24), label="Identity line")
    ax1.set_xlim(0, qmax)
    ax1.set_ylim(0, qmax)
    ax1.set_xlabel(r'Input variance ($\nu^{l-1})$')
    ax1.set_ylabel(r'Output variance ($\nu^l$)')
    ax1.set_title("Iterative variance map")
    ax1.text(2, 10, r'$\sigma^2_w > \frac{2}{\mu_2}$')
    # ax1.text(2, 10, r'$\sigma^2_w > \frac{2}{\mu_2}$', fontsize=20)
    ax1.text(10, 1, r'$\sigma^2_w < \frac{2}{\mu_2}$')
    # ax1.text(10, 1, r'$\sigma^2_w < \frac{2}{\mu_2}$', fontsize=20)
    ax1.text(11, 10, r'$\sigma^2_w = \frac{2}{\mu_2}$')
    # ax1.text(11, 8.5, r'$\sigma^2_w = \frac{2}{\mu_2}$', fontsize=20)

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
                label = "dropout"
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
                        # ax1.plot(qrange, mu, w, b, marker='o', ls='none', markersize=1, alpha=0.9, zorder=0, c=pal[col_i][shade_i])

                        # add confidence interval around simulation
                        std = single_layer_sims[w, b].mean(-1).std(-2)
                        ax1.fill_between(qrange, mu-std, mu+std, alpha=0.4, label='_nolegend_', color=pal[col_i][shade_i])

                        # theory line
                        ax1.plot(qrange, qmaps[0, 0, :, 1], c=pal[col_i][shade_i], label=label, dashes=dashes)
                        # fixed point


                    ############################################################################
                    # middle
                    ############################################################################
                    q = 1
                    xx = np.arange(multi_layer_sims.shape[-2])
                    for w, b in zip(widxs, bidxs):
                        # confidence intervals
                        mu = multi_layer_sims[w, b].mean(axis=-1).mean(axis=0)
                        std = multi_layer_sims[w, b].mean(axis=-1).std(axis=0)

                        # plot theory
                        if "dropout" in label and not bern_label:
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
                        # ax2.plot(xx, mu, 'o', markersize=4, alpha=0.9, color=pal[col_i][shade_i])

    # leg = ax2.legend()
    # leg = ax2.legend()

    # # set the linewidth of each legend object
    # for legobj in leg.legendHandles:
    #     legobj.set_linewidth(3.0)

    # mu21 = np.linspace(1, 2, 100)
    # sigma1 = 2/mu21

    # ############################################################################
    # # right
    # ############################################################################
    # ax3.plot(mu21, sigma1, c="purple", label="Variance critical boundary", linestyle='--')
    # ax3.fill_between(mu21, 1, sigma1, facecolor='blue', alpha=0.2)
    # ax3.fill_between(mu21, 2, sigma1, facecolor='red', alpha=0.2)
    # ax3.text(1.5, 1.6, 'Overflow')
    # # ax3.text(1.5, 1.6, 'Overflow', fontsize=25)
    # ax3.text(1.55, 1.5, r'($\sigma^2_w > \frac{2}{\mu_2}$)')
    # # ax3.text(1.55, 1.5, r'($\sigma^2_w > \frac{2}{\mu_2}$)', fontsize=15)
    # ax3.text(1.1, 1.2, 'Underflow')
    # # ax3.text(1.1, 1.2, 'Underflow', fontsize=25)
    # ax3.text(1.15, 1.1, r'($\sigma^2_w < \frac{2}{\mu_2}$)')
    # # ax3.text(1.15, 1.1, r'($\sigma^2_w < \frac{2}{\mu_2}$)', fontsize=15)
    # ax3.text(1.2, 1.7, r'$\sigma^2_w = \frac{2}{\mu_2}$')
    # # ax3.text(1.2, 1.7, r'$\sigma^2_w = \frac{2}{\mu_2}$', fontsize=18)
    # ax3.set_xlim(1, 2)
    # ax3.set_ylim(1, 2)
    # ax3.set_xlabel('Weight initialisation ($\sigma^2_w$)')
    # ax3.set_ylabel('Second moment of noise dist. ($\mu_2$)')

    # leg = ax3.legend()
    # leg = ax3.legend(prop={'size': 15})

    # # set the linewidth of each legend object
    # for legobj in leg.legendHandles:
    #     legobj.set_linewidth(3.0)

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
        final_handles, final_labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5,-0.1)
    )

    fig.text(0.02, 0.95, "(a)")
    fig.text(0.52, 0.95, "(b)")

    plt.gcf().tight_layout()
    plt.savefig(os.path.join(figures_dir, "vairance.pdf"), bbox_inches='tight')

def plot_variance_edge():
    mu21 = np.linspace(1, 2, 100)
    sigma1 = 2/mu21

    fig, ax3 = plt.subplots(1, 1, figsize=(4, 3))

    ax3.plot(mu21, sigma1, c="purple", label="Variance critical boundary", linestyle='--')
    ax3.fill_between(mu21, 1, sigma1, facecolor='blue', alpha=0.2)
    ax3.fill_between(mu21, 2, sigma1, facecolor='red', alpha=0.2)
    ax3.text(1.5, 1.6, 'Overflow')
    # ax3.text(1.5, 1.6, 'Overflow', fontsize=25)
    ax3.text(1.55, 1.5, r'($\sigma^2_w > \frac{2}{\mu_2}$)')
    # ax3.text(1.55, 1.5, r'($\sigma^2_w > \frac{2}{\mu_2}$)', fontsize=15)
    ax3.text(1.1, 1.2, 'Underflow')
    # ax3.text(1.1, 1.2, 'Underflow', fontsize=25)
    ax3.text(1.15, 1.1, r'($\sigma^2_w < \frac{2}{\mu_2}$)')
    # ax3.text(1.15, 1.1, r'($\sigma^2_w < \frac{2}{\mu_2}$)', fontsize=15)
    ax3.text(1.2, 1.7, r'$\sigma^2_w = \frac{2}{\mu_2}$')
    # ax3.text(1.2, 1.7, r'$\sigma^2_w = \frac{2}{\mu_2}$', fontsize=18)
    ax3.set_xlim(1, 2)
    ax3.set_ylim(1, 2)
    ax3.set_xlabel('Weight initialisation ($\sigma^2_w$)')
    ax3.set_ylabel('Second moment of noise dist. ($\mu_2$)')
    ax3.set_title('Variance propagation dynamics')

    ax3.legend()
    # fig.legend()
    # fig.legend(
    #     loc='lower center', ncol=3, bbox_to_anchor=(0.5,-0.15)
    # )

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
    # fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 3))
    # fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(18, 5))
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
                        # ax1.text(x, y, r"($\sigma_\epsilon = {})".format(dist['std']))
                    elif "prob_1" in dist:
                        label = "dropout"

                        ax1.text(
                            x, y, f"($p = {dist['prob_1']}$)",
                            horizontalalignment='right',
                            verticalalignment='top',
                        )
                    else:
                        label = dist["dist"]

                    # try:
                    #     label = "Mult Gauss ($\sigma_\epsilon = {}$) ".format(str(dist['std']))
                    #     label = "Mult Gauss"
                    # except:
                    #     try:
                    #         label ="dropout ($p = {}$)".format(str(dist['prob_1']))
                    #         label = "dropout"
                    #     except:
                    #         label = dist['dist']

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
                    # ax1.scatter(mu_x, mu_y, marker="o", s=4, alpha=0.9, color=pal[col_i][shade_i])
                    ax1.fill_between(mu_x, mu_y - std_y, mu_y + std_y, alpha=0.2, label='_nolegend_', color=pal[col_i][shade_i])

                    # Add unity line
                    ax1.plot((0, 1), (0, 1), '--', color='k', zorder=900)
                    ax1.set_xlim(0, 1)
                    ax1.set_ylim(0, 1)

                    # ax1.plot(cstars[0, 0], cstars[0, 0], marker='x', mew=3, c=pal[col_i][shade_i], zorder=999, alpha=1.0, clip_on=False)
                    ax1.plot(cstars[0, 0], cstars[0, 0], markersize=15, marker='x', mew=3, c=pal[col_i][shade_i], zorder=999, alpha=1.0, clip_on=False)

                    ax1.set_title(r'Iterative correlation map')

                    ############################################################
                    # middle - dynamics of convergence
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
                        # ax2.text(x, y, r"($\sigma_\epsilon = {})".format(dist['std']))
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
                        # ax2.scatter(np.arange(means[i].shape[0]), means[i], marker='o', s=8, alpha=0.9, color=pal[col_i][shade_i])
                        ax2.fill_between(x_axis, means[i] - std[i], means[i] + std[i], alpha=0.2, label='_nolegend_', color=pal[col_i][shade_i])

    ax1.set_xticks([0, 0.5, 1.0])
    ax1.set_yticks([0, 0.5, 1.0])
    ax1.set_xlabel(r'Input correlation ($c^{l-1})$')
    ax1.set_ylabel(r'Output correlation ($c^{l}$)')
    # leg = ax1.legend()
    # leg = ax1.legend(prop={'size': 12})

    # # set the linewidth of each legend object
    # for legobj in leg.legendHandles:
    #     legobj.set_linewidth(3.0)

    ax2.set_xlabel(r'Layer ($l$)')
    ax2.set_ylabel(r'Correlation ($c^{l})$')
    ax2.set_title(r'Dynamics of $c$')
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, correlations.shape[-1] - 1)

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
        final_handles, final_labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5,-0.1)
    )

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
    ax3.scatter(1, 1, c="red", marker='*', s=10, label='Edge of chaos')
    # ax3.scatter(1, 1, c="red", marker='*', label='Edge of chaos')
    # ax3.scatter(1, 1, c="red", marker='*', label='Edge of chaos', s=100)
    ax3.fill_between(mu2s, 0, 1, facecolor='cyan', alpha=0.2)
    ax3.text(3, 0.5, 'Ordered regime \n (vanishing gradients)')
    # ax3.text(3, 0.5, 'Ordered regime \n (vanishing gradients)', fontsize=12)
    leg = ax3.legend()
    # leg = ax3.legend(prop={'size': 12})

    # # set the linewidth of each legend object
    # for legobj in leg.legendHandles:
    #     legobj.set_linewidth(3.0)

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
    fig.text(0.02, 0.87, "(a)")
    fig.text(0.52, 0.87, "(b)")

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
    fig.text(0.02, 0.87, "(a)")
    fig.text(0.52, 0.87, "(b)")

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
            "c_stars": c_stars,
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
        "Bernoulli": dropout_data["convergence"],
        "Gaussian": gauss_data["convergence"]
    }

    depth_scale_data = {
        "Bernoulli": dropout_data["depth_scale"],
        "Gaussian": gauss_data["depth_scale"]
    }

    plot_rate_of_convergence(convergence_data)
    plot_theoretical_vs_measured_depth_scale(depth_scale_data)

def plot_depth_scales():
    shading = "gouraud"

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

    # fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2, 3)
    fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2, 3, figsize=(18.25, 7.5))
    # fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2, 3, figsize=(23, 10))

    for dataset, [ax1, ax2, ax3] in zip(["mnist", "cifar-10"], [[ax1, ax2, ax3], [ax4, ax5, ax6]]):

        ###############################
        # Variance propagation dynamics
        ###############################
        dataset_dir_name = dataset.replace("-", "")
        variance_path = os.path.join(results_dir, "variance_depth", dataset_dir_name)
        trainable_path = os.path.join(results_dir, "trainable_depth", dataset_dir_name)
        example_dict = np.load(os.path.join(variance_path, "variance_depth.npy"))
        inits = np.load(os.path.join(variance_path, "variance_depth_sigma.npy"))
        p = 0.6
        num_layers = 1000
        nets = np.linspace(10, num_layers, 1000, dtype=int)
        xv, yv = np.meshgrid(nets, inits, sparse=False, indexing='ij')

        Z1 = np.log(np.array(example_dict))

        bad_indices = np.isnan(Z1) + np.isinf(Z1)
        Z1 = np.ma.array(Z1, mask=bad_indices)
        cmap = mpl.cm.get_cmap(name="Spectral_r")
        cmap.set_bad('black')

        pcm = ax1.pcolormesh(yv, xv, Z1.T, cmap=cmap, shading=shading, linewidth=0)
        cbar = fig.colorbar(pcm, ax=ax1, extend='max')
        cbar.ax.set_title(r'$log(\nu^l)$')

        ax1.set_xlabel('Weight initialisation ($\sigma^2_w$)')
        ax1.set_ylabel("Number of layers")
        ax1.set_title("{} - Variance propagation depth:\ndropout with $p$ = 0.6, crit. init. at $\sigma^2_w = 1.2$".format(dataset.upper()))

        max_depth = 0
        init_theory = np.linspace(0, 2.5, 1000)
        depth_per_p_theory = depth("Dropout", init_theory, p)
        max_depth = np.max([max_depth, np.max(depth_per_p_theory)])
        ax1.plot(init_theory, depth_per_p_theory, label="Theory", c='cyan')
        # ax1.plot(init_theory, depth_per_p_theory, label="Theory", c='cyan', linewidth=3)

        crit_point = critical_point("Dropout", p)
        ax1.plot([crit_point,]*2, [0, num_layers], color="black", linestyle="--", label="criticality", dashes=(2, 2))
        # ax1.plot([crit_point,]*2, [0, num_layers], color="black", linestyle="--", label="criticality", linewidth=4, dashes=(2, 2))

        ax1.set_ylim(0, 1000)
        ax1.set_xlim(0.1, 2.5)
        ax1.legend()
        ax1.set_xticks(inits[2:-2:3])
        ax1.text(0.2, 400, 'Underflow', color="white")
        # ax1.text(0.2, 400, 'Underflow', fontsize=25, color="white")
        ax1.text(1.7, 400, 'Overflow', color="white")
        # ax1.text(1.7, 400, 'Overflow', fontsize=25, color="white")

        rates = np.linspace(0.1, 1, 100)


        if dataset == "cifar-10":
            example_dict = np.load(os.path.join(trainable_path, "trainable_depth.npy"))
            example_dict = example_dict[:, :, -1, 0].T
        elif dataset == "mnist":
            pickle_in = open(os.path.join(results_dir, "val_loss_per_depth.pk"),"rb")
            example_dict = np.array(pickle.load(pickle_in))[:, 0]
        else:
            raise ValueError("dataset not supported")

        nets = np.linspace(2, 40, 10, dtype=int)
        d_rates = 2*np.linspace(0.1, 1, 10)
        xv, yv = np.meshgrid(nets, d_rates, sparse=False, indexing='ij')

        Z1 = np.array(example_dict)
        pcm = ax2.pcolormesh(yv+0.1, xv+0.1, Z1.reshape(10,10), cmap='Spectral_r', shading=shading, linewidth=0)
        cbar1 = fig.colorbar(pcm, ax=ax2, extend='max')
        cbar1.ax.set_title('Train loss')

        ################################
        # Training loss and depth scales
        ################################
        t = rates
        x = 2*rates
        y = 6*depth_scale(fp_slopes)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=plt.get_cmap('Greens'))
        lc.set_array(t)
        lc.set_linewidth(3)
        cbar2 = fig.colorbar(lc, ax=ax2)
        cbar2.ax.set_title('p')

        ax2.add_collection(lc)
        ax2.set_xlabel("Critical initialisation for $p$ ($\sigma^2_w$)")
        ax2.set_ylabel("Number of layers")
        txt = ax2.text(1.5, 19, r'$6\xi_c$', color="white")
        # txt = ax2.text(1.5, 19, r'$6\xi_c$', fontsize=30, color="white")
        txt.set_path_effects([PathEffects.withStroke(foreground='black')])
        # txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
        ax2.set_title("{} - Depth at criticality".format(dataset.upper()))
        ax2.set_xlim(0.3, 2.1)
        ax2.set_ylim(2, 40)

        if dataset == "cifar-10":
            example_dict = np.load(os.path.join(trainable_path, "trainable_depth.npy"))
            example_dict = example_dict[:, :, -1, 1].T
        elif dataset == "mnist":
            pickle_in = open(os.path.join(results_dir, "val_loss_per_depth.pk"),"rb")
            example_dict = np.array(pickle.load(pickle_in))[:,1]
        else:
            raise ValueError("dataset not supported")

        nets = np.linspace(2, 40, 10, dtype=int)
        d_rates = 2*np.linspace(0.1, 1, 10)
        xv, yv = np.meshgrid(nets, d_rates, sparse=False, indexing='ij')

        Z1 = np.array(example_dict)
        pcm = ax3.pcolormesh(yv+0.1, xv+0.1, Z1.reshape(10,10), cmap='Spectral_r', shading=shading, linewidth=0)
        cbar1 = fig.colorbar(pcm, ax=ax3, extend='max')
        cbar1.ax.set_title('Val. loss')

        ##################################
        # Validation loss and depth scales
        ##################################
        t = rates
        x = 2*rates
        y = 6*depth_scale(fp_slopes)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=plt.get_cmap('Greens'))
        lc.set_array(t)
        lc.set_linewidth(3)
        cbar2 = fig.colorbar(lc, ax=ax3)
        cbar2.ax.set_title('p')

        ax3.add_collection(lc)
        ax3.set_xlabel("Critical initialisation for $p$ ($\sigma^2_w$)")
        ax3.set_ylabel("Number of layers")
        txt = ax3.text(1.5, 19, r'$6\xi_c$', color="white")
        # txt = ax3.text(1.5, 19, r'$6\xi_c$', fontsize=30, color="white")
        txt.set_path_effects([PathEffects.withStroke(foreground='black')])
        # txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='black')])
        ax3.set_title("{} - Depth at criticality".format(dataset.upper()))
        ax3.set_xlim(0.3, 2.1)
        ax3.set_ylim(2, 40)

    ##############
    # add labels #
    ##############
    fig.text(0.03, 0.95, "(a)")
    # fig.text(0.03, 0.95, "(a)", fontsize=25)
    fig.text(0.36, 0.95, "(b)")
    # fig.text(0.36, 0.95, "(b)", fontsize=25)
    fig.text(0.68, 0.95, "(c)")
    # fig.text(0.68, 0.95, "(c)", fontsize=25)
    fig.text(0.03, 0.46, "(d)")
    # fig.text(0.03, 0.46, "(d)", fontsize=25)
    fig.text(0.36, 0.46, "(e)")
    # fig.text(0.36, 0.46, "(e)", fontsize=25)
    fig.text(0.68, 0.46, "(f)")
    # fig.text(0.68, 0.46, "(f)", fontsize=25)

    plt.gcf().tight_layout()
    plt.show()
    plt.savefig(os.path.join(figures_dir, "depth_scales.pdf"), bbox_inches='tight')

if __name__ == "__main__":
    # plot settings
    set_plot_params()

    # results directory
    results_dir = os.path.join(file_dir, "../results")

    # figures directory
    figures_dir = os.path.join(file_dir, "../figures")
    os.makedirs(figures_dir, exist_ok=True)

    # plot_tanh()
    # plot_variance()
    # plot_variance_edge()
    # plot_correlation()
    # plot_correlation_edge_theory()
    # plot_fixed_point_convergence()
    plot_depth_scales()
