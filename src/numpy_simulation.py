# Figures 1 and 2
# Generate figures matching theory and simulation of the propagation of norms
# and inner products in a random deep neural network.
import os, sys, pickle
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm

file_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_dir)
# sys.path.append(file_dir + '/../')

from theory import *
from constants import *
from numpy_net import Network

# file_dir = os.path.dirname(os.path.realpath(__file__))
# sys.path.insert(0, file_dir + '/../')

# file_dir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(os.path.join(file_dir, "../cornet/"))

# Theory for squared length map, $\mathcal{V}(q^{l-1} | \sigma_w, \sigma_b)$
# Compute fixed points, dynamics, and convergence of the $q$ map.

def qmap_statistics(
	dist=None, std=None, diversity=None, prob_1=None, _lambda=None,
	weight_sigmas=None, nonlinearity=None, replace=True):
	stats_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_q_maps.npz".format(
		dist, std, diversity, prob_1, _lambda
	))

	if os.path.exists(stats_path) and replace == False:
		return

	# Note: This could be sped up a bunch... - they left this, we should maybe speed it up then XD

	# Range of squared lengths for q map plot (panel A)
	nq = 100
	qmax = 15
	qrange = np.linspace(0, qmax, nq)

	# Number of iterations for the dynamics of convergence plot (panel B)
	nt = 20

	# Maximum number of iterations when computing convergence time (panel D)
	nconverge = 25

	qconverge = np.linspace(0, qmax, nconverge)
	qmaps = np.zeros((nw, nb, nq, nt))
	qstars = np.zeros((nw, nb))
	tstars = np.zeros((nw, nb, nconverge))
	qmaps[..., 0] = qrange[None, None, None, :]

	widx = bidx = 0
	weight_sigma = weight_sigmas[0]
	bias_sigma = bias_sigmas[0]

	# Compute fixed points
	_, qstars[widx, bidx] = q_fixed_point_noise(
		weight_sigma, bias_sigma, nonlinearity, qinit=3.0,
		tol_frac=0.01, dist=dist, std=std, diversity=diversity,
		prob_1=prob_1, _lambda=_lambda
	)
	# Iterate over initial norms, computing convergence time
	for tidx, qinit in enumerate(tqdm(qconverge)):
		tstars[widx, bidx, tidx], _ = q_fixed_point_noise(
			weight_sigma, bias_sigma, nonlinearity, qinit=qinit,
			tol_frac=0.01, dist=dist, std=std, diversity=diversity,
			prob_1=prob_1, _lambda=_lambda
		)
	# Dynamics of convergence of q map
	for t in tqdm(range(1, nt)):
		qmaps[widx, bidx, :, t] = qmap_noise(
			qmaps[widx, bidx, :, t - 1], weight_sigma, bias_sigma,
			nonlinearity, dist=dist, std=std, diversity=diversity,
			prob_1=prob_1, _lambda=_lambda
		)

	np.savez(stats_path, qmaps=qmaps, qstars=qstars, tstars=tstars)


def single_layer_net_statistics(
	dist=None, std=None, diversity=None, prob_1=None, _lambda=None,
	weight_sigmas=None, nonlinearity=None, init=None, replace=False
):
	stats_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_single_layer_qmap_sim.npy".format(
		dist, std, diversity, prob_1, _lambda
	))
	if os.path.exists(stats_path) and replace == False:
		return

	# Simulations for squared length in a random neural network
	# Create a feedforward fully-connected neural network.
	# Note that here we flip the order of activations and linear layers so that
	# the first operation is the activation function.
	nq = 100
	qmax = 15
	qrange = np.linspace(0, qmax, nq)

	# n_hidden_layers = 0
	n_hidden_layers = 1
	n_hidden_units = 1000
	din = 1000

	net = Network(
		input_dim=din, output_dim=din,
		hidden_layer_sizes=[n_hidden_units,]*(n_hidden_layers),
		activation=nonlinearity, weight_sigma=weight_sigmas[0], bias_sigma=bias_sigmas[0],
		dist=dist, std=std, p=prob_1, init_method=init
	)

	# Simulate a single layer at a large number of input norms (for panel A)
	npnts = 20
	qmap_sim = np.empty((nw, nb, nq, npnts))
	for norm_idx, qnorm in enumerate(tqdm(qrange)):
		xs = np.random.normal(loc=0, scale=np.sqrt(qnorm), size=(npnts, din))
		h = net.get_acts(xs)[-1]
		qmap_sim[0, 0, norm_idx] = (1.0 / din) * (h**2).sum(-1)

	np.save(stats_path, qmap_sim)


def multi_layer_net_statistics(
	dist=None, std=None, diversity=None, prob_1=None, _lambda=None,
	weight_sigmas=None, nonlinearity=None, init=None, replace=False
):
	stats_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_multi_layer_qmap_sim.npy".format(
		dist, std, diversity, prob_1, _lambda
	))

	if os.path.exists(stats_path) and replace == False:
		return

	# n_hidden_layers = 20
	n_hidden_layers = 19
	n_hidden_units = 1000
	din = 1000

	net = Network(
		input_dim=din, output_dim=din,
		hidden_layer_sizes=[n_hidden_units,]*(n_hidden_layers),
		activation=nonlinearity, weight_sigma=weight_sigmas[0], bias_sigma=bias_sigmas[0],
		dist=dist, std=std, p=prob_1, init_method=init
	)

	# Simulate multiple layers at a small set of points (for dynamics in panel B)
	qidx = [26, 53, 80]
	nq = 100
	qmax = 15
	qrange = np.linspace(0, qmax, nq)
	qnorms = qrange[qidx]
	nqnorms = len(qnorms)

	qmaps_sim = np.zeros((nw, nb, nqnorms, n_hidden_layers+1, nq))
	for norm_idx, qnorm in enumerate(tqdm(qnorms)):
		xs = np.random.randn(nq, din) * np.sqrt(qnorm)
		acts = np.array(net.get_acts(xs))
		qmaps_sim[0, 0, norm_idx, 0] = 1.0 / din * (xs**2).sum(-1)
		qmaps_sim[0, 0, norm_idx, 1:] = 1.0 / din * (acts**2).sum(-1)
		# qmaps_sim[0, 0, norm_idx] = 1.0 / din * (acts**2).sum(-1)

	np.save(stats_path, qmaps_sim)


# ## Theory for covariance propagation and curvature
nq = 51
c12s = np.linspace(0, 1.0, nq)
crange = c12s


def cov_prop(dist=None, std=None, diversity=None, prob_1=None, _lambda=None,
weight_sigmas=None, nonlinearity=None, replace=False):
	save_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_cmap.npz".format(
		dist, std, diversity, prob_1, _lambda
	))

	if os.path.exists(save_path) and replace == False:
		return

	qmap_theory_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_q_maps.npz".format(
		dist, std, diversity, prob_1, _lambda
	))
	statistics = np.load(qmap_theory_path)
	qstars = statistics["qstars"]

	# ### Covariance propagation
	cmaps = np.zeros((nw, nb, nq))
	cstars = np.zeros((nw, nb))
	for widx, weight_sigma in enumerate(tqdm(weight_sigmas, desc="weights")):
		for bidx, bias_sigma in enumerate(tqdm(bias_sigmas, desc="bias'")):
			q1 = qstars[widx, bidx]
			cin = c12s  # * q1
			cout = covmap(q1, q1,  cin, weight_sigma, bias_sigma, nonlinearity)
			cmaps[widx, bidx, :] = cout
			# Remove  fixed point at cin = q*
			cstar = cin[np.argmin(np.abs(cin[:-1] - cout[:-1]))]
			cstars[widx, bidx] = cstar

	# cstars /= qstars
	# cmaps /= qstars[:, :, None]

	np.savez(save_path, cstars=cstars, cmaps=cmaps)


def get_correlated_set(length=1000, size=50, set_type="linspace", start_correlation=None):
	# define a function that generates observations that are correlated to the input
	def correlated_value(x, r):
		# This code was adapted from this source:
		# https://stats.stackexchange.com/questions/38856/how-to-generate-correlated-random-numbers-given-means-variances-and-degree-of
		# the post by "gung" answered Oct 7 '12 at 22:11 edited Jan 11 '16 at 17:33
		# BibTex reference:
		# @MISC {38867,
		# 	TITLE = {How to generate correlated random numbers (given means, variances and degree of correlation)?},
		# 	AUTHOR = {gung (https://stats.stackexchange.com/users/7290/gung)},
		# 	HOWPUBLISHED = {Cross Validated},
		# 	NOTE = {URL:https://stats.stackexchange.com/q/38867 (version: 2016-01-11)},
		# 	EPRINT = {https://stats.stackexchange.com/q/38867},
		# 	URL = {https://stats.stackexchange.com/q/38867}
		# }
		r_squared = r**2
		variance_error = 1 - r_squared
		std_error = np.sqrt(variance_error)
		error = np.random.normal(loc=0, scale=std_error, size=x.shape)
		y = r * x + error
		return y

	# generate an initial observation
	x = np.random.randn(1, length)

	# create a set (matrix) to store all correlated observations
	X = [x, ]

	# create a list to store all actual correlation values
	C = []

	# create the correlation values we would like our generated points to have
	if start_correlation is None:
		start_correlation = 1 - (1 / size)

	if set_type == "linspace":
		correlations = np.linspace(start_correlation, 0, size)
	elif set_type == "same":
		correlations = np.array([start_correlation, ] * size)
	else:
		raise ValueError("a set_type of {} is invalid".format(set_type))

	for corr in correlations:
		# generate an observation that is approximately correlated to x by corr
		y = correlated_value(x, corr)

		# add it to the set
		X.append(y)

		# calculate and store it's actual correlation to x
		# Note: the `np.corrcoef` function returns a matrix but we are only
		# interested in the cross correlation entry in it (C0,1 or C1,0)
		actual_corr = np.corrcoef(x, y)[0, 1]
		C.append(actual_corr)

	return [np.array(X).squeeze(), np.array(C)]


def get_set_correlation(X):
	num_observations = X.shape[0]

	# create list to store calculated correlations in
	C = np.empty(num_observations - 1)

	for i in np.arange(1, num_observations):
		# Note: the `np.corrcoef` function returns a matrix but we are only
		# interested in the cross correlation entry in it (C0,1 or C1,0)
		C[i - 1] = np.corrcoef(X[0], X[i])[0, 1]

	return C

# Simulation for the curvature propagation - left panel
def cov_prop_sim(dist=None, std=None, diversity=None, prob_1=None, _lambda=None,
weight_sigmas=None, nonlinearity=None, init=None, replace=False):
	save_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_cmap_sim.npz".format(
		dist, std, diversity, prob_1, _lambda
	))

	if os.path.exists(save_path) and replace == False:
		return

	# may need to change this to a circle rand net
	n_hidden_layers = 1
	n_hidden_units = 1000
	din = 1000
	net = Network(
		input_dim=din, output_dim=din,
		hidden_layer_sizes=[n_hidden_units,]*(n_hidden_layers),
		activation=nonlinearity, weight_sigma=weight_sigmas[0], bias_sigma=bias_sigmas[0],
		dist=dist, std=std, p=prob_1, init_method=init
	)

	num_trials = 100
	num_correlations = 50

	# create places to store results over multiple runs
	input_correlations = np.empty((num_trials, num_correlations))
	output_correlations = np.empty(input_correlations.shape)

	for trial in tqdm(np.arange(num_trials)):
		# get a set of observations where the correlations between them is known
		net_input, input_correlation = get_correlated_set(
			length=din, size=num_correlations)

		# feed this set into the network
		net_output = net.get_acts(net_input)[-1]

		# calculate the correlations between the new set
		output_correlation = get_set_correlation(net_output)

		# store results
		input_correlations[trial] = input_correlation
		output_correlations[trial] = output_correlation

	np.savez(save_path, input_correlations=input_correlations,
			 output_correlations=output_correlations)

# Simulation for the curvature propagation for a multi-layer network - right panel


def multi_layer_cov_prop_sim(dist=None, std=None, diversity=None, prob_1=None, _lambda=None,
weight_sigmas=None, nonlinearity=None, init=None, replace=False):
	save_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_multi_layer_cmap_sim.npy".format(
		dist, std, diversity, prob_1, _lambda
	))

	if os.path.exists(save_path) and replace == False:
		return

	# may need to change this to a circle rand net
	n_hidden_layers = 30
	n_hidden_units = 1000
	din = 1000
	net = Network(
		input_dim=din, output_dim=din,
		hidden_layer_sizes=[n_hidden_units,]*(n_hidden_layers),
		activation=nonlinearity, weight_sigma=weight_sigmas[0], bias_sigma=bias_sigmas[0],
		dist=dist, std=std, p=prob_1, init_method=init
	)

	num_trials = 1000
	n_starting_correlations = cin.shape[0]
	starting_correlations = np.array(cin)

	# create places to store results over multiple runs
	correlations = np.empty(
		(n_starting_correlations, num_trials, n_hidden_layers + 1))

	for starting_correlation_index in tqdm(np.arange(n_starting_correlations)):

		# fetch the value we want the correlation to start at
		starting_correlation = starting_correlations[starting_correlation_index]

		# get a set of observations where the correlations between them is known
		net_input, input_correlation = get_correlated_set(
			length=din, size=num_trials, set_type="same",
			start_correlation=starting_correlation
		)

		# add the input to the network as the 0'th correlation
		correlations[starting_correlation_index, :, 0] = input_correlation

		# feed this set into the network
		net_activations = net.get_acts(net_input)

		for layer_index in tqdm(np.arange(n_hidden_layers)):
			# calculate the correlations between the new set (formed by each layer's activations)
			correlations[starting_correlation_index, :, layer_index + 1] =\
				get_set_correlation(net_activations[layer_index])

	np.save(save_path, correlations)


cin = np.array([0.0, 0.5, 0.9])  # NB: this must be changed in plot_exp.py too!
nctraj = len(cin)
nt = 31


def cov_prop_specific(dist=None, std=None, diversity=None, prob_1=None, _lambda=None,
weight_sigmas=None, nonlinearity=None, replace=False):
	save_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_ctrajs.npy".format(
		dist, std, diversity, prob_1, _lambda
	))

	if os.path.exists(save_path) and replace == False:
		return

	qmap_theory_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_q_maps.npz".format(
		dist, std, diversity, prob_1, _lambda
	))
	statistics = np.load(qmap_theory_path)
	qstars = statistics["qstars"]

	# ### Covariance dynamics for subset of points
	ctrajs = np.zeros((len(widxs), nctraj, nt))
	for i, (widx, bidx) in enumerate(tqdm(zip(widxs, bidxs), total=len(widxs))):
		q1 = qstars[widx, bidx]
		ctrajs[i, :, 0] = cin  # * q1
		for t in range(1, nt):
			ctrajs[i, :, t] = covmap(q1, q1, ctrajs[i, :, t-1],
									 weight_sigmas[widx], bias_sigmas[bidx], nonlinearity)
		# ctrajs[i] /= q1
	ctrajs_ = ctrajs.copy()

	ctrajs = np.zeros((nw, nb, nctraj, nt))
	for i, (widx, bidx) in enumerate(tqdm(zip(widxs, bidxs), total=len(widxs))):
		ctrajs[widx, bidx] = ctrajs_[i]

	np.save(save_path, ctrajs)


def curv_prop(dist=None, std=None, diversity=None, prob_1=None, _lambda=None, weight_sigmas=None, dphi=None, replace=False):
	save_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_chi1.npy".format(
		dist, std, diversity, prob_1, _lambda
	))

	if os.path.exists(save_path) and replace == False:
		return

	# ### Theory for curvature propagation
	chi1 = np.zeros((nw, nb))
	q_fixed_points = np.zeros((nw, nb))
	n_layers = 31  # only plot up to layer 30 in panel 2B

	qmap_theory_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_q_maps.npz".format(
		dist, std, diversity, prob_1, _lambda
	))
	statistics = np.load(qmap_theory_path)
	q_fixed_points = statistics["qstars"]

	for widx, weight_sigma in enumerate(tqdm(weight_sigmas, desc="weights")):
		for bidx, bias_sigma in enumerate(tqdm(bias_sigmas, desc="bias'")):
			chi1[widx, bidx] = compute_chi1(
				q_fixed_points[widx, bidx], weight_sigma, bias_sigma, dphi)

	np.save(save_path, chi1)


def load_qstars(dist=None, std=None, diversity=None, prob_1=None, _lambda=None):
	qmap_theory_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_q_maps.npz".format(
		dist, std, diversity, prob_1, _lambda
	))
	statistics = np.load(qmap_theory_path)
	return statistics["qstars"]


def load_tstars(dist=None, std=None, diversity=None, prob_1=None, _lambda=None):
	qmap_theory_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_q_maps.npz".format(
		dist, std, diversity, prob_1, _lambda
	))
	statistics = np.load(qmap_theory_path)
	return statistics["tstars"].mean(-1)


def load_qmaps(dist=None, std=None, diversity=None, prob_1=None, _lambda=None):
	qmap_theory_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_q_maps.npz".format(
		dist, std, diversity, prob_1, _lambda
	))
	statistics = np.load(qmap_theory_path)
	return statistics["qmaps"]


def single_layer_qmap_sim(dist=None, std=None, diversity=None, prob_1=None, _lambda=None):
	qmap_sim_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_single_layer_qmap_sim.npy".format(
		dist, std, diversity, prob_1, _lambda
	))
	return np.load(qmap_sim_path)


def multi_layer_qmap_sim(dist=None, std=None, diversity=None, prob_1=None, _lambda=None):
	qmap_sim_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_multi_layer_qmap_sim.npy".format(
		dist, std, diversity, prob_1, _lambda
	))
	return np.load(qmap_sim_path)


def load_cmap(dist=None, std=None, diversity=None, prob_1=None, _lambda=None):
	cmap_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_cmap.npz".format(
		dist, std, diversity, prob_1, _lambda
	))
	cmap_data = np.load(cmap_path)
	return cmap_data["cmaps"]


def load_cstar(dist=None, std=None, diversity=None, prob_1=None, _lambda=None):
	cmap_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_cmap.npz".format(
		dist, std, diversity, prob_1, _lambda
	))
	cmap_data = np.load(cmap_path)
	return cmap_data["cstars"]


def load_ctraj(dist=None, std=None, diversity=None, prob_1=None, _lambda=None):
	ctrajs_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_ctrajs.npy".format(
		dist, std, diversity, prob_1, _lambda
	))
	return np.load(ctrajs_path)


def load_acorr_mu(dist=None, std=None, diversity=None, prob_1=None, _lambda=None):
	autocorr_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_curv_stats.npz".format(
		dist, std, diversity, prob_1, _lambda
	))
	autocorr_data = np.load(autocorr_path)
	return autocorr_data['r_acts'].squeeze()


def load_acorr_std(dist=None, std=None, diversity=None, prob_1=None, _lambda=None):
	autocorr_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_curv_stats.npz".format(
		dist, std, diversity, prob_1, _lambda
	))
	autocorr_data = np.load(autocorr_path)
	return autocorr_data['r_acts_std'].squeeze()

# # ## Curvature Simulations


# def get_circle_RandNet(
#     dist=None, std=None, diversity=None, prob_1=None, _lambda=None
# ):
#     # Construct a model that maps an angle to an input vector in $d$ dimensions, and then passes it through a deep fully connected neural network.
#     #
#     # The `GreatCircle` input layer takes a scalar angle, $\theta$, as input, and outputs the point along a great cirlce in $d$ dimensions as its output:
#     # $$\mathtt{gc\_input}(\theta) = \sqrt{q} \left(u_0 \cos(\theta) + u_1 \sin (\theta)\right)$$
#     # where $\theta, q \in \mathbb{R}$ and $u_0, u_1 \in \mathbb{R}^d$.

#     net_save_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_circle_model".format(
#         dist, std, diversity, prob_1, _lambda
#     ))

#     if os.path.exists(net_save_path):
#         net = RandNet(None, None, None, model_file=net_save_path)
#     else:
#         _n_hidden_layers = 30  # Number of hidden layers
#         _n_hidden_units = 1000  # Number of hidden units
#         _din = _n_hidden_units  # Input dimensionality

#         gc_input = GreatCircle(_din)
#         net = RandNet(
#             _din, _n_hidden_units, _n_hidden_layers, nonlinearity=nonlinearity_str,
#             input_layer=gc_input, dist=dist, std=std, diversity=diversity,
#             prob_1=prob_1, _lambda=_lambda, flip=True, init_method=init
#         )

#     return net

# ### Compute the fixed point of the squared length from theory, $q^*(\sigma_w, \sigma_b)$
# This is needed to appropriately scale the input norm for the simulations
def load_q_stars(
	dist=None, std=None, diversity=None, prob_1=None, _lambda=None
):
	qstars_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_q_maps_curv.npy".format(
		dist, std, diversity, prob_1, _lambda
	))

	if os.path.exists(qstars_path):
		qstars = np.load(qstars_path)
	else:
		_weight_sigmas = weight_sigmas  # weight_sigmas[widxs]
		_bias_sigmas = bias_sigmas  # bias_sigmas[bidxs][:1]
		_nw = len(weight_sigmas)
		_nb = len(bias_sigmas)

		qstars = np.zeros((_nw, _nb))
		for widx, weight_sigma in enumerate(tqdm(_weight_sigmas, desc="weights")):
			for bidx, bias_sigma in enumerate(tqdm(_bias_sigmas, desc="bias'")):
				_, qstars[widx, bidx] = q_fixed_point_noise(
					weight_sigma, bias_sigma, nonlinearity, dist=dist, std=std,
					diversity=diversity, prob_1=prob_1, _lambda=_lambda
				)
		np.save(qstars_path, qstars)
	return qstars

# # ### Compute statistics on simulated network
# def curvature_stats(
#     dist=None, std=None, diversity=None, prob_1=None, _lambda=None, replace=False
# ):
#     net_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_circle_net.npy".format(
#         dist, std, diversity, prob_1, _lambda
#     ))
#     stats_path = os.path.join(results_dir, "{}_{}_{}_{}_{}_curv_stats.npz".format(
#         dist, std, diversity, prob_1, _lambda
#     ))

#     if os.path.exists(stats_path) and replace == False:
#         return

#     net = get_circle_RandNet(
#         dist=dist, std=std, diversity=diversity, prob_1=prob_1, _lambda=_lambda
#     )
#     qstars = load_q_stars(
#         dist=dist, std=std, diversity=diversity, prob_1=prob_1, _lambda=_lambda
#     )

#     _n_interp = 500  # Number of points to interpolate along the input angle, theta
#     ts = np.linspace(-np.pi, np.pi, _n_interp, endpoint=False)

#     _weight_sigmas = weight_sigmas  # weight_sigmas[widxs]
#     _bias_sigmas = bias_sigmas  # bias_sigmas[bidxs][:1]
#     _n_hidden_layers = 30  # Number of hidden layers
#     _n_hidden_units = 1000  # Number of hidden units
#     _din = _n_hidden_units  # Input dimensionality

#     compute_minimal_curvature_net(
#         net, ts, [0], _weight_sigmas, _bias_sigmas, qstar=qstars, save_file=net_path
#     )
#     compute_minimal_curvature_statistics_no_net(
#         _n_hidden_layers, _n_hidden_units, ts, [0], _weight_sigmas, _bias_sigmas,
#         qstar=qstars, acts_file=net_path, save_file=stats_path
#     )


def load(test, data_names):
	data = {}
	for dist in test["distributions"]:
		data[dist["dist"]] = {}

		for act in test["activations"]:
			data[dist["dist"]][act] = {}

			for init in test["inits"]:
				data[dist["dist"]][act][init] = {}

				for name in data_names:
					base_path = os.path.join(
						file_dir, relative_results_dir, dist["dist"], act, init
					)
					data[dist["dist"]][act][init][name] = load_file(base_path, name, **dist)

	return data


def load_file(path, name, dist=None, std=None, diversity=None, prob_1=None, _lambda=None):
	try:
		data_path = os.path.join(path, "{}_{}_{}_{}_{}_{}.npy".format(
			dist, std, diversity, prob_1, _lambda, name
		))
		print(data_path)
		return np.load(data_path)
	except:
		print("cant' find npy file, looking for npz...")
		try:
			data_path = os.path.join(path, "{}_{}_{}_{}_{}_{}.npz".format(
				dist, std, diversity, prob_1, _lambda, name
			))
			print(data_path)
			return np.load(data_path)
		except:
			raise FileNotFoundError("can't find {}".format(path))


def mu_2(dist=None, std=None, prob_1=None): #std=None, prob_1=None):
	if isinstance(dist, str):
		dist = dist.lower()

		if "gauss" in dist:
			return std**2 + 1
		elif "lap" in dist:
			raise NotImplementedError("Laplace noise not implemented yet.")
		elif "pois" in dist:
			return 2
		elif "drop" in dist or "bern" in dist:
			return 1 / prob_1
	else:
		raise TypeError("dist must be a string")


def noisy_signal_prop_simulations(dist=None, noise=None, act=None, init=None, replace=True):
	file_dir = os.path.dirname(os.path.realpath(__file__))
	sys.path.insert(0, file_dir + '/../')

	noise_type, noise_level = noise
	if noise_type is not None:
		test = {"dist": dist, noise_type:noise_level}
	else:
		test = {"dist": dist}

	if "over" in init:
		weight_sigmas = [1.25 * np.sqrt(2 / mu_2(**test))]
		# weight_sigmas = [(1 + (np.random.rand() * 0.1 + 0.05)) * np.sqrt(2 / mu_2(**test))]
	elif "under" in init:
		weight_sigmas = [1 - (np.random.rand() * 0.1 + 0.05) * np.sqrt(2 / mu_2(**test))]
	elif "crit" in init:
		if dist == "none":
			weight_sigmas = [np.sqrt(2)]
		else:
			weight_sigmas = [np.sqrt(2 / mu_2(**test))]

	global results_dir
	results_dir = os.path.join(file_dir, "results", dist, act, init)
	if not os.path.exists(results_dir):
		os.makedirs(results_dir, exist_ok=True)

	nonlinearity = nonlinearities[act]
	nonlinearity_str = act
	phi = nonlinearity
	dphi = dphis[act]
	d2phi = d2phis[act]

	print("####### EXPERIMENT: dist: ", dist, '; ', noise_type, ': ', noise_level, '; activation: ', act, '; initialisation: ', init, '##############')
	REPLACE=replace
	# REPLACE=False

	############################################################################

	print("qmap calculations...")
	print("Calculating Theory:")
	qmap_statistics(**test, weight_sigmas=weight_sigmas, nonlinearity=nonlinearity, replace=REPLACE)

	print("Simulating network")
	print("Single layer sims...")
	single_layer_net_statistics(**test, weight_sigmas=weight_sigmas,
								nonlinearity=nonlinearity_str, init=init, replace=REPLACE)

	print("Multi-layer sims...")
	multi_layer_net_statistics(**test, weight_sigmas=weight_sigmas,
							   nonlinearity=nonlinearity_str, init=init, replace=REPLACE)

	############################################################################


	############################################################################

	print("Cmap calculations...")
	print("General cov prop...")
	cov_prop(**test, weight_sigmas=weight_sigmas, nonlinearity=nonlinearity, replace=REPLACE)

	print("Specific cov prop...")
	cov_prop_specific(**test, weight_sigmas=weight_sigmas, nonlinearity=nonlinearity, replace=REPLACE)

	print("Curvature prop...")
	curv_prop(**test, weight_sigmas=weight_sigmas, dphi=dphi, replace=REPLACE)

	print("General cov prop simulation...")
	cov_prop_sim(**test, weight_sigmas=weight_sigmas,
				 nonlinearity=nonlinearity_str, init=init, replace=REPLACE)

	print("Simulate multi-layer cov prop...")
	multi_layer_cov_prop_sim(**test, weight_sigmas=weight_sigmas,
							 nonlinearity=nonlinearity_str, init=init, replace=REPLACE)

	############################################################################
