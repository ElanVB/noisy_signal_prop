import numpy as np
import mnist, cifar10, sys, os

from tqdm import tqdm
from theory import depth
from numpy_net import Network
from data_iterator import DataIterator

dataset = "mnist" # "cifar10"

file_dir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(file_dir, "results/variance_depth/{}".format(dataset))
data_save_path = os.path.join(save_dir, "variance_depth.npy")
sigma_save_path = os.path.join(save_dir, "variance_depth_sigma.npy")

os.makedirs(save_dir, exist_ok=True)

def test_init(weight_sigma_square):
	p = 0.6
	max_num_layers = 1000
	num_hidden_layers = np.floor(np.min([max_num_layers, depth("Dropout", weight_sigma_square, p) * 1.1])).astype(int)

	print("Testing sigma_w = {: 2.2f}; Using {: 4d} layers...".format(weight_sigma_square, num_hidden_layers))

	batch_size = 128

	input_data = None
	if dataset == "mnist":
		input_data = mnist.load().astype(np.float32)
	elif dataset == "cifar10":
		input_data = cifar10.load().astype(np.float32)

	# batch data for memory purposes
	input_data_iterator = DataIterator(batch_size, input_data, input_data)
	num_batches = input_data_iterator.size()

	input_size = input_data.shape[-1]
	net = Network(
		input_size, input_size, [1000,] * num_hidden_layers, activation="relu",
		weight_sigma=np.sqrt(weight_sigma_square), dist="bern", p=p
	)

	variances = np.empty((num_batches, max_num_layers))
	variances.fill(np.nan)

	with tqdm(desc="batches", total=num_batches) as progress_bar:
		for i, batch in enumerate(input_data_iterator):
			variance = net.get_acts(batch.input, early_stopping=True, return_variance=True)
			variances[i, :len(variance)] = variance
			progress_bar.update(1)

	bad_indices = np.isnan(variances) + np.isinf(variances)
	variances = np.ma.array(variances, mask=bad_indices)
	means = np.mean(variances, axis=0)
	mask = np.ma.getmask(means)
	means[mask] = np.nan
	means = np.array(means)

	if os.path.exists(data_save_path):
		previous_sims = np.load(data_save_path)
		previous_sims = np.vstack([previous_sims, means])
		np.save(data_save_path, previous_sims)

		previous_sims = np.load(sigma_save_path)
		previous_sims = np.append(previous_sims, [weight_sigma_square,], axis=0)
		np.save(sigma_save_path, previous_sims)
	else:
		np.save(sigma_save_path, np.array([weight_sigma_square,]))
		np.save(data_save_path, means)
