import numpy as np
import mnist, cifar10

from tensorflow_net import Network
from data_iterator import DataIterator

batch_size = 128

def test(depth, p, dataset, num_epochs=200, seed=None):
	if seed is None:
		seed = 0

	np.random.seed(seed)

	data = None
	if dataset == "mnist":
		data = mnist.load().astype(np.float32)
	elif dataset == "cifar10":
		data = cifar10.load().astype(np.float32)

	num_observations, input_dim = data.shape
	data_split_index = int(num_observations * 0.9)
	training_data_iterator = DataIterator(batch_size, data[:data_split_index], data[:data_split_index])
	validation_data_iterator = DataIterator(batch_size, data[data_split_index:], data[data_split_index:])

	# make net
	net = Network(input_dim, input_dim, hidden_layers=([1000,] * depth), p=p)
	losses = net.train(training_data_iterator, validation_data_iterator, num_epochs=num_epochs)
	net.close()

	return losses
