import numpy as np
# from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm

class Network():
	def __init__(
		self, input_dim, output_dim, hidden_layer_sizes, activation,
		weight_sigma, bias_sigma=0, dist=None, mean=None, std=None, p=None,
		init_method=None
	):
		# find a more elegant solution later
		if (mean is None) and ("gauss" in dist):
			if "add" in dist:
				mean = 0
			elif "mult" in dist:
				mean = 1

		self.input_dim = input_dim
		self.output_dim = output_dim
		self.hidden_layer_sizes = hidden_layer_sizes
		self.activation = activation
		self.weight_sigma = weight_sigma
		self.bias_sigma = bias_sigma

		self._DTYPE = np.float32

		if not dist:
			raise ValueError("Noise layer's distribution is not valid.")

		dist = dist.lower()

		if dist not in ["add gauss", "mult gauss", "bern", "none", None]:
			raise ValueError("Noise layer's distribution is not valid.")

		if dist in ["mult gauss", "add gauss"]:
			if (mean is None) or (std is None):
				raise ValueError("mean and std must both be defined for Gaussian noise layer.")
		elif dist == "bern":
			if p is None:
				raise ValueError("p must be defined for Bernoulli noise layer.")

		self.dist = dist
		self.mean = mean
		self.std = std
		self.p = p
		self.init_method = init_method

		self._check_activation()
		self._create_network()

	def _ReLU(self, x):
		return x * (x > 0)

	def _tanh(self, x):
		return np.tanh(x)

	def _linear(self, x):
		return x

	def _check_activation(self):
		if isinstance(self.activation, str):
			# convert the string "relu" to a callable function (that applies ReLU)
			if self.activation.lower() == "relu":
				# print("Setting activation to ReLU...")
				self.activation = self._ReLU
			# same for "tanh"
			elif self.activation.lower() == "tanh":
				self.activation = self._tanh
			else:
				raise ValueError(
					"'relu' and 'tanh' are the only the strings supported for the activation argument."
				)
		elif not callable(self.activation):
			raise TypeError("You must either provide a function or a string to the activation argument.")

	def _create_network(self):
		self.weight_shapes = []

		if len(self.hidden_layer_sizes) > 0:
			# add weights for input layer
			self.weight_shapes.append((self.input_dim, self.hidden_layer_sizes[0]))

			# add hidden layers
			for i, size in enumerate(self.hidden_layer_sizes[:-1]):
				self.weight_shapes.append((size, self.hidden_layer_sizes[i+1]))

			# # add weights for output layer
			# self.weight_shapes.append((self.hidden_layer_sizes[-1], self.output_dim))
		else:
			# add single shape
			self.weight_shapes.append((self.input_dim, self.output_dim))

		self._initialize_network()
		self._initialize_noise()

	def _check_weight_sigma(self):
		if self.init_method is not None:
			if isinstance(self.init_method, str):
				# set weight sigma according to known initialization schemes
				self.init_method = self.init_method.lower()
				if self.init_method == "xavier":
					self.weight_sigma = 1
				elif self.init_method == "he":
					self.weight_sigma = np.sqrt(2)
				elif "crit" in self.init_method:
					# self.weight_sigma = 2 * self.p
					pass
				else:
					print(
						"'{}' not a supported initialization method, will use weight_sigma as is.".format(self.init_method)
					)
					# raise ValueError(
					# 	"The only supported strings for the init_method argument are 'xavier', 'he' and 'critical'."
					# )
			else:
				raise TypeError("The init_method argument must be a string.")

	def initialize_weights(self):
		# check if weights have already been initialized
		try:
			# free the allocated memory if they exist
			del self.weight
		except:
			pass
		finally:
			self.weight = []

		# same for bias'
		try:
			del self.bias
		except:
			pass

		# initialize weights
		# print("initializing weights...")
		for shape in self.weight_shapes:
			sigma = self.weight_sigma / np.sqrt(shape[0])
			self.weight.append(np.random.normal(
				loc=0, scale=sigma, size=shape
			).astype(self._DTYPE))

		# initialize bias' if they will be non-zero
		if self.bias_sigma != 0:
			# print("initializing bias'...")
			self.bias = []

			for shape in self.weight_shapes:
				self.bias.append(np.random.normal(
					loc=0, scale=self.bias_sigma, size=shape[-1]
				).astype(self._DTYPE))
		# else:
			# print("No biases needed...")

	def _initialize_network(self):
		self._check_weight_sigma()
		self.initialize_weights()

	def _bern_noise(self, x):
		sample = (1/self.p) * np.random.binomial(
			n=1, p=self.p, size=x.shape
		).astype(self._DTYPE)
		return x * sample

	def _mult_gauss_noise(self, x):
		sample = np.random.normal(
			loc=self.mean, scale=self.std, size=x.shape
		).astype(self._DTYPE)
		return x * sample

	def _add_gauss_noise(self, x):
		sample = np.random.normal(
			loc=self.mean, scale=self.std, size=x.shape
		).astype(self._DTYPE)
		return x + sample

	def _initialize_noise(self):
		if self.dist == "bern":
			self.noise = self._bern_noise
		elif "gauss" in self.dist:
			if "mult" in self.dist:
				self.noise = self._mult_gauss_noise
			elif "add" in self.dist:
				self.noise = self._add_gauss_noise
			else:
				raise ValueError(
					"Must define whether you want multiplicative or additive gaussian noise."
				)
		elif (self.dist is None) or (self.dist == "none"):
			self.noise = self._linear
		else:
			raise ValueError("Given noise distribution is invalid.")

	def get_acts(self, x, early_stopping=False, return_variance=False):
		# don't apply noise on last layer?
		pre_activations = []

		activation = np.array(x).astype(self._DTYPE)
		if self.bias_sigma == 0:
			# Our layer architecture expects noise to be added at the input to the layer
			# activation = self.noise(self.activation(activation))
			# that goes into network
			# remember linear layer at end?
			for w in self.weight:
				# flipped layer topology
				activation = self.noise(self.activation(activation))
				activation = np.matmul(activation, w)

				if return_variance:
					variance = np.mean(np.sum(activation**2, axis=-1) / activation.shape[-1], axis=0)
					pre_activations.append(variance)
				else:
					pre_activations.append(activation)

				# pre_activation = np.matmul(activation, w)
				# activation = self.noise(self.activation(pre_activation))
				# if return_variance:
				# 	variance = np.mean(np.sum(pre_activation**2, axis=-1) / pre_activation.shape[-1], axis=0)
				# 	pre_activations.append(variance)
				# else:
				# 	pre_activations.append(pre_activation)

				if early_stopping and (np.sum(np.isnan(variance) + np.isinf(variance)) > 0):
					break
		else:
			for i, w in enumerate(self.weight):
				################################################################
				# NB!!!! CHECK THIS!!!!!!!!!!
				################################################################
				pre_activation = np.matmul(activation, w) + self.bias[i]
				pre_activations.append(pre_activation)
				activation = self.activation(self.noise(pre_activation))

		return pre_activations
