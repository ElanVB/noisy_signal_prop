import sys
import numpy as np
import tensorflow as tf

# from tqdm import tqdm
from tensorflow_var_init import VarInitializer

class Network():
	"""
	A tensorflow implementation of a neural network.
	"""
	def __init__(self, input_dim, output_dim, hidden_layers=([1000,] * 1), initialization="crit", p=1.0, learning_rate=1e-3, load=False, net_type="reg", act="relu", optimizer="SGD", seed=None):
		self.verbose = False
		self.TF_DTYPE = tf.float32

		self.input_dim = input_dim
		self.output_dim = output_dim
		self.hidden_layers = hidden_layers
		self.learning_rate = learning_rate
		self.type = net_type
		self.initialization = initialization
		self.p = p

		tf.reset_default_graph()

		if seed is None:
			seed = 0

		tf.set_random_seed(seed)

		# determine which activation function to use
		if act == "relu":
			self.activation = tf.nn.relu
		else:
			raise ValueError("'{}' not a valid activation!".format(act))

		self.keep_prob = tf.placeholder(dtype=self.TF_DTYPE, name="keep_prob")

		self.net = self.build_network()

		# placeholder for the training labels
		self.target = tf.placeholder(
			dtype=self.TF_DTYPE, shape=(None, self.output_dim), name="target"
		)

		# define loss function
		if net_type == "reg":
			self.loss = tf.losses.mean_squared_error(self.target, self.net)
		elif net_type == "class":
			self.loss = tf.losses.softmax_cross_entropy(self.target, self.net)

		# define optimizer
		if optimizer == "SGD":
			self.optimizer = tf.train.GradientDescentOptimizer
		elif optimizer == "Adam":
			self.optimizer = tf.train.AdamOptimizer
		elif optimizer == "Nadam":
			self.optimizer = tf.contrib.opt.NadamOptimizer

		if learning_rate is None:
			self.update_weights = self.optimizer(
				name=optimizer
			).minimize(self.loss)
		else:
			self.update_weights = self.optimizer(
				learning_rate, name=optimizer
			).minimize(self.loss)

		self.session = tf.Session()

		# either load an existing model or initialize the weights
		if load:
			self.load()
		else:
			# init model
			self.init = tf.global_variables_initializer()
			self.session.run(self.init)

	def close(self):
		self.session.close()

	def linear(self, x):
		return x

	def layer(self, prev_layer, output_dim):
		"""
		Append a single layer to the network (prev_layer).
		"""
		input_dim = prev_layer.get_shape().as_list()[-1]

		W = tf.get_variable(
			"W", [input_dim, output_dim], dtype=self.TF_DTYPE,
			initializer=self.initializer
		)
		b = tf.get_variable(
			"b", [output_dim], dtype=self.TF_DTYPE,
			initializer=tf.initializers.zeros()
		)

		return tf.matmul(prev_layer, W) + b

	def build_network(self):
		"""
		Construct the whole network.
		"""
		# add input layer
		self.input_layer = tf.placeholder(self.TF_DTYPE, shape=(None, self.input_dim), name="input_layer")
		layer = self.input_layer

		if self.verbose:
			print(
				"Added input layer with shape {}"
				.format(
					layer.get_shape().as_list()
				)
			)

		if "crit" in self.initialization:
			self.initializer = VarInitializer(2 * self.p, distribution="normal")
		elif self.initialization == "he":
			self.initializer = VarInitializer(2, distribution="normal")
		elif self.initialization == "xavier":
			self.initializer = VarInitializer(1, distribution="normal")
		else:
			raise ValueError("{} not value for initialization".format(self.initialization))

		# add hidden layers
		for layer_index, layer_dim in enumerate(self.hidden_layers):
			with tf.variable_scope("layer_{}".format(layer_index)):
				# append a new layer and return the output of this new layer
				layer = tf.nn.dropout(self.activation(layer), keep_prob=self.keep_prob)
				layer = self.layer(layer, layer_dim)

				if self.verbose:
					print(
						"Added layer {} with shape {}"
						.format(
							layer_index,
							layer.get_shape().as_list()
						)
					)

		# add output layer
		with tf.variable_scope("output_layer"):
			layer = tf.nn.dropout(self.activation(layer), keep_prob=self.keep_prob)
			layer = self.layer(layer, self.output_dim)

			if self.type == "reg":
				layer = tf.identity(layer, name="output_layer")
			elif self.type == "class":
				layer = tf.nn.softmax(layer, name="output_layer")

		if self.verbose:
			print(
				"Added output layer with shape {}".format(
					layer.get_shape().as_list()
				)
			)

		return layer

	def save(self):
		saver = tf.train.Saver(tf.global_variables())
		saver.save(self.session, "./model")

		if self.verbose:
			print("model saved")

	def load(self):
		saver = tf.train.Saver()
		saver.restore(self.session, "./model")

		if self.verbose:
			print("model restored")

	def train(self, train_data, val_data, num_epochs=1):
		"""
		Perform stochastic gradient decent on train_data and record the loss
		over time.
		"""
		statistics = np.empty((num_epochs, 2), dtype=np.float16)

		num_train_batches = train_data.size()
		num_val_batches = val_data.size()
		train_losses = np.empty((num_train_batches, ), dtype=np.float16)
		val_losses = np.empty((num_val_batches, ), dtype=np.float16)
		total_batches = num_train_batches + num_val_batches

		for epoch_index in np.arange(num_epochs):
		# for epoch_index in tqdm(np.arange(num_epochs), desc="epochs"):

			# run once through the training data and store loss
			# with tqdm(total=total_batches, desc="batches") as progress_bar:
			for batch_index, batch in enumerate(train_data):
				_, train_loss, _ = self.session.run(
					[self.update_weights, self.loss, self.net], feed_dict={
						self.input_layer: batch.input,
						self.target: batch.output,
						self.keep_prob: self.p
					}
				)
				train_losses[batch_index] = train_loss

				# progress_bar.update(1)
				# progress_bar.set_description("batches - train loss: {: 3.4f}".format(train_loss))

			for batch_index, batch in enumerate(val_data):
				val_loss, _ = self.session.run(
					[self.loss, self.net], feed_dict={
						self.input_layer: batch.input,
						self.target: batch.output,
						self.keep_prob: 1.0
					}
				)
				val_losses[batch_index] = val_loss

				# progress_bar.update(1)
				# progress_bar.set_description("batches - val loss: {: 3.4f}".format(val_loss))

			# store the training and val loss
			statistics[epoch_index, 0] = np.mean(train_losses)
			statistics[epoch_index, 1] = np.mean(val_losses)

		print()

		return statistics

	def predict(self, input_data):
		"""
		Perform a forward pass with the network to acquire a action value
		prediction.
		"""
		prediction = None

		prediction = self.session.run(
			self.net, feed_dict={
				self.input_layer: input_data, self.keep_prob: 1.0
			}
		)

		if self.type == "class":
			prediction = np.argmax(prediction, axis=-1)

		return prediction
