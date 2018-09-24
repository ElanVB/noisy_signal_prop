import tensorflow as tf
import keras.backend as K
from keras.layers import Lambda
from keras.engine.topology import Layer

class Noise(Layer):
	def __init__(
		self, dist=None, mean=None, std=None, location=None,
		diversity=None, prob_1=None, _lambda=None, **kwargs
	):
		if not dist:
			raise ValueError("Noise layer's distribution is not valid.")

		dist = dist.lower()

		if dist not in ["add gauss", "mult gauss", "lap", "bern", "pois"]:
			raise ValueError("Noise layer's distribution is not valid.")

		if dist == "mult gauss":
			if (mean is not None) and (std is not None):
				self.tf_mean = tf.cast(1, tf.float64)
				self.tf_std = tf.cast(std, tf.float64)
			else:
				raise ValueError("mean and std must both be defined for Gaussian noise layer.")
		if dist == "add gauss":
			if (mean is not None) and (std is not None):
				self.tf_mean = tf.cast(mean, tf.float64)
				self.tf_std = tf.cast(std, tf.float64)
			else:
				raise ValueError("mean and std must both be defined for Gaussian noise layer.")
		elif dist == "lap":
			if (location is not None) and (diversity is not None):
				self.tf_location = tf.cast(location, tf.float64)
				self.tf_diversity = tf.cast(diversity, tf.float64)
			else:
				raise ValueError("location and diversity must both be defined for Laplace noise layer.")
		elif dist == "bern":
			if prob_1 is not None:
				self.tf_prob_1 = tf.cast(prob_1, tf.float64)
			else:
				raise ValueError("prob_1 must be defined for Bernoulli noise layer.")
		elif dist == "pois":
			if _lambda is not None:
				self.tf_lambda = tf.cast(_lambda, tf.float64)
			else:
				raise ValueError("_lambda must be defined for Poisson noise layer.")

		self.mean = mean
		self.std = std
		self.location = location
		self.diversity = diversity
		self.prob_1 = prob_1
		self._lambda = _lambda

		self.dist_str = dist

		super(Noise, self).__init__(**kwargs)

	def get_config(self):
		return {
			"dist": self.dist_str,
			"mean": self.mean,
			"std": self.std,
			"location": self.location,
			"diversity": self.diversity,
			"prob_1": self.prob_1,
			"_lambda": self._lambda
		}

	def build(self, input_shape):
		if "gauss" in self.dist_str:
			self.dist = tf.distributions.Normal(loc=self.tf_mean, scale=self.tf_std)
		elif self.dist_str == "lap":
			self.dist = tf.distributions.Laplace(loc=self.tf_location, scale=self.tf_diversity)
		elif self.dist_str == "bern":
			self.scale = 1 / self.tf_prob_1
			self.dist = tf.distributions.Bernoulli(probs=self.tf_prob_1)
		elif self.dist_str == "pois":
			self.dist = tf.contrib.distributions.Poisson(rate=self.tf_mean)

		super(Noise, self).build(input_shape)

	def call(self, x):
		if self.dist_str == "bern":
			sample = K.cast(self.dist.sample(tf.shape(x)), x.dtype)
			self.scale = K.cast(self.scale, x.dtype)
			return self.scale * sample * x
		elif self.dist_str == "mult gauss":
			sample = K.cast(self.dist.sample(tf.shape(x)), x.dtype)
			return sample * x
		else:
			return K.cast(self.dist.sample(), x.dtype)

	def compute_output_shape(self, input_shape):
		return input_shape
