import numpy as np
import tensorflow as tf

def _compute_fans(shape):
	"""Computes the number of input and output units for a weight shape.
		Args:
		shape: Integer shape tuple or TF tensor shape.
		Returns:
		A tuple of scalars (fan_in, fan_out).
	"""
	if len(shape) < 1: # Just to avoid errors for constants.
		fan_in = fan_out = 1
	elif len(shape) == 1:
		fan_in = fan_out = shape[0]
	elif len(shape) == 2:
		fan_in = shape[0]
		fan_out = shape[1]
	else:
		# Assuming convolution kernels (2D, 3D, or more).
		# kernel shape: (..., input_depth, depth)
		receptive_field_size = 1.

		for dim in shape[:-2]:
			receptive_field_size *= dim

		fan_in = shape[-2] * receptive_field_size
		fan_out = shape[-1] * receptive_field_size

	return fan_in, fan_out

class VarInitializer(tf.initializers.variance_scaling):
	def __call__(self, shape, dtype=None, partition_info=None):
		if dtype is None:
			dtype = self.dtype

		scale = self.scale
		scale_shape = shape

		if partition_info is not None:
			scale_shape = partition_info.full_shape

		fan_in, fan_out = _compute_fans(scale_shape)

		if self.mode == "fan_in":
			scale /= max(1., fan_in)
		elif self.mode == "fan_out":
			scale /= max(1., fan_out)
		else:
			scale /= max(1., (fan_in + fan_out) / 2.)

		if self.distribution == "normal":
			stddev = np.sqrt(scale)
			return tf.random_normal(shape, 0.0, stddev, dtype, seed=self.seed)
		else:
			limit = np.sqrt(3.0 * scale)
			return tf.random_uniform(shape, -limit, limit, dtype, seed=self.seed)
