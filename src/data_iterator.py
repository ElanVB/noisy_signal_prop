import numpy as np

class Container():
	"""
	A simple container for a single batch of input and output data.
	"""
	def __init__(self, input_data, output_data):
		self.input = input_data
		self.output = output_data

class DataIterator():
	"""
	An iterable object that feeds a dataset in batches. Typical use case is for
	stochastic gradient decent.
	"""
	def __init__(self, batch_size, inputs, outputs):
		self.input_data = np.array(inputs)
		self.output_data = np.array(outputs)
		self.batch_size = batch_size
		self.max_index = self.input_data.shape[0]
		self.num_batches = np.ceil(self.max_index / self.batch_size).astype(int)

	def next_batch(self):
		next_index = self.current_index + self.batch_size
		input_batch = self.input_data[self.current_index : next_index]
		output_batch = self.output_data[self.current_index : next_index]
		self.current_index = next_index

		return Container(input_batch, output_batch)

	def __iter__(self):
		self.current_index = 0
		while self.current_index < self.max_index:
			yield self.next_batch()

	def size(self):
		return self.num_batches
