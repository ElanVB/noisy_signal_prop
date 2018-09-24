import numpy as np

class Noise():
	def __init__(
		self, dist=None, mean=None, std=None, p=None
	):
		if not dist:
			raise ValueError("Noise layer's distribution is not valid.")

		dist = dist.lower()

		if dist not in ["add gauss", "mult gauss", "bern"]:
			raise ValueError("Noise layer's distribution is not valid.")

		if dist in ["mult gauss", "add gauss"]:
			if (mean is None) or (std is None):
				raise ValueError("mean and std must both be defined for Gaussian noise layer.")
		elif dist == "bern":
			if p is None:
				raise ValueError("p must be defined for Bernoulli noise layer.")

		self.mean = mean
		self.std = std
		self.p = p

	def apply_to(self, x):
		if self.dist == "bern":
			sample = (1/self.p) * np.random.binomial(n=1, p=self.p, size=x.shape)

		elif "gauss" in self.dist:
			sample = np.random.normal(loc=self.mean, scale=self.std, size=x.shape)

			if "add" in self.dist:
				return x + sample

		return x * sample
