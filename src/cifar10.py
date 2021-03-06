import os
import numpy as np

save_dir="./cifar-10"
_FILENAME = "cifar10.npy"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, _FILENAME)

def download(check_if_present=True):
	if check_if_present:
		already_present = False

		for filename in os.listdir(save_dir):
			if _FILENAME == filename:
				already_present = True
				break

		if not already_present:
			download(check_if_present=False)
		else:
			print("File already present.")
	else:
		import tensorflow as tf

		(train_input, _), (_, _) = tf.keras.datasets.cifar10.load_data()
		train_input = pre_process(np.array(train_input, dtype=np.float64))
		np.save(save_path, train_input.astype(np.float32))

def pre_process(data, unit_length=False):
	data = np.reshape(data, (data.shape[0], -1))

	if unit_length:
		lengths = np.sum(data**2, axis=-1) / data.shape[-1]
		data = data / np.sqrt(lengths)[:, np.newaxis]
	else:
		data /= 255
		lengths = np.sum(data**2, axis=-1) / data.shape[-1]
		np.save(os.path.join(save_dir, "lengths.npy"), lengths)

	np.random.shuffle(data)
	return data

def load():
	try:
		return np.load(save_path)
	except FileNotFoundError:
		download()
		return load()
