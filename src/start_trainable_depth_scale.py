import sys, os
import numpy as np

from trainable_depth_scale import test

dropout_index = int(sys.argv[1])
depth_index = int(sys.argv[2])
dataset_index = int(sys.argv[3])

num_dropouts = 10
num_depths = 10
num_epochs = 200

dropouts = np.linspace(0.1, 1, num_dropouts)
depths = np.linspace(2, 40, num_depths, dtype=int)
datasets = ["mnist", "cifar10"]

dropout_value = dropouts[dropout_index]
depth_value = depths[depth_index]
dataset = datasets[dataset_index]
seed = num_dropouts * dropout_index + depth_index

file_dir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(file_dir, "results/trainable_depth/{}".format(dataset))
data_save_path = os.path.join(save_dir, "trainable_depth.npy")
p_depth_save_path = os.path.join(save_dir, "trainable_depth_p_depth.npy")

os.makedirs(save_dir, exist_ok=True)

if not os.path.exists(data_save_path):
	data_save_file = np.empty((num_dropouts, num_depths, num_epochs, 2), dtype=np.float16)
	p_depth_save_file = np.empty((num_dropouts, num_depths, 2), dtype=np.float16)

	np.save(data_save_path, data_save_file)
	np.save(p_depth_save_path, p_depth_save_file)

print("Testing p = {}, depth = {}, seed = {} for {} epochs on {}".format(dropout_value, depth_value, seed, num_epochs, dataset))
losses = test(depths[depth_index], dropouts[dropout_index], dataset=dataset, num_epochs=num_epochs, seed=seed)

prev_data = np.load(data_save_path)
prev_data[dropout_index, depth_index] = losses
np.save(data_save_path, prev_data)

prev_p_depth = np.load(p_depth_save_path)
prev_p_depth[dropout_index, depth_index] = np.array([dropout_value, depth_value])
np.save(p_depth_save_path, prev_p_depth)
