source activate noisy_signal_prop_test

for index in {0..24}
do
	python start_variance_depth_scale.py $index
done