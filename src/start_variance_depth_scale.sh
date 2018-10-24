source activate noisy_signal_prop_test

for index in {0..11}
do
	python start_variance_depth_scale.py $index
done