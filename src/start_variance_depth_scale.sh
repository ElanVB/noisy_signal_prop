source activate noisy_signal_prop_test

for index in {0..23}
do
	python start_variance_depth_scale.py $index
done