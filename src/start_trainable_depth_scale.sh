source activate noisy_signal_prop_test

for dataset_index in {0..1}
do
	for dropout_index in {0..9}
	do
		for depth_index in {0..9}
		do
			python start_trainable_depth_scale.py $dropout_index $depth_index $dataset_index
		done
	done
done