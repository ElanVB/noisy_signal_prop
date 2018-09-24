import sys
from numpy_simulation import noisy_signal_prop_simulations

experiments = [
	{"dist": "bern", "noise": ('prob_1', 0.6), "act":"relu", "init":"underflow"},
	{"dist": "bern", "noise": ('prob_1', 0.6), "act":"relu", "init":"overflow"},
	{"dist": "bern", "noise": ('prob_1', 0.6), "act":"relu", "init":"crit"},
	{"dist": "mult gauss", "noise": ('std', 0.25), "act":"relu", "init":"underflow"},
	{"dist": "mult gauss", "noise": ('std', 0.25), "act":"relu", "init":"overflow"},
	{"dist": "mult gauss", "noise": ('std', 0.25), "act":"relu", "init":"crit"}
]

# experiments = [
#     {"dist": "none", "noise": (None, None), "act":"relu", "init":"crit"},
#     {"dist": "bern", "noise": ('prob_1', 0.6), "act":"relu", "init":"crit"},
#     {"dist": "bern", "noise": ('prob_1', 0.8), "act":"relu", "init":"crit"},
#     {"dist": "mult gauss", "noise": ('std', 0.25), "act":"relu", "init":"crit"},
#     {"dist": "mult gauss", "noise": ('std', 2), "act":"relu", "init":"crit"}
# ]

test_index = int(sys.argv[1])
noisy_signal_prop_simulations(**experiments[test_index])
