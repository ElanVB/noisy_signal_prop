import sys
from numpy_simulation import noisy_signal_prop_simulations

experiments = [
	{"dist": "none", "noise": (None, None), "act":"relu", "init":"crit"},
	{"dist": "bern", "noise": ('prob_1', 0.6), "act":"relu", "init":"crit"},
	{"dist": "bern", "noise": ('prob_1', 0.8), "act":"relu", "init":"crit"},
	{"dist": "mult gauss", "noise": ('std', 0.25), "act":"relu", "init":"crit"},
	{"dist": "mult gauss", "noise": ('std', 2), "act":"relu", "init":"crit"},
	{"dist": "bern", "noise": ('prob_1', 0.6), "act":"relu", "init":"underflow"},
	{"dist": "bern", "noise": ('prob_1', 0.6), "act":"relu", "init":"overflow"},
	{"dist": "bern", "noise": ('prob_1', 0.6), "act":"relu", "init":"crit"},
	{"dist": "bern", "noise": ('prob_1', 0.1), "act":"relu", "init":"crit"},
	{"dist": "bern", "noise": ('prob_1', 0.2), "act":"relu", "init":"crit"},
	{"dist": "bern", "noise": ('prob_1', 0.3), "act":"relu", "init":"crit"},
	{"dist": "bern", "noise": ('prob_1', 0.4), "act":"relu", "init":"crit"},
	{"dist": "bern", "noise": ('prob_1', 0.5), "act":"relu", "init":"crit"},
	{"dist": "bern", "noise": ('prob_1', 0.7), "act":"relu", "init":"crit"},
	{"dist": "bern", "noise": ('prob_1', 0.8), "act":"relu", "init":"crit"},
	{"dist": "bern", "noise": ('prob_1', 0.9), "act":"relu", "init":"crit"},
	{"dist": "mult gauss", "noise": ('std', 0.25), "act":"relu", "init":"underflow"},
	{"dist": "mult gauss", "noise": ('std', 0.25), "act":"relu", "init":"overflow"},
	{"dist": "mult gauss", "noise": ('std', 0.25), "act":"relu", "init":"crit"},
	{"dist": "mult gauss", "noise": ('std', 0.1), "act":"relu", "init":"crit"},
	{"dist": "mult gauss", "noise": ('std', 0.4), "act":"relu", "init":"crit"},
	{"dist": "mult gauss", "noise": ('std', 0.55), "act":"relu", "init":"crit"},
	{"dist": "mult gauss", "noise": ('std', 0.7), "act":"relu", "init":"crit"},
	{"dist": "mult gauss", "noise": ('std', 0.85), "act":"relu", "init":"crit"},
	{"dist": "mult gauss", "noise": ('std', 1.0), "act":"relu", "init":"crit"},
	{"dist": "mult gauss", "noise": ('std', 1.15), "act":"relu", "init":"crit"},
	{"dist": "mult gauss", "noise": ('std', 1.3), "act":"relu", "init":"crit"},
	{"dist": "mult gauss", "noise": ('std', 1.45), "act":"relu", "init":"crit"},
	{"dist": "mult gauss", "noise": ('std', 1.6), "act":"relu", "init":"crit"},
	{"dist": "mult gauss", "noise": ('std', 1.75), "act":"relu", "init":"crit"},
	{"dist": "mult gauss", "noise": ('std', 1.9), "act":"relu", "init":"crit"}
]

# experiments = [
# 	{"dist": "none", "noise": (None, None), "act":"relu", "init":"crit"},
# 	{"dist": "bern", "noise": ('prob_1', 0.6), "act":"relu", "init":"crit"},
# 	{"dist": "bern", "noise": ('prob_1', 0.8), "act":"relu", "init":"crit"},
# 	{"dist": "mult gauss", "noise": ('std', 0.25), "act":"relu", "init":"crit"},
# 	{"dist": "mult gauss", "noise": ('std', 2), "act":"relu", "init":"crit"}
# ]

# experiments = [
#     {"dist": "bern", "noise": ('prob_1', 0.6), "act":"relu", "init":"underflow"},
#     {"dist": "bern", "noise": ('prob_1', 0.6), "act":"relu", "init":"overflow"},
#     {"dist": "bern", "noise": ('prob_1', 0.6), "act":"relu", "init":"crit"},
#     {"dist": "mult gauss", "noise": ('std', 0.25), "act":"relu", "init":"underflow"},
#     {"dist": "mult gauss", "noise": ('std', 0.25), "act":"relu", "init":"overflow"},
#     {"dist": "mult gauss", "noise": ('std', 0.25), "act":"relu", "init":"crit"}
# ]

test_index = int(sys.argv[1])
noisy_signal_prop_simulations(**experiments[test_index])
