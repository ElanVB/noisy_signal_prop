import sys
import numpy as np

from variance_depth_scale import test_init

inits = np.round(np.linspace(0.1, 2.5, 12), 1)

test_init(inits[int(sys.argv[1])])
