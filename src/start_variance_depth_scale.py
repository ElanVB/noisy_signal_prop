import sys
import numpy as np

from variance_depth_scale import test_init

# any more fine grained and this line MUST CHANGE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
inits = np.linspace(0.1, 2.5, 25)

test_init(inits[int(sys.argv[1])])
