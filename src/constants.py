# Hardcoded constants used throughout the paper
import numpy as np
from scipy.stats import hypsecant

nq = 100
nw = 41
nb = 41
qmax = 15
qrange = np.linspace(0, qmax, nq)
qidx = [26, 53, 80]
n_hidden_layers = 20
bias_sigmas = [0]
widxs = [0] 
bidxs = [0] 

def relu(x):
	# one of the fastest ways of applying the relu function
	return x * (x > 0)

# functions needed for First and second derivatives of tanh, needed for curvature computations
def sech(x):
    return hypsecant.pdf(x) * np.pi

nonlinearities = {
	"relu": relu,
	"tanh": np.tanh
}

dphis = {
	"relu": lambda x: (x > 0), # derivative of relu
	"tanh": lambda x: sech(x)**2 # derivative of tanh
}

d2phis = {
	"relu": 0, # second derivative of relu
	"tanh": lambda x: 2 * (np.tanh(x)**3 - np.tanh(x)) # second derivative of tanh
}

# Ranges for weight and bias standard deviations

#wmax = 5
#bmax = 4
#weight_sigmas = 1 #np.linspace(1, wmax, nw)
#bias_sigmas = 0 #np.linspace(0, bmax, nb)

# Chosen indices for plotting.
# widxs = [3] # new figure 2
# widxs = [15]
# widxs = [3, 15] # new figure 1
#widxs = [3, 15, 30] # all data pertaining to single noise sigma
#bidxs = [3] * len(widxs)

#weight_sigmas = [np.sqrt(2)] #weight_sigmas[widxs]
 #bias_sigmas[bidxs][:1]
#nw = 1 #len(weight_sigmas)
#nb = 1 #len(bias_sigmas)


