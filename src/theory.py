import numpy as np
from scipy.integrate import quad, dblquad
from scipy.stats import norm

epsabs = 1e-2
epsrel = 1e-2
dz = 0.05
zmin = -10.0
zmax = 10.0

def fast_integral(integrand, zmin, zmax, dz, ndim=1):
	zs = np.r_[zmin:zmax:dz]
	if ndim > 1:
		zgrid = np.meshgrid(*((zs,) * ndim))
	else:
		zgrid = (zs,)
	out = integrand(*zgrid)
	return out.sum(tuple(np.arange(ndim))) * dz**ndim


def qmap(qin, weight_sigma=1.0 , bias_sigma=0.0, nonlinearity=np.tanh,
		 epsabs=epsabs, epsrel=epsrel, zmin=-10, zmax=10, dz=dz, fast=True):
	qin = np.atleast_1d(qin)

	def integrand(z):
		return norm.pdf(z[:, None]) * nonlinearity(np.sqrt(qin[None, :]) * z[:, None])**2
	integral = fast_integral(integrand, zmin, zmax, dz=dz)
	return weight_sigma**2 * integral + bias_sigma**2


def qmap_noise(
	qin, weight_sigma=1.0, bias_sigma=0.0, nonlinearity=np.tanh, epsabs=epsabs,
	epsrel=epsrel, zmin=-10, zmax=10, dz=dz, fast=True, dist=None, std=None,
	diversity=None, prob_1=None, _lambda=None
):

	if dist is not None and dist == "none":
		return qmap(
			qin=qin, weight_sigma=weight_sigma, bias_sigma=bias_sigma,
			nonlinearity=nonlinearity, epsabs=epsabs, epsrel=epsrel, zmin=zmin,
			zmax=zmax, dz=dz, fast=fast
		)

	if not dist:
		raise ValueError("Specified distribution is not valid.")

	dist = dist.lower()

	if dist not in ["add gauss", "mult gauss", "lap", "bern", "pois"]:
		raise ValueError("Specified distribution is not valid.")

	qin = np.atleast_1d(qin)
	
	def integrand(z):
		return norm.pdf(z[:, None]) * nonlinearity(np.sqrt(qin[None, :]) * z[:, None])**2
	def integrand_pois(z):
		return norm.pdf(z[:, None]) * nonlinearity(np.sqrt(qin[None, :]) * z[:, None])

	if dist == "add gauss":
		integral = fast_integral(integrand, zmin, zmax, dz=dz)
		if std is not None:
			return weight_sigma**2 * (integral + std**2) + bias_sigma**2
		else:
			raise ValueError("std must be defined for Gaussian noise.")

	if dist == "mult gauss":
		integral = fast_integral(integrand, zmin, zmax, dz=dz)
		if std is not None:
			return weight_sigma**2 * (integral * (std**2+1)) + bias_sigma**2
		else:
			raise ValueError("std must be defined for Gaussian noise.")

	if dist == "lap":
		integral = fast_integral(integrand, zmin, zmax, dz=dz)
		if diversity is not None:
			return weight_sigma**2 * (integral + 2*diversity**2) + bias_sigma**2
		else:
			raise ValueError("diversity must be defined for Laplace noise.")

	if dist == "bern":
		integral = fast_integral(integrand, zmin, zmax, dz=dz)
		if prob_1 is not None:
			return ((weight_sigma**2)/prob_1) * integral + bias_sigma**2
		else:
			raise ValueError("prob_1 must be defined for Bernoulli noise.")

	if dist == "pois":
		integral_pois = fast_integral(integrand_pois, zmin, zmax, dz=dz)
		return weight_sigma**2 * (integral + integral_pois) + bias_sigma**2

def compute_chi1(qstar, weight_sigma=1.0, bias_sigma=0.01, dphi=np.tanh):
	def integrand(z):
		return norm.pdf(z) * dphi(np.sqrt(qstar) * z)**2
	integral = quad(integrand, zmin, zmax, epsabs=epsabs, epsrel=epsrel)[0]
	return weight_sigma**2 * integral

def compute_chi2(qstar, weight_sigma=1.0, bias_sigma=0.01, d2phi=np.tanh, hidden_units=100):
	def integrand(z):
		return norm.pdf(z) * d2phi(np.sqrt(qstar) * z)**2
	integral = quad(integrand, zmin, zmax, epsabs=epsabs, epsrel=epsrel)[0]
	return  weight_sigma**2 * integral  / hidden_units 

def kappa_map(kappa, chi1, chi2):
	return 3 * chi2 / chi1**2 + 1/chi1 * kappa

def covmap(q1, q2, q12, weight_sigma, bias_sigma, nonlinearity=np.tanh, zmin=-10, zmax=10, dz=dz, fast=True):
	# NB: this only works for critically initialized networks!
	return (weight_sigma**2)/2 * (
		(q12/np.pi) * np.arcsin(q12) +
		np.sqrt(1 - q12**2)/np.pi +
		0.5 * q12
	)

def q_fixed_point(weight_sigma, bias_sigma, nonlinearity, max_iter=500, tol=1e-9, qinit=3.0, fast=True, tol_frac=0.01):
	"""Compute fixed point of q map"""
	q = qinit
	qs = []
	for i in range(max_iter):
		qnew = qmap(q, weight_sigma, bias_sigma, nonlinearity, fast=fast)
		err = np.abs(qnew - q)
		qs.append(q)
		if err < tol:
			break
		q = qnew
	# Find first time it gets within tol_frac fracitonal error of q*
	frac_err = (np.array(qs) - q)**2 / (1e-9 + q**2)
	try:
		t = np.flatnonzero(frac_err < tol_frac)[0]
	except IndexError:
		t = max_iter
	return t, q


def q_fixed_point_noise(
	weight_sigma, bias_sigma, nonlinearity, max_iter=500, tol=1e-9, qinit=3.0,
	fast=True, tol_frac=0.01, dist=None, std=None, diversity=None, prob_1=None,
	_lambda=None
):
	if dist is not None or dist == "none":
		return q_fixed_point(
			weight_sigma=weight_sigma, bias_sigma=bias_sigma,
			nonlinearity=nonlinearity, max_iter=max_iter, tol=tol, qinit=qinit,
			fast=fast, tol_frac=tol_frac
		)

	"""Compute fixed point of q map"""
	q = qinit
	qs = []
	for i in range(max_iter):
		qnew = qmap_noise(
			q, weight_sigma, bias_sigma, nonlinearity, fast=fast, dist=dist,
			std=std, diversity=diversity, prob_1=prob_1, _lambda=_lambda
		)
		err = np.abs(qnew - q)
		qs.append(q)
		if err < tol:
			break
		q = qnew
	# Find first time it gets within tol_frac fracitonal error of q*
	frac_err = (np.array(qs) - q)**2 / (1e-9 + q**2)
	try:
		t = np.flatnonzero(frac_err < tol_frac)[0]
	except IndexError:
		t = max_iter
	return t, q

def mu(dist, noise_param):
    if isinstance(dist, str):
        dist = dist.lower()

        if "gauss" in dist:
            return noise_param + 1
        elif "lap" in dist:
            return noise_param + 1 # be careful! 2*beta = noise_param (the variance)
        elif "pois" in dist:
            return 2
        elif "drop" in dist or "bern" in dist:
            return 1 / noise_param
    else:
        raise TypeError("dist must be a string")

def gamma(dist, sigma, noise_param):
    _mu = mu(dist, noise_param)
    return sigma * _mu / 2

def depth(dist, sigma, noise_param=None, q_0=1):
    growth_rate = gamma(dist, np.array(sigma), noise_param)

    if isinstance(growth_rate, float):
        if growth_rate < 1:
            value = np.finfo("float32").tiny
        else:
            value = np.finfo("float32").max

        return (np.log10(value) - np.log10(q_0))/np.log10(2*p/sigma)
    elif isinstance(growth_rate, np.ndarray):
        explode_value = np.finfo("float32").max
        shrink_value = np.finfo("float32").tiny

        ret_val = np.empty(growth_rate.shape)

        exploding_ps = growth_rate >= 1
        ret_val[exploding_ps] = (np.log10(explode_value) - np.log10(q_0))/np.log10(growth_rate[exploding_ps])

        shrinking_ps = growth_rate < 1
        ret_val[shrinking_ps] = (np.log10(shrink_value) - np.log10(q_0))/np.log10(growth_rate[shrinking_ps])

        return ret_val
    else:
        raise ValueError("growth rate of type {} not supported, check that you have valid values for p and sigma".format(type(growth_rate)))


def critical_point(dist, noise_param):
    if isinstance(dist, str):
        dist = dist.lower()

        if "gauss" in dist:
            return 2 / (noise_param + 1)
        elif "lap" in dist:
            return 2 / (noise_param + 1)
        elif "pois" in dist:
            return 1
        elif "drop" in dist or "bern" in dist:
            return 2 * noise_param
    else:
        raise TypeError("dist must be a string")

def fixed_point(f, guess, sigma=1, mu2=1, epsilon=10**(-8), n=1000):
    itr=0
    fp = 0
    test=f(guess, sigma, mu2)
    if (abs(test-guess)<epsilon):
        fp = guess
    while ((n>itr) and (abs(test-guess)>=epsilon)):
        itr+=1
        guess = test
        test = f(test, sigma, mu2)
        if ((abs(test-guess))<epsilon):
            fp = guess
    return fp

def c_map(x, sigma=1, mu2=1):
    return (sigma/(2*np.pi))*(x*np.arcsin(x) + np.sqrt(1-x**2)) + (sigma/4)*x + 1 - (sigma/2)*mu2

def c_map_slope(x, sigma):
    return (sigma/(2*np.pi))*np.arcsin(x) + sigma/4

def depth_scale(xi):
    return -1/(np.log(xi))