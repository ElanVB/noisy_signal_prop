from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.noise import GaussianDropout
from keras.initializers import RandomNormal, Orthogonal

from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm

import keras.backend as K
import numpy as np

# import theano.tensor as _T
import tensorflow as _tf

import os, sys
import pickle

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import noise_layers

def hard_tanh(xx):
    return K.minimum(K.relu(xx + 1) - 1, 1)

def randomize_weight(w, mu=None, sigma=None):
    if mu is None:
        mu = w.mean()
    if sigma is None:
        sigma = w.std()
    return sigma * np.random.randn(*w.shape) + mu

def randomize_weights(weights, bias_sigma=0.1, weight_sigma=1.0, method="xavier"):
    random_weights = []
    for w in weights:
        if w.ndim == 1:
            rw = randomize_weight(w, mu=0, sigma=bias_sigma)
            rw = -rw
        else:
            if method == "xavier":
                sigma = 1 / np.sqrt(w.shape[0])
            elif method == "he":
                sigma = 1 / np.sqrt(w.shape[0]/2)
            elif method == "crit_drop":
                sigma = weight_sigma / np.sqrt(w.shape[0])
            elif method in ["underflow", "overflow", "crit"]:
                sigma = weight_sigma / np.sqrt(w.shape[0])
            else:
                raise NotImplementedError("This initialisation method has not been implemented")

            rw = randomize_weight(w, mu=0, sigma=sigma)
        random_weights.append(rw)
    return random_weights

class RandNet(object):
    """Simple wrapper around Keras model that throws in some useful functions like randomization"""
    def __init__(
        self, input_dim, n_hidden_units, n_hidden_layers, nonlinearity='tanh',
        bias_sigma=0.0, weight_sigma=1.25, input_layer=None, flip=False,
        output_dim=None, dist=None, std=None, diversity=None, prob_1=None,
        _lambda=None, init_method="xavier", model_file=None, Rop_data_file=None
    ):
        if model_file is not None:
            print("loading model from file...")
            self.load(model_file)
        else:
            print("Compiling net...")
            self.input_dim = input_dim
            self.n_hidden_units = n_hidden_units
            self.n_hidden_layers = n_hidden_layers
            self.nonlinearity = nonlinearity
            self.bias_sigma = bias_sigma
            self.weight_sigma = weight_sigma
            self.input_layer = input_layer

            self.dist = dist
            self.rate = prob_1
            self.init_method = init_method

            if output_dim is None:
                output_dim = n_hidden_units
            self.output_dim = output_dim

            self.check_backend()

            noise_mean = np.float64(0.0)

            if std is not None:
                std = np.float64(std)

            if diversity is not None:
                diversity = np.float64(diversity)

            if prob_1 is not None:
                prob_1 = np.float64(prob_1)

            if _lambda is not None:
                _lambda = np.float64(_lambda)

            model = Sequential()
            prev_layer = None

            if input_layer is not None:
                model.add(input_layer)
                prev_layer = input_layer
            else:
                if not flip:
                    raise NotImplementedError("non-flipped model implementation is not complete...")
            for i in range(n_hidden_layers):
                nunits = n_hidden_units if i < n_hidden_layers - 1 else output_dim
                if flip:
                    prev_layer = Activation(nonlinearity, input_shape=(input_dim,), name='a%d'%i)
                    model.add(prev_layer)

                    if dist is not None and dist != "none":
                        noise_mean = K.cast(prev_layer.output, dtype="float64")
                        model.add(noise_layers.Noise(dist, noise_mean, std, noise_mean, diversity, prob_1, _lambda))

                    model.add(Dense(nunits, name='d%d'%i))
                else:
                    raise NotImplementedError("non-flipped model implementation is not complete...")

            model.build()
            self.model = model
            self.weights = model.get_weights()
            self.dense_layers = filter(lambda x:  x.name.startswith('d'), model.layers)
            self.hs = [h.output for h in self.dense_layers]
            self.act_layers = filter(lambda x: x.name.startswith('a'), model.layers)
            self.f_acts = self.f_jac = self.f_jac_hess = self.f_act = None

            vec = K.ones_like(self.model.input)

            if Rop_data_file is not None:
                Rop_data = np.load(Rop_data_file)
                self.Js = Rop_data["Js"]
                self.Hs = Rop_data["Hs"]
            else:
                print("Do Rops...")
                self.Js = [self.Rop(h, self.model.input, vec) for h in tqdm(self.hs, desc="jacobians")]
                self.Hs = [self.Rop(J, self.model.input, vec) for J in tqdm(self.Js, desc="hessians")]

    def save_Rop_data(self, file_name):
        return
        np.savez(file_name, Js=self.Js, Hs=self.Hs)

    def save(self, file_name):
        return
        self.model.save(file_name)
        del self.model

        with open("{}.pkl".format(file_name), "wb") as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

        self.model = load_model(file_name)


    def load(self, file_name):
        return
        with open("{}.pkl".format(file_name), "rb") as input_file:
            self = pickle.load(input_file)

        self.model = load_model(file_name)

    def check_backend(self):
        self.backend = K.backend()
        print("backend is {}".format(self.backend))

        if self.backend == "theano":
            pass
        elif self.backend == "tensorflow":
            pass
        else:
            raise Exception("This backend is not explicitly supported, you will need to add support for it. Specifically by implementing the \"Rop\" function for your backend.")

    def Rop(self, f, x, v):
        if self.backend == "theano":
            return _T.Rop(f, x, v)

        elif self.backend == "tensorflow":
            def Lop(f, x, v):
                return _tf.gradients(f, x, grad_ys=v)

            def Rop(f, x, v):
                w = K.ones_like(f)
                return _tf.gradients(Lop(f, x, w), x, grad_ys=v)

            value = Rop(f, x, v)
            return value[0]
        else:
            raise Exception("This backend is not explicitly supported, you will need to add support for it. Specifically by implementing the \"Rop\" function for your backend.")

    def compile(self, jacobian=False):
        self.f_acts = K.function([self.model.input], self.hs)

    def get_acts(self, xs):
        if self.f_acts is None:
            self.f_acts = K.function([self.model.input], self.hs)
        return self.f_acts((xs,))

    def get_act(self, xs):
        if self.f_act is None:
            self.f_act = K.function([self.model.input], [self.hs[-1]])

        ret = self.f_act((xs,))
        if type(ret) is list and len(ret) == 1:
            ret = ret[0] # such a hack

        return ret

    def get_jacobians(self, xs):
        assert self.model.input_shape[1] == 1
        if self.f_jac is None:
            self.f_jac = K.function([self.model.input], self.Js)
        return self.f_jac((xs,))

    def get_acts_and_derivatives(self, xs, include_hessian=False):
        # one should note that if this function is run with a different value for
        # include_hessian than the first time this function was run, it will not
        # take the new parameter value into account (ie the function is defined
        # the first time it is run)
        assert self.model.input_shape[1] == 1
        if self.f_jac_hess is None:
            if include_hessian:
                self.f_jac_hess = K.function([self.model.input], self.hs + self.Js + self.Hs)
            else:
                self.f_jac_hess = K.function([self.model.input], self.hs + self.Js)
        return self.f_jac_hess((xs,))


    def randomize(self, bias_sigma=None, weight_sigma=None):
        """Randomize the weights and biases in a model.

        Note this overwrites the current weights in the model.
        """

        if bias_sigma is None:
            bias_sigma = self.bias_sigma
        if weight_sigma is None:
            weight_sigma = self.weight_sigma

        self.model.set_weights(
            randomize_weights(
                self.weights, bias_sigma, weight_sigma,
                method=self.init_method
            )
        )

    def randomize_trained(self):
        weights = self.model.get_weights()
        rand_weights = randomize_weights(weights)
        self.model.set_weights(weights)

from keras.layers.core import Layer
class GreatCircle(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        U, _, _ = np.linalg.svd(np.random.randn(output_dim, 2), full_matrices=False)
        self.U = K.variable(U.T)
        self.scale = K.variable(1.0)
        kwargs['input_shape'] = (1, )
        super(GreatCircle, self).__init__(**kwargs)

    def set_scale(self, scale):
        K.set_value(self.scale, np.array(scale).astype(np.float32))

    def call(self, x, mask=None):
        return self.scale * K.dot(K.concatenate((K.cos(x), K.sin(x))), self.U)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

    # a warning asked me to add this and hopefully it solves my problem
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class InterpLine(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.x1 = K.variable(np.random.randn(1, output_dim))
        self.x2 = K.variable(np.random.randn(1, output_dim))
        self.scale = K.variable(1.0)
        kwargs['input_shape'] = (1, )
        super(InterpLine, self).__init__(**kwargs)

    def set_scale(self, scale):
        self.scale.set_value(np.array(scale).astype(np.float32))

    def set_points(self, x1, x2):
        self.x1.set_value(x1[None, :].astype(np.float32))
        self.x2.set_value(x2[None, :].astype(np.float32))

    def call(self, x, mask=None):
        return K.dot(K.cos(x), self.x1) + K.dot(K.sin(x), self.x2)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)


def great_circle(dim, n_interp):
    """Generate circle dataset"""
    ts = np.linspace(0, 2 * np.pi, n_interp, endpoint=False)
    u, _, _ = np.linalg.svd(np.random.randn(dim, 2), full_matrices=False)
    xs = np.dot(u, np.vstack((np.cos(ts), np.sin(ts)))).T
    return ts, xs
