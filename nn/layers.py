""" Code from https://github.com/Eyyub/numpy-convnet """
from abc import ABCMeta, abstractmethod
import numpy as np

class AbstractLayer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError()

    @abstractmethod
    def train_forward(self, inputs):
        raise NotImplementedError()

    @abstractmethod
    def get_layer_error(self, z, backwarded_err):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, layer_err):
        raise NotImplementedError()

    @abstractmethod
    def get_grad(self, inputs, layer_err):
        raise NotImplementedError()

    @abstractmethod
    def update(self, grad):
        raise NotImplementedError()

class FullyConnected(AbstractLayer):
    def __init__(self, wshape, activation, weight_init):
        self.wshape = wshape
        self.W = weight_init(self.wshape)
        self.activation = activation

    def forward(self, inputs):
        return self.activation.compute(inputs.dot(self.W))

    def train_forward(self, inputs):
        z = inputs.dot(self.W)
        return (z, self.activation.compute(z))

    def get_layer_error(self, z, backwarded_err):
        return backwarded_err * self.activation.deriv(z)

    def backward(self, layer_err):
        return layer_err.dot(self.W.T)

    def get_grad(self, inputs, layer_err):
        return inputs.T.dot(layer_err)

    def update(self, grad):
        self.W -= grad
        
class Flatten(AbstractLayer):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, inputs):
        return np.reshape(inputs, (inputs.shape[0], -1))

    def train_forward(self, inputs):
        z = np.reshape(inputs, (inputs.shape[0], -1))
        return (z, z)

    def get_layer_error(self, z, next_layer_err):
        return next_layer_err

    def backward(self, layer_err):
        return np.reshape(layer_err, (layer_err.shape[0],) + self.shape)

    def get_grad(self, inputs, layer_err):
        return 0.

    def update(self, grad):
        pass

class Conv(AbstractLayer):
    def __init__(self, fshape, activation, filter_init, strides=1):
        self.fshape = fshape
        self.strides = strides
        self.filters = filter_init(self.fshape)
        self.activation = activation

    def forward(self, inputs):
        s = (inputs.shape[1] - self.fshape[0]) // self.strides + 1
        fmap = np.zeros((inputs.shape[0], s, s, self.fshape[-1]))
        for j in range(s):
            for i in range(s):
                fmap[:, j, i, :] = np.sum(inputs[:, j * self.strides:j * self.strides + self.fshape[0], i * self.strides:i * self.strides + self.fshape[1], :, np.newaxis] * self.filters, axis=(1, 2, 3))
        return self.activation.compute(fmap)

    def train_forward(self, inputs):
        s = (inputs.shape[1] - self.fshape[0]) // self.strides + 1
        fmap = np.zeros((inputs.shape[0], s, s, self.fshape[-1]))
        for j in range(s):
            for i in range(s):
                fmap[:, j, i, :] = np.sum(inputs[:, j * self.strides:j * self.strides + self.fshape[0], i * self.strides:i * self.strides + self.fshape[1], :, np.newaxis] * self.filters, axis=(1, 2, 3))
        return (fmap, self.activation.compute(fmap))

    def get_layer_error(self, z, backwarded_err):
        return backwarded_err * self.activation.deriv(z)

    def backward(self, layer_err):
        bfmap_shape = (layer_err.shape[1] - 1) * self.strides + self.fshape[0]
        backwarded_fmap = np.zeros((layer_err.shape[0], bfmap_shape, bfmap_shape, self.fshape[-2]))
        s = (backwarded_fmap.shape[1] - self.fshape[0]) // self.strides + 1
        for j in range(s):
            for i in range(s):
                backwarded_fmap[:, j * self.strides:j  * self.strides + self.fshape[0], i * self.strides:i * self.strides + self.fshape[1]] += np.sum(self.filters[np.newaxis, ...] * layer_err[:, j:j+1, i:i+1, np.newaxis, :], axis=4)
        return backwarded_fmap

    def get_grad(self, inputs, layer_err):
        total_layer_err = np.sum(layer_err, axis=(0, 1, 2))
        filters_err = np.zeros(self.fshape)
        s = (inputs.shape[1] - self.fshape[0]) // self.strides + 1
        summed_inputs = np.sum(inputs, axis=0)
        for j in range(s):
            for i in range(s):
                filters_err += summed_inputs[j  * self.strides:j * self.strides + self.fshape[0], i * self.strides:i * self.strides + self.fshape[1], :, np.newaxis]
        return filters_err * total_layer_err

    def update(self, grad):
        self.filters -= grad

