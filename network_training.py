""" Code that trains a neural network using data from training_data_inputs.p
    and training_data_outputs.p. This saves a network (network.nn) which can 
    be put onto the Raspberry Pi to allow for autonomous control """

import numpy as np
from nn.network import Network
from nn.layers import FullyConnected, Flatten, Conv
from nn.activation import sigmoid, relu, lkrelu, mse, linear, cross_entropy

import pickle

def accuracy(net, X, Y):
    # Return average loss of the neural network against all the training data
    return np.sum(mse.compute((net.forward(X), Y))) / len(X)

train_data_X = np.array(pickle.load(open("training_data_inputs.p", "rb")))
train_data_Y = np.array(pickle.load(open("training_data_outputs.p", "rb")))

if __name__ == '__main__':
    batch_size = 20

    # A simple convnet
    # weight_init and filter_init initilialize the paramaters of the neural network,
    # making the weights inversely proportional to the parameters in each layer
    layers = [
        Conv((4, 4, 3, 20), strides=2, activation=lkrelu, filter_init=lambda shp: np.random.normal(size=shp) * np.sqrt(1.0 / (28*28 + 13*13*20)) ),
        Conv((5, 5, 20, 40), strides=2, activation=lkrelu, filter_init=lambda shp:  np.random.normal(size=shp) *  np.sqrt(1.0 / (13*13*20 + 5*5*40)) ),
        Flatten((5, 5, 40)),
        FullyConnected((5*5*40, 100), activation=relu, weight_init=lambda shp: np.random.normal(size=shp) * np.sqrt(1.0 / (5*5*40 + 100.))),
        FullyConnected((100, 2), activation=sigmoid, weight_init=lambda shp: np.random.normal(size=shp) * np.sqrt(1.0 / (110.)))
    ]

    lr = 0.000000001

    net = Network(layers, lr=lr, loss=mse)
    # If you want to continue training a network uncomment the line below
    # net = pickle.load(open("network.nn", "rb"))

    train_data_X = np.reshape(train_data_X, [-1, 28, 28, 3])

    try:
        for epoch in range(100000):
            shuffled_index = np.random.permutation(train_data_X.shape[0])

            # Get minibatch of training data
            batch_train_X = train_data_X[shuffled_index[:batch_size]]
            batch_train_Y = train_data_Y[shuffled_index[:batch_size]]
            net.train_step((batch_train_X, batch_train_Y))

            loss = np.sum(mse.compute((net.forward(batch_train_X), batch_train_Y)))

            if epoch % 200 == 0:
                # Print average loss of neural network
                print(accuracy(net, train_data_X, train_data_Y) / len(train_data_X))
                pickle.dump(net, open("network.nn", "wb"))

    except KeyboardInterrupt:
        print(accuracy(net, train_data_X, train_data_Y))
        pickle.dump(net, open("network.nn", "wb"))

print(accuracy(net, train_data_X, train_data_Y))
pickle.dump(net, open("network.nn", "wb"))
