All the code in /Application can be run on a desktop / laptop.
Start the app by running application.py.
On the top left an image is shown, with two speedometers and steering wheels displayed underneath this image
The speedometer and steering wheel pair on the left show what the user did when the car saw this image
The speedometer and steering wheel pair on the right show what the neual network would do when presented this situation

network_training.py runs on a desktop / laptop, 
and trains the neural network using training_data_inputs.p and training_data_outputs.pair
it saves the network as network.nn every 200 epochs (loops of the training data)

network_test was a failed attempt to model a Pytorch network using numpy. It uses alot of code from https://towardsdatascience.com/building-convolutional-neural-network-using-np-from-scratch-b30aac50e50a:

The folder nn contains all the code from a numpy convolutional network library I used, available here: https://github.com/Eyyub/numpy-convnet
The only modification I made was to add a TanH activation function, which I ended up not using, instead favouring the sigmoid function

All the other code is run on a raspberry Pi, that controls a small car