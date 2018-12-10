""" Code that attempted to model a pytorch network using only numpy (Didnt work) """

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

torch.random.manual_seed(2)

class ConvBlock(nn.Module):
    """ A block to be used in the convolutional neural network
        Is composed of a convolutional layer and a pooling layer
    """
    def __init__(self, inner_channel, outer_channel, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(inner_channel, outer_channel, kernel_size)
        self.pool = nn.MaxPool2d(2, 2) # used to downscale the output block

    def forward(self, x):
        return self.pool(F.relu(self.conv(x)))

class ConvNet(nn.Module):
    """ Define a convolutional neural network
        composed of multiple conv blocks """
    def __init__(self, layer_num, channel_size, kernel_size):
        super().__init__()

        # Accept RGB images (3 Channels)
        self.conv_blocks = [ConvBlock(3, channel_size, kernel_size)]

        # Add more conv layers
        for i in range(layer_num):
            self.conv_blocks.append(ConvBlock(channel_size * (i + 1), channel_size * (i + 2), kernel_size))

            
    def forward(self, x):
        i = 0
        for block in self.conv_blocks:
            x = block(x)
            i += 1
            if i == 1:
                print("First pass", x, x.shape)
        return x

class Network(nn.Module):
    """ Define the network we will be aiming to replicate
        Specs:
            1.) ConvBlock(in_channel = 3, out_channel = 5, kernel=(3, 3)) (3x32x32) -> (15x15x5)
            2.) ConvBlock(in_channel = 5, out_channel = 10, kernel=(3, 3)) (5x15x15) -> (6x6x10)
            3.) ConvBlock(in_channel = 10, out_channel = 15, kernel=(3, 3)) (15x2x2)
            3.) FC layer: 60 -> 64 
            4.) FC layer 64 -> 32
            5.) FC layer 32 -> 2 """

    def __init__(self):
        super().__init__()

        # Define the neural network
        self.conv_net = ConvNet(2, 5, 3)

        # Calculate the size of the output layer
        output_channels = 15
        output_size = 32
        for i in range(3):
            output_size = output_size // 2 - 1

        # Define fully connected layers
        self.fc1 = nn.Linear(output_channels * output_size ** 2, 64) # Add 3 for the accelerometer readings
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

        self.optimizer = optim.Adam(self.parameters(), lr = 2e-4, betas = (0.5, 0.999))
        self.loss = nn.MSELoss()

    def forward(self, image):
        """ Return the output of the neural network
            Uses .unsqueeze_(0) to add a batch dimension """
        image = torch.tensor(image)

        # Add a batch dimension if one is not present
        if image.shape[0] != 1:
            image = image.unsqueeze_(0)

        
        x = self.conv_net(image)

        # Convert convolutional layer to a vector
        # This allows it to be passed into a feedforward neural network
        x = x.view(-1).unsqueeze_(0)

        return F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))


""" Following are functions that model layers of a neural network using numpy
    I did not write this code, it is from https://towardsdatascience.com/building-convolutional-neural-network-using-np-from-scratch-b30aac50e50a
    I only made one change, which was to fix up some incorrect math in the calculation of the output size of pooling layer """
def conv_(img, conv_filter):
    # https://towardsdatascience.com/building-convolutional-neural-network-using-np-from-scratch-b30aac50e50a
    filter_size = conv_filter.shape[1]
    result = np.zeros((img.shape))
    #Looping through the image to apply the convolution operation.
    for r in np.uint16(np.arange(filter_size/2.0, 
                          img.shape[0]-filter_size/2.0+1)):
        for c in np.uint16(np.arange(filter_size/2.0, 
                                           img.shape[1]-filter_size/2.0+1)):
            
            curr_region = img[r-np.uint16(np.floor(filter_size/2.0)):r+np.uint16(np.ceil(filter_size/2.0)), 
                              c-np.uint16(np.floor(filter_size/2.0)):c+np.uint16(np.ceil(filter_size/2.0))]
            #Element-wise multipliplication between the current region and the filter.
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)
            result[r, c] = conv_sum
            
    #Clipping the outliers of the result matrix.
    final_result = result[np.uint16(filter_size/2.0):result.shape[0]-np.uint16(filter_size/2.0), 
                          np.uint16(filter_size/2.0):result.shape[1]-np.uint16(filter_size/2.0)]
    return final_result

def conv(img, conv_filter):
    # https://towardsdatascience.com/building-convolutional-neural-network-using-np-from-scratch-b30aac50e50a
    # An empty feature map to hold the output of convolving the filter(s) with the image.
    feature_maps = np.zeros((img.shape[0]-conv_filter.shape[1]+1, 
                                img.shape[1]-conv_filter.shape[1]+1, 
                                conv_filter.shape[0]))
    # Convolving the image by the filter(s).
    for filter_num in range(conv_filter.shape[0]):
        curr_filter = conv_filter[filter_num, :] # getting a filter from the bank.
        """ 
        Checking if there are mutliple channels for the single filter.
        If so, then each channel will convolve the image.
        The result of all convolutions are summed to return a single feature map.
        """
        if len(curr_filter.shape) > 2:
            conv_map = conv_(img[:, :, 0], curr_filter[:, :, 0]) # Array holding the sum of all feature maps.
            for ch_num in range(1, curr_filter.shape[-1]): # Convolving each channel with the image and summing the results.
                conv_map = conv_map + conv_(img[:, :, ch_num], 
                                  curr_filter[:, :, ch_num])
        else: # There is just a single channel in the filter.
            conv_map = conv_(img, curr_filter)
        feature_maps[:, :, filter_num] = conv_map # Holding feature map with the current filter.
    return feature_maps # Returning all feature maps.

def relu(feature_map):
    # https://towardsdatascience.com/building-convolutional-neural-network-using-np-from-scratch-b30aac50e50a
    #Preparing the output of the ReLU activation function.
    relu_out = np.zeros(feature_map.shape)
    for map_num in range(feature_map.shape[-1]):
        for r in np.arange(0,feature_map.shape[0]):
            for c in np.arange(0, feature_map.shape[1]):
                relu_out[r, c, map_num] = np.max([feature_map[r, c, map_num], 0])
    return relu_out

def pool(feature_map, size=2, stride=2):
    # https://towardsdatascience.com/building-convolutional-neural-network-using-np-from-scratch-b30aac50e50a
    #Preparing the output of the pooling operation.
    # Fixed up some incorrect math in the calculation of the output size
    pool_out = np.zeros((np.uint16((feature_map.shape[0]-size)/stride + 1),
                            np.uint16((feature_map.shape[1]-size)/stride + 1),
                            feature_map.shape[-1]))
    for map_num in range(feature_map.shape[-1]):
        r2 = 0
        for r in np.arange(0,feature_map.shape[0]-size-1, stride):
            c2 = 0
            for c in np.arange(0, feature_map.shape[1]-size-1, stride):
                pool_out[r2, c2, map_num] = np.max([feature_map[r:r+size,  c:c+size]])
                c2 = c2 + 1
            r2 = r2 +1
    return pool_out

def ReLU(n):
    """Rectified Linear Unit
       Returns x if x > 0, else 0"""
    return np.maximum(0, n)

""" Network that implements the above functions in an object (I wrote this part onwards) """
class npNetwork:
    def __init__(self):
        self.l1_filter = np.zeros((5, 3, 3, 3))
        self.l2_filter = np.zeros((10, 3, 3, 5))
        self.l3_filter = np.zeros((15, 3, 3, 10))

        self.weights = []
        self.biases = []

        for prev, current in [(60, 64), (64, 32), (32, 2)]:
            # Randomly initialize weights and biases in the range -1 to 1
            self.weights.append(2 * np.random.random((current, prev)) - 1)
            self.biases.append(2 * np.random.random((current, 1)) - 1)
    
    def forward(self, img):
        # Pass image through convnet
        img = np.array(img)
        img = pool(relu(conv(img, self.l1_filter)))
        img = pool(relu(conv(img, self.l2_filter)))
        img = pool(relu(conv(img, self.l3_filter)))

        # Flatten image to pass through feed forward neural network
        layer = np.array([img.flatten()]).transpose()
        for b, w in zip(self.biases, self.weights):
            layer = ReLU(np.dot(w, layer) + b)
        return layer

# Pytorch had RGB channels as the last dimension when inputted into a convent
# whilst the numpy library has this as the first dimension
# This meant two different inputs needed to be used for each network (note img and nimg are the same except for the dimensions swapped)
img = [[[i/3 for x in range(32)] for y in range(32)] for i in range(3)]
nimg = [[[0.0, 0.5, 1.0] for x in range(32)] for y in range(32)]


# Use weights in torch network
# To copy paste them use print(repr(notideal.l1_filter))
# Use convolutional filters
# print(notideal.l1_filter.shape, np.transpose(ideal.conv_net.conv_blocks[0].conv.weight.detach().numpy(), (0, 3, 2, 1)).shape)
# print(notideal.l2_filter.shape, np.transpose(ideal.conv_net.conv_blocks[1].conv.weight.detach().numpy(), (0, 3, 2, 1)).shape)
# print(notideal.l3_filter.shape, np.transpose(ideal.conv_net.conv_blocks[2].conv.weight.detach().numpy(), (0, 3, 2, 1)).shape)

ideal = torchNetwork()
notideal = npNetwork()

notideal.l1_filter = np.transpose(ideal.conv_net.conv_blocks[0].conv.weight.detach().numpy(), (0, 3, 2, 1))
notideal.l2_filter = np.transpose(ideal.conv_net.conv_blocks[1].conv.weight.detach().numpy(), (0, 3, 2, 1))
notideal.l3_filter = np.transpose(ideal.conv_net.conv_blocks[2].conv.weight.detach().numpy(), (0, 3, 2, 1))

print(ideal(img))
print(notideal.forward(nimg))

# Fully connected weights
print("Fully connected weights")
print(notideal.weights[0].shape, ideal.fc1.weight.detach().numpy().shape)
print(notideal.weights[1].shape, ideal.fc2.weight.detach().numpy().shape)
print(notideal.weights[2].shape, ideal.fc3.weight.detach().numpy().shape)


notideal.weights[0] = ideal.fc1.weight.detach().numpy()
notideal.weights[1] = ideal.fc2.weight.detach().numpy()
notideal.weights[2] = ideal.fc3.weight.detach().numpy()

# Fully connected layer biases
print("Fully connected biases")
print(notideal.biases[0].shape, np.expand_dims(ideal.fc1.bias.detach().numpy(), axis=1).shape)
print(notideal.biases[1].shape, np.expand_dims(ideal.fc2.bias.detach().numpy(), axis=1).shape)
print(notideal.biases[2].shape, np.expand_dims(ideal.fc3.bias.detach().numpy(), axis=1).shape)

notideal.biases[0] = np.expand_dims(ideal.fc1.bias.detach().numpy(), axis=1)
notideal.biases[1] = np.expand_dims(ideal.fc2.bias.detach().numpy(), axis=1)
notideal.biases[2] = np.expand_dims(ideal.fc3.bias.detach().numpy(), axis=1)

