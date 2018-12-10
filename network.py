import constants

import torch
import torch.nn as nn
import torch.optim as optim

""" Pytorch implemenation of the neural network
    It ended up not being used as it could not be run on the Raspberry Pi  """
class ConvBlock(nn.Module):
    """ A block to be used in the convolutional neural network
        Is composed of a convolutional layer and a pooling layer
    """
    def __init__(self, inner_channel, outer_channel, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(inner_channel, outer_channel, kernel_size)
        self.pool = nn.MaxPool2d(2, 2) # used to downscale the output block

    def forward(self, x):
        return self.pool(self.conv(x))

class ConvNet(nn.Module):
    """ Define a convolutional neural network
        composed of multiple conv blocks """
    def __init__(self, layer_num, input_channel, channel_size, kernel_size):
        super().__init__()

        # Accept RGB images (3 Channels)
        self.conv_blocks = [ConvBlock(3, channel_size, kernel_size)]

        # Add more conv layers
        for i in range(layer_num):
            self.conv_blocks.append(ConvBlock(channel_size * (i + 1), channel_size * (i + 2), kernel_size))

            
    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        return x

class Network(nn.Module):
    """ The network for the plane
        Uses a Convnet to analyze an image from the camera,
        which is then connected to a feed forward neural network,
        where there accelerometer readings are added.
        """
    def __init__(self):
        super().__init__()

        # Define the neural network
        self.conv_net = ConvNet(constants.CONV_LAYER_NUM, 8, 5, 3)

        # Calculate the size of the output layer
        output_channels = 5 * (constants.CONV_LAYER_NUM + 1)
        output_size = constants.IMAGE_SIZE
        for i in range(constants.CONV_LAYER_NUM + 1):
            output_size = output_size // 2 - 1

        # Define fully connected layers
        self.fc1 = nn.Linear(output_channels * output_size ** 2 + 3, 64) # Add 3 for the accelerometer readings
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

        self.optimizer = optim.Adam(self.parameters(), lr = 2e-4, betas = (0.5, 0.999))
        self.loss = nn.MSELoss()

    def forward(self, image, accel_readings):
        """ Return the output of the neural network
            Uses .unsqueeze_(0) to add a batch dimension """
        image = torch.tensor(image)
        accel = torch.tensor(accel_readings)

        # Add a batch dimension if one is not present
        if image.shape[0] != 1:
            image = image.unsqueeze_(0)
        if accel.shape[0] != 1:
            accel = accel.unsqueeze_(0)

        
        x = self.conv_net(image)

        # Convert convolutional layer to a vector
        # This allows it to be passed into a feedforward neural network
        x = x.view(-1).unsqueeze_(0)

        # Add accelerometer readings to
        x = torch.cat([x, accel], 1)
        return self.fc3(self.fc2(self.fc1(x)))[0] # Index to remove the batch dimension

    def optimize_parameters(self, inputs, targets):
        """ Update the parameters of the neural network based on a training data 
            Does this by using backpropagation
            Returns the loss """

        image, accel = inputs 

        self.optimizer.zero_grad() # Reset gradients to zero

        # calculates how far away the actual output was from the target output
        loss = self.loss(self(image, accel), torch.tensor(targets))

        loss.backward() # Calculate the gradient of each parameter with respect to the loss

        # update parameters of neural network to minimize the loss
        # Does this by updating each parameter by a function influenced by its gradient
        self.optimizer.step() 

        return loss
