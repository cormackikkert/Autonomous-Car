import events, gui, vectors, constants
import pygame 

import sys
# allow code and files in directory above to be used 
# Change this to the file location of the directory above this file
sys.path.append(r"C:\Users\corma\OneDrive\Documents\Major Project\Code")

from nn.network import Network
from nn.layers import FullyConnected, Flatten, Conv
from nn.activation import sigmoid, relu, lkrelu, mse, linear

import pickle
import numpy as np

pygame.display.set_caption('Neural network demonstration')
clock = pygame.time.Clock()
pygame.init()

gameDisplay = pygame.display.set_mode((1000, 660), pygame.DOUBLEBUF) # Create display
controller = events.EventManager()

# These objects need references, as they are controlled by the NetworkHandler object
speedometerIdeal = gui.Speedometer(250, 630, 120, gameDisplay, evManager=controller)
wheelIdeal = gui.SteeringWheel(250, 630, 160, gameDisplay, evManager=controller)
speedometerFake = gui.Speedometer(750, 630, 120, gameDisplay, evManager=controller)
wheelFake = gui.SteeringWheel(750, 630, 160, gameDisplay, evManager=controller)

# Load network and training information
network = pickle.load(open("network.nn", "rb"))
train_data_X = np.array(pickle.load(open("training_data_inputs.p", "rb")))
train_data_Y = np.array(pickle.load(open("training_data_outputs.p", "rb")))

network_handler = gui.NetworkHandler(network, [train_data_X, train_data_Y], [speedometerFake, wheelFake], [speedometerIdeal, wheelIdeal], gameDisplay, evManager=controller)
    
# Define objects used in scene
sketch_objects = [
    events.KeyboardController(evManager=controller),
    network_handler,
    gui.TextBoxEvent(250 - 200, 400, 400, 50, "Ideal Output", gameDisplay, evManager=controller),
    gui.TextBoxEvent(750 - 200, 400, 400, 50, "Network Output", gameDisplay, evManager=controller),
    gui.Button("Next Image", network_handler.update_next, gameDisplay, rect=gui.Rect(450, 80, 450, 80), evManager=controller),
    gui.Button("Random Image", network_handler.update_random, gameDisplay, rect=gui.Rect(450, 190, 450, 80), evManager=controller)
]


for obj in sketch_objects:
    controller.registerListener(obj)

while True:
    gameDisplay.fill((0, 0, 0))
    
    controller.push(events.TickEvent())
    controller.push(events.RenderEvent())

    pygame.display.update()

    clock.tick(30)

