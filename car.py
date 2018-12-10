import time
from picamera import PiCamera
import constants
from functions import *
from pin_declarations import *
import RPi.GPIO as GPIO

from PIL import Image

import io

import numpy as np
from nn.network import Network
from nn.layers import FullyConnected, Flatten, Conv
from nn.activation import sigmoid, relu , lkrelu , mse, linear, cross_entropy

import pickle

class Car:
    """ Define the Car virtually
        Responsible for controlling and monitoring the vehicle 
        as well as handling all its sensors and outputs """

    def __init__(self, left_motor_pins, right_motor_pins):
        # Define inputs
        self.camera = PiCamera()

        # Define outputs
        self.left_motor = GPIO.PWM(left_motor_pins[1], 1000)
        self.right_motor = GPIO.PWM(right_motor_pins[1], 1000)

        self.left_motor_pins = [left_motor_pins[0], left_motor_pins[2]]
        self.right_motor_pins = [right_motor_pins[0], right_motor_pins[2]]

        # Start with motors not turning at all
        self.left_motor.start(0)
        self.right_motor.start(0)

        self._motor_steer = 0
        self._motor_throttle = 0

        self._left_motor_speed = 0
        self._right_motor_speed = 0    
        
        # Start preview here and leave it on constantly. This avoids restarting the camera every time the car takes a photo
        # Allowing to respond faster
        self.camera.resolution = (640, 480) 
        self.camera.start_preview()   
        

    def get_photo(self):
        """ Take a photo with the PiCamera and return an array 
            representing the image
            For not it just returns a blank array """
        # Implements the process shown in Section 4.5 Here
        # https://picamera.readthedocs.io/en/release-1.10/recipes1.html

        
        stream = io.BytesIO()
        time.sleep(0.1)
        self.camera.capture(stream, format='jpeg')
        stream.seek(0)
        return np.asarray(Image.open(stream).resize((28, 28)))

    def input_state(self):
        """ Returns the inputs (what the camera is reading) """
        photo = self.get_photo()
        return photo

    """ Uses property attributes to associate variables with their respective outputs
        e.g setting motor_speed changes both the variable and the actual speed of the motor """

    @property
    def motor_steer(self):
        """" motor_speed getter """
        return self._motor_steer
    
    @motor_steer.setter
    def motor_steer(self, value):
        """ motor_speed setter """
        self._motor_steer = value
        self.set_motors(self._motor_steer, self._motor_throttle)

    @property
    def motor_throttle(self):
        """ motor_throttle getter """
        return self._motor_throttle

    @motor_throttle.setter
    def motor_throttle(self, value):
        """ motor_throttle getter """
        self._motor_throttle = value
        self.set_motors(self._motor_steer, self._motor_throttle)

    def set_motors(self, motor_steer, motor_throttle):
        """ set physical motor speeds from each value """
        motor_speed = abs(self._motor_throttle - 0.5) * 200
        motor_dir = self._motor_throttle > 0.5

        # Code that linearly interpolates between edge cases
        self.left_motor.start(constrained_map(motor_steer, 0, 0.5, 0, 1) * motor_speed)
        self.right_motor.start(constrained_map(motor_steer, 1, 0.5, 0, 1) * motor_speed)  

        left_motor_speed = 
        GPIO.output(self.left_motor_pins[0], motor_dir)
        GPIO.output(self.left_motor_pins[1], not motor_dir)

        GPIO.output(self.right_motor_pins[0], motor_dir)
        GPIO.output(self.right_motor_pins[1], not motor_dir)

    """ Define the state as the outputs of the plane
            i.e. the motor_steer and motor_throttle """

    @property
    def output_state(self):
        """ state getter """
        return [self.motor_steer, self.motor_throttle]

    @output_state.setter
    def output_state(self, values):
        """ state setter """
        self.motor_steer = values[0]
        self.motor_throttle = values[1]

class AutonomousCar(Car):
    """ Like the car class, but contains a neural network that updates the state
        based on inputs allowing for autonomous flight """
    def __init__(self, left_motor_pins, right_motor_pins):
        super().__init__(left_motor_pins, right_motor_pins)
        self.net = pickle.load(open("network.nn", "rb"))

    def autonomous_flight(self):
        self.state = self.net.forward([self.get_photo()])[0]
