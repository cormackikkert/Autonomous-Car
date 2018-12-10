import serial
from car import AutonomousCar
from pin_declarations import *
import pickle

from functions import *

import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)

for pin in MOTORL_PINS:
    GPIO.setup(pin, GPIO.OUT)
for pin in MOTORR_PINS:
    GPIO.setup(pin, GPIO.OUT)

ser = serial.Serial('/dev/ttyS0', 9600, timeout=1) 

AUTONOMOUS_CONTROL = 0
MANUAL_CONTROL = 1

# Change this depending on what you want the car to do
STATE = AUTONOMOUS_CONTROL

# When in autonomous control there are two secondary states
# 0. Manual control: Car is controlled by the user
# 1. Autonomous control: Car is controlled by onboard neural network

# When in manual control there are again two secondary states
# 0. Manaul control: Car is controlled by the user
# 1. Learning: Car is again controlled by the user but now its situation and actions are recorded

car = AutonomousCar(MOTORL_PINS, MOTORR_PINS)

train_X = []
train_Y = []

# readings[0] -> Steering value (0, 255)
# readings[1] -> Throttle value (0, 255)
# readings[2] -> Switch value (0 or 1)
readings = [0, 0, 0]

# Variable used to remember the last state
# Used to save the neural network upon leaving the learning state
last_state = False

try:
    while True:

        # Read inputs
        orig = str(ser.readline())
        data = orig[2:-5].split(':')

        # only accept data if its valid (a list of 3 integers)
        try:
            if len(data) == 3:
                readings = list(map(int, data))
        except:
            pass
            
        ser.flushInput()

        secondary_state = readings[2]
        
        
        if STATE == AUTONOMOUS_CONTROL:
            if secondary_state == 0:
                car.autonomous_flight()
            elif secondary_state == 1:
                # Manual control
                car.output_state = [readings[0] / 255, readings[1] / 255]

        elif STATE == MANUAL_CONTROL:
            # Manual control
            car.output_state = [readings[0] / 255, readings[1] / 255]
            
            if secondary_state == 1:
                last_state = True

                # Record data if in recording substate
                train_X.append(car.input_state())
                train_Y.append(car.output_state)
                
            if secondary_state == 0 and last_state == True:
                last_state = False

                # Save data if exiting recording substate
                pickle.dump(train_X, open("training_data_inputs.p", "wb"))
                pickle.dump(train_Y, open("training_data_outputs.p", "wb"))
                
        
# The code can only be stopped by doing a KeyboardInterrupt
# So use that to end the program
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()
    ser.close()
    
# Note that even though the code is run on the raspberry pi without a keyboard, 
# running and stopping the code is done remotely on a laptop through SSH



