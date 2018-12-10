import serial
import time

""" Test of serial communication between Raspberry Pi and ATTiny """
ser = serial.Serial('/dev/ttyS0', 9600, timeout=1)    

try:
    while True:
        orig = str(ser.readline())
        data = orig[2:-5].split(':')
        if len(data) == 3:
            print(data)
        ser.flushInput()
        
except KeyboardInterrupt:
    ser.close()
