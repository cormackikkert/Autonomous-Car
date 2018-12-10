def amap(x, in_min, in_max, out_min, out_max): 
    """ remaps a number from one range to another
        https://www.arduino.cc/reference/en/language/functions/math/map/ """
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def constrained_map(x, in_min, in_max, out_min, out_max):
    """ Same thing as the arduino map but instead ensures the output
        is between out_min and out_max """
    return min(max(amap(x, in_min, in_max, out_min, out_max), out_min), out_max)

