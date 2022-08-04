import array
import struct
import sys
from collections import namedtuple

import plotly.express as px
import numpy as np
from scipy.ndimage import uniform_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
from math import degrees, atan
import scipy.signal

TYPE_DIGITAL = 0
TYPE_ANALOG = 1
expected_version = 0

AnalogData = namedtuple('AnalogData', ('begin_time', 'sample_rate', 'downsample', 'num_samples', 'samples'))

def parse_analog(f):
    # Parse header
    identifier = f.read(8)
    if identifier != b"<SALEAE>":
        raise Exception("Not a saleae file")

    version, datatype = struct.unpack('=ii', f.read(8))

    if version != expected_version or datatype != TYPE_ANALOG:
        raise Exception("Unexpected data type: {}".format(datatype))

    # Parse analog-specific data
    begin_time, sample_rate, downsample, num_samples = struct.unpack('=dqqq', f.read(32))

    # Parse samples
    samples = array.array("f")
    samples.fromfile(f, num_samples)

    return AnalogData(begin_time, sample_rate, downsample, num_samples, samples)


if __name__ == '__main__':
    times = []
    volts = []
    anchor = 0
    filename = sys.argv[1]
    print("Opening " + filename)

    with open(filename, 'rb') as f:
        data = parse_analog(f)

    # Print out all analog data
    print("Begin time: {}".format(data.begin_time))
    print("Sample rate: {}".format(data.sample_rate))
    print("Downsample: {}".format(data.downsample))
    print("Number of samples: {}".format(data.num_samples))
    j = 0
    for idx, voltage in enumerate(data.samples):
        sample_num = idx * data.downsample
        #thing/(thing/sec) = thing*(sec/thing) = sec
        time = data.begin_time + (float(sample_num) / data.sample_rate)
        times.append(time)
        volts.append(min(voltage,1.3345))
        j = j + 1
    
    volts = scipy.ndimage.median_filter(volts, int((data.sample_rate/data.downsample)*.002)+1)
    
    #volts = uniform_filter1d(volts, size=int((data.sample_rate/data.downsample)*.002))
    

    """
    filtered = lowess(volts, times, frac=0.0005)
    plt.plot(filtered[:, 0], filtered[:, 1], 'r-', linewidth=3)
    plt.show()
    """

    upper_bound = lower_bound = volts[0]
    for i in range(0,int(data.num_samples*.2)):
        upper_bound = max(upper_bound, volts[i])
        lower_bound = min(lower_bound, volts[i])
        

    v_noise = .0
    sample_size = .3
    slope_range = int(data.num_samples*.05)
    
    temp_threshold = 0.0
    angle_threshold = 30.0
    tslope_range = 10
    """
    for s in range(100,11000,100):
        i = 0
        while i < int(data.num_samples*sample_size):
            l_b = max(i-s,0)
            r_b = min(i+s,data.num_samples) 
            v_noise = volts[r_b] - volts[l_b]
            if temp_threshold <= abs(degrees(atan(v_noise/((times[r_b]-times[l_b]))))):
                temp_threshold = abs(degrees(atan(v_noise/((times[r_b]-times[l_b])))))
                print("({},{})({},{})".format(times[l_b], volts[l_b], times[r_b], volts[r_b]))
            i = i + 1
        print("Temp Threshold: {}".format(temp_threshold))
        if temp_threshold < angle_threshold:
            angle_threshold = temp_threshold
            slope_range = s
            """
    print("Angle Threshold: {}".format(angle_threshold))
    start = 0
    state = 0
    #red is horizontal, b is rise, green is fall
    colors = ['r','b','g']
    i = 1
    angle_threshold = 1
    slope_range = int(data.num_samples*.002)
    while i < data.num_samples:
        l_b = max(i-slope_range,0)
        r_b = min(i+slope_range,data.num_samples-1)
        v_noise = volts[r_b] - volts[l_b]
        angle = degrees(atan(v_noise/((times[r_b]-times[l_b]))))
        if abs(angle) <= angle_threshold and state != 0:
            #print("Horizontal line detected: {}\n".format(angle))
            plt.plot(times[start:i], volts[start:i], colors[state])
            state = 0
            start = i
        elif angle > angle_threshold and state != 1:
            #print("Rise detected: {}\n".format(angle))
            plt.plot(times[start:i], volts[start:i], colors[state])
            state = 1
            start = i
        elif angle < -angle_threshold and state != 2:
            #print("Descent detected: {}\n".format(angle))
            plt.plot(times[start:i], volts[start:i], colors[state])
            state = 2
            start = i
        i = i + 1

    plt.plot(times[start:i], volts[start:i], colors[state])
    #plt.plot(times, volts)
    plt.show()