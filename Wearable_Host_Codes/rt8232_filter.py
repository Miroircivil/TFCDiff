from highpass_filter import EnhancedHighPassFilter
from lowpass_filter import LowPassFilter
from noth_filter import NotchFilter
import numpy as np

class EcgFilter:

    def __init__(self):

        self.DictHpFilter = {}
        self.DictHpFilter[500] = EnhancedHighPassFilter(0.5, 500, 2)

        self.DictLowPassFilter = {}
        self.DictLowPassFilter[500] = LowPassFilter(sample_rate=500.0, cutoff_freq=40.0)

        self.DictNothFilter = {}
        self.DictNothFilter[500] = NotchFilter(fs=500, filter_type='comb', harmonics=5)

    # def getWindowFilter(self, signal, window_size=9):
    #     return np.convolve(signal, np.ones(window_size)/window_size, mode='valid')
    
    def process_buffer(self,rate,buffer):

        output = buffer

        # if rate in self.DictHpFilter:
        #     output = self.DictHpFilter[rate].process_buffer(buffer)

        if rate in self.DictLowPassFilter:
            output = self.DictLowPassFilter[rate].process(buffer)

        if rate in self.DictNothFilter:
            output = self.DictNothFilter[rate].process_buffer(output)

        # output = self.getWindowFilter(output, window_size=9)

        return output
