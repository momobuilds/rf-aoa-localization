import numpy as np
import adi

sample_rate = 1e6 # Hz
center_freq = 700e6 # Hz
num_samps = 20 # number of samples returned per call to rx()

sdr = adi.Pluto('ip:192.168.2.1')

sdr.rx_enabled_channels = [0,1]
sdr.gain_control_mode_chan1 = 'manual'
sdr.gain_control_mode_chan2 = 'manual'

sdr.rx_hardwaregain_chan0 = 30.0 # dB
sdr.rx_hardwaregain_chan1 = 30.0 # dB

sdr.rx_lo = int(center_freq)
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate) # filter width, just set it to the same as sample rate for now
sdr.rx_buffer_size = num_samps

samples = sdr.rx() # receive samples off Pluto

rx0= samples[0]
rx1= samples[1]
print(samples)
print('10 samples for rx0')
print(rx0[:5])
print('10 samples for rx1')
print(rx1[:5])
print(sdr.rx_enable_channels)

