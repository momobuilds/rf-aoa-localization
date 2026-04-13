import adi


sdr = adi.Pluto('ip:192.168.2.1')

print(sdr.rx_channel_names)
print(sdr.rx_enabled_channels)
