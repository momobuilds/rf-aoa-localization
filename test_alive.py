import numpy as np
import matplotlib.pyplot as plt
import adi

# ------------------------------------------------------------
# Receiver settings
# ------------------------------------------------------------

sample_rate = int(5e6)
center_freq = int(886.6e6)   # CHANGE if the lab transmitter uses another RF frequency
rx_bw = int(3.84e6)

capture_time = 0.050       # 50 ms
num_samps = int(sample_rate * capture_time)

print("Capturing samples:", num_samps)
print("Capture duration [ms]:", 1000 * num_samps / sample_rate)

# ------------------------------------------------------------
# Pluto setup
# ------------------------------------------------------------

sdr = adi.ad9361(uri="ip:192.168.2.1")
sdr.rx_enabled_channels = [0, 1]

sdr.sample_rate = sample_rate
sdr.rx_lo = center_freq
sdr.rx_rf_bandwidth = rx_bw
sdr.rx_buffer_size = num_samps

sdr.gain_control_mode_chan0 = "manual"
sdr.gain_control_mode_chan1 = "manual"
sdr.rx_hardwaregain_chan0 = 30.0
sdr.rx_hardwaregain_chan1 = 30.0

try:
    sdr.rx_destroy_buffer()
except Exception:
    pass

# ------------------------------------------------------------
# Capture
# ------------------------------------------------------------

samples = sdr.rx()

rx0 = np.asarray(samples[0], dtype=np.complex64)
rx1 = np.asarray(samples[1], dtype=np.complex64)

rx0 = rx0 - np.mean(rx0)
rx1 = rx1 - np.mean(rx1)

print("RX0 length:", len(rx0))
print("RX1 length:", len(rx1))

print("RX0 max abs:", np.max(np.abs(rx0)))
print("RX1 max abs:", np.max(np.abs(rx1)))

# ------------------------------------------------------------
# Smooth received power
# ------------------------------------------------------------

t_ms = np.arange(len(rx0)) / sample_rate * 1000

p0 = np.abs(rx0) ** 2
p1 = np.abs(rx1) ** 2

# Try 100 us smoothing. This makes packets easier to see.
smooth_time_us = 100
smooth_len = max(1, int(sample_rate * smooth_time_us * 1e-6))

window = np.ones(smooth_len) / smooth_len

p0_smooth = np.convolve(p0, window, mode="same")
p1_smooth = np.convolve(p1, window, mode="same")

# Plot relative to median noise/power floor
p0_rel_db = 10 * np.log10((p0_smooth + 1e-12) / (np.median(p0_smooth) + 1e-12))
p1_rel_db = 10 * np.log10((p1_smooth + 1e-12) / (np.median(p1_smooth) + 1e-12))

# ------------------------------------------------------------
# Packet detection by threshold
# ------------------------------------------------------------

threshold_db = 4.0

active0 = p0_rel_db > threshold_db
active1 = p1_rel_db > threshold_db

# Joint condition: packet should appear on either/both channels
active = active0 | active1

edges = np.diff(active.astype(int))
starts = np.where(edges == 1)[0]
ends = np.where(edges == -1)[0]

# Handle packet active at beginning/end
if active[0]:
    starts = np.r_[0, starts]
if active[-1]:
    ends = np.r_[ends, len(active) - 1]

packets = []

for s, e in zip(starts, ends):
    width_ms = (e - s) / sample_rate * 1000
    center_ms = ((s + e) / 2) / sample_rate * 1000

    # Ignore very tiny random spikes
    if width_ms > 0.03:
        packets.append((center_ms, width_ms))

print("\nDetected packet candidates:")
for i, (center_ms, width_ms) in enumerate(packets):
    print(f"Packet {i+1}: center = {center_ms:.3f} ms, width = {width_ms:.3f} ms")

print("Number of packet candidates:", len(packets))

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure()
plt.plot(t_ms, p0_rel_db, label="RX0 relative power")
plt.plot(t_ms, p1_rel_db, label="RX1 relative power")
plt.axhline(threshold_db, linestyle="--", label="threshold")

for center_ms, width_ms in packets:
    plt.axvline(center_ms, linestyle=":")

plt.xlabel("Time [ms]")
plt.ylabel("Power relative to median [dB]")
plt.title("50 ms transmitter-alive test")
plt.grid(True)
plt.legend()
plt.show()

# ------------------------------------------------------------
# Save capture
# ------------------------------------------------------------

np.savez(
    "pluto_50ms_alive_test.npz",
    rx0=rx0,
    rx1=rx1,
    sample_rate=sample_rate,
    center_freq=center_freq,
)
