import numpy as np
import matplotlib.pyplot as plt
import adi

from srs_config import (
    SystemParams,
    SRSConfigCommon,
    SRSConfigDedicated,
    generateSRSsequence,
)

# ------------------------------------------------------------
# Receiver settings
# ------------------------------------------------------------

sample_rate = int(5e6)
center_freq = int(886.6e6)
rx_bw = int(3.84e6)

capture_time = 0.050
num_samps = int(sample_rate * capture_time)

# ------------------------------------------------------------
# SRS configuration
# ------------------------------------------------------------

sys_params = SystemParams(
    C_RNTI=1,
    N_RB_UL=25,
    PCID=0,
)

cfg_com = SRSConfigCommon(
    srs_bandwidthConfig=5,
    srs_subframeConfig=0,
)

cfg_ded = SRSConfigDedicated(
    srs_bandwidth=0,
    srs_hoppingBandwidth=3,
    freqDomainPosition=0,
    duration=1,
    config_Index=77,
    transmissionComb=0,
    cyclicShift=0,
    srs_AntennaPort=10,
    transmissionCombNum=2,
)

r, k0, sc_idx, T_per, T_off = generateSRSsequence(
    cfg_ded,
    cfg_com,
    sys_params,
)

print("Frequency-domain SRS length:", len(r))
print("k0:", k0)
print("sc_idx first:", sc_idx[0])
print("sc_idx last:", sc_idx[-1])
print("T_per [ms]:", T_per)
print("T_off [ms/subframes]:", T_off)

# ------------------------------------------------------------
# Convert SRS to time-domain OFDM symbol
# ------------------------------------------------------------

def make_time_domain_srs(r, sc_idx, N_RB_UL, n_fft=512, cp_len=36):
    freq_grid = np.zeros(n_fft, dtype=np.complex64)

    n_used = N_RB_UL * 12
    grid_start = n_fft // 2 - n_used // 2

    fft_bins = grid_start + sc_idx.astype(int)

    if np.any(fft_bins < 0) or np.any(fft_bins >= n_fft):
        raise ValueError("SRS subcarriers are outside FFT grid.")

    freq_grid[fft_bins] = r

    time_no_cp = np.fft.ifft(np.fft.ifftshift(freq_grid))
    time_with_cp = np.r_[time_no_cp[-cp_len:], time_no_cp]

    time_with_cp = time_with_cp / (
        np.sqrt(np.mean(np.abs(time_with_cp) ** 2)) + 1e-12
    )

    return time_with_cp.astype(np.complex64)


local_srs_td = make_time_domain_srs(
    r,
    sc_idx,
    N_RB_UL=sys_params.N_RB_UL,
    n_fft=512,
    cp_len=36,
)

L = len(local_srs_td)

print("Time-domain local SRS length:", L)

# ------------------------------------------------------------
# Pluto setup
# ------------------------------------------------------------

print("Capturing samples:", num_samps)
print("Capture duration [ms]:", 1000 * num_samps / sample_rate)

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
# Receive IQ samples
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
# Normalize before correlation
# ------------------------------------------------------------

rx0_norm = rx0 / (np.sqrt(np.mean(np.abs(rx0) ** 2)) + 1e-12)
rx1_norm = rx1 / (np.sqrt(np.mean(np.abs(rx1) ** 2)) + 1e-12)

# ------------------------------------------------------------
# Correlation
# ------------------------------------------------------------

corr0_complex = np.correlate(rx0_norm, local_srs_td.conj(), mode="valid")
corr1_complex = np.correlate(rx1_norm, local_srs_td.conj(), mode="valid")

corr0 = np.abs(corr0_complex)
corr1 = np.abs(corr1_complex)

corr0_norm = corr0 / (np.max(corr0) + 1e-12)
corr1_norm = corr1 / (np.max(corr1) + 1e-12)

corr_joint = corr0_norm + corr1_norm
corr_joint = corr_joint / (np.max(corr_joint) + 1e-12)

peak0 = np.argmax(corr0_norm)
peak1 = np.argmax(corr1_norm)
peak_joint = np.argmax(corr_joint)

print()
print("Correlation results:")
print("RX0 peak index:", peak0)
print("RX1 peak index:", peak1)
print("Joint peak index:", peak_joint)

print("RX0 peak time [ms]:", peak0 / sample_rate * 1000)
print("RX1 peak time [ms]:", peak1 / sample_rate * 1000)
print("Joint peak time [ms]:", peak_joint / sample_rate * 1000)

print("RX1 - RX0 peak difference [samples]:", peak1 - peak0)
print("RX1 - RX0 peak difference [us]:", (peak1 - peak0) / sample_rate * 1e6)

# ------------------------------------------------------------
# Extract same SRS region from both channels
# ------------------------------------------------------------

start = peak_joint
end = start + L

if end > len(rx0):
    raise RuntimeError("Peak is too close to the end of the received buffer.")

rx0_srs = rx0[start:end]
rx1_srs = rx1[start:end]

print()
print("Extracted SRS region:")
print("start:", start)
print("end:", end)
print("length:", len(rx0_srs))

# Preliminary phase check
phase_diff = np.angle(np.vdot(rx0_srs, rx1_srs))
print("Preliminary phase difference [rad]:", phase_diff)
print("Preliminary phase difference [deg]:", np.rad2deg(phase_diff))

# ------------------------------------------------------------
# Plot correlation result
# ------------------------------------------------------------

t_corr_ms = np.arange(len(corr_joint)) / sample_rate * 1000

plt.figure()
plt.plot(t_corr_ms, corr0_norm, label="RX0 correlation")
plt.plot(t_corr_ms, corr1_norm, label="RX1 correlation")
plt.plot(t_corr_ms, corr_joint, label="Joint correlation", linewidth=2)
plt.axvline(peak_joint / sample_rate * 1000, linestyle=":", label="Joint peak")
plt.xlabel("Time [ms]")
plt.ylabel("Normalized correlation magnitude")
plt.title("SRS detection by correlation")
plt.grid(True)
plt.legend()
plt.show()

# ------------------------------------------------------------
# Zoom around peak
# ------------------------------------------------------------

zoom_ms = 2.0
zoom_samps = int(sample_rate * zoom_ms * 1e-3)

z0 = max(0, peak_joint - zoom_samps)
z1 = min(len(corr_joint), peak_joint + zoom_samps)

plt.figure()
plt.plot(t_corr_ms[z0:z1], corr0_norm[z0:z1], label="RX0 correlation")
plt.plot(t_corr_ms[z0:z1], corr1_norm[z0:z1], label="RX1 correlation")
plt.plot(t_corr_ms[z0:z1], corr_joint[z0:z1], label="Joint correlation", linewidth=2)
plt.axvline(peak_joint / sample_rate * 1000, linestyle=":", label="Joint peak")
plt.xlabel("Time [ms]")
plt.ylabel("Normalized correlation magnitude")
plt.title("Zoom around detected SRS")
plt.grid(True)
plt.legend()
plt.show()

# ------------------------------------------------------------
# Save useful data
# ------------------------------------------------------------

np.savez(
    "pluto_srs_correlation_result.npz",
    rx0=rx0,
    rx1=rx1,
    rx0_srs=rx0_srs,
    rx1_srs=rx1_srs,
    local_srs_td=local_srs_td,
    r_freq=r,
    sc_idx=sc_idx,
    sample_rate=sample_rate,
    center_freq=center_freq,
    peak0=peak0,
    peak1=peak1,
    peak_joint=peak_joint,
)
