import numpy as np
import matplotlib.pyplot as plt
import adi
from scipy.signal import find_peaks

from srs_config import (
    SystemParams,
    SRSConfigCommon,
    SRSConfigDedicated,
    generateSRSsequence,
)

# ============================================================
# USER SETTINGS
# ============================================================

sample_rate = int(5e6)
center_freq = int(886.6e6)
rx_bw = int(3.84e6)

capture_time = 0.050
num_samps = int(sample_rate * capture_time)

# Local OFDM/SRS settings
N_FFT = 512
CP_LEN = 36

# Peak detection settings
MIN_PEAK_DISTANCE_MS = 8.0
PEAK_HEIGHT_THRESHOLD = 0.5

# Pluto gain
RX_GAIN_CH0 = 30.0
RX_GAIN_CH1 = 30.0

# ============================================================
# SRS CONFIGURATION
# ============================================================

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

print("============================================================")
print("SRS CONFIG RESULT")
print("============================================================")
print("Frequency-domain SRS length:", len(r))
print("k0:", k0)
print("First SRS subcarrier index:", sc_idx[0])
print("Last SRS subcarrier index:", sc_idx[-1])
print("T_per [ms]:", T_per)
print("T_off [subframes/ms]:", T_off)

# ============================================================
# CREATE LOCAL TIME-DOMAIN SRS
# ============================================================

def make_time_domain_srs(r, sc_idx, N_RB_UL, n_fft=512, cp_len=36):
    """
    Convert frequency-domain SRS sequence into a time-domain OFDM symbol.
    """

    freq_grid = np.zeros(n_fft, dtype=np.complex64)

    n_used = N_RB_UL * 12
    grid_start = n_fft // 2 - n_used // 2

    fft_bins = grid_start + sc_idx.astype(int)

    if np.any(fft_bins < 0) or np.any(fft_bins >= n_fft):
        raise ValueError("Some SRS subcarriers fall outside the FFT grid.")

    freq_grid[fft_bins] = r

    time_no_cp = np.fft.ifft(np.fft.ifftshift(freq_grid))

    time_with_cp = np.r_[time_no_cp[-cp_len:], time_no_cp]

    time_with_cp = time_with_cp / (
        np.sqrt(np.mean(np.abs(time_with_cp) ** 2)) + 1e-12
    )

    return time_with_cp.astype(np.complex64)


local_srs_td = make_time_domain_srs(
    r=r,
    sc_idx=sc_idx,
    N_RB_UL=sys_params.N_RB_UL,
    n_fft=N_FFT,
    cp_len=CP_LEN,
)

L = len(local_srs_td)

# Useful OFDM symbol only: remove CP
local_useful = local_srs_td[CP_LEN : CP_LEN + N_FFT]

print()
print("============================================================")
print("LOCAL TIME-DOMAIN SRS")
print("============================================================")
print("N_FFT:", N_FFT)
print("CP_LEN:", CP_LEN)
print("Full local SRS length CP + symbol:", L)
print("Useful local SRS length:", len(local_useful))

# ============================================================
# PLUTO SETUP
# ============================================================

print()
print("============================================================")
print("PLUTO CAPTURE SETTINGS")
print("============================================================")
print("Capturing samples:", num_samps)
print("Capture duration [ms]:", 1000 * num_samps / sample_rate)
print("Sample rate:", sample_rate)
print("Center frequency:", center_freq)
print("RX bandwidth:", rx_bw)

sdr = adi.ad9361(uri="ip:192.168.2.1")
sdr.rx_enabled_channels = [0, 1]

sdr.sample_rate = sample_rate
sdr.rx_lo = center_freq
sdr.rx_rf_bandwidth = rx_bw
sdr.rx_buffer_size = num_samps

sdr.gain_control_mode_chan0 = "manual"
sdr.gain_control_mode_chan1 = "manual"
sdr.rx_hardwaregain_chan0 = RX_GAIN_CH0
sdr.rx_hardwaregain_chan1 = RX_GAIN_CH1

try:
    sdr.rx_destroy_buffer()
except Exception:
    pass

# ============================================================
# RECEIVE IQ SAMPLES
# ============================================================

samples = sdr.rx()

rx0 = np.asarray(samples[0], dtype=np.complex64)
rx1 = np.asarray(samples[1], dtype=np.complex64)

rx0 = rx0 - np.mean(rx0)
rx1 = rx1 - np.mean(rx1)

print()
print("============================================================")
print("RECEIVED SIGNAL")
print("============================================================")
print("RX0 length:", len(rx0))
print("RX1 length:", len(rx1))
print("RX0 max abs:", np.max(np.abs(rx0)))
print("RX1 max abs:", np.max(np.abs(rx1)))
print("RX0 RMS:", np.sqrt(np.mean(np.abs(rx0) ** 2)))
print("RX1 RMS:", np.sqrt(np.mean(np.abs(rx1) ** 2)))

# ============================================================
# NORMALIZE RX BEFORE CORRELATION
# ============================================================

rx0_norm = rx0 / (np.sqrt(np.mean(np.abs(rx0) ** 2)) + 1e-12)
rx1_norm = rx1 / (np.sqrt(np.mean(np.abs(rx1) ** 2)) + 1e-12)

# ============================================================
# CORRELATION WITH LOCAL SRS
# ============================================================

print()
print("============================================================")
print("CORRELATING")
print("============================================================")

corr0_complex = np.correlate(rx0_norm, local_srs_td.conj(), mode="valid")
corr1_complex = np.correlate(rx1_norm, local_srs_td.conj(), mode="valid")

corr0 = np.abs(corr0_complex)
corr1 = np.abs(corr1_complex)

corr0_norm = corr0 / (np.max(corr0) + 1e-12)
corr1_norm = corr1 / (np.max(corr1) + 1e-12)

corr_joint = corr0_norm + corr1_norm
corr_joint = corr_joint / (np.max(corr_joint) + 1e-12)

peak0 = int(np.argmax(corr0_norm))
peak1 = int(np.argmax(corr1_norm))
peak_joint = int(np.argmax(corr_joint))

print("Strongest RX0 peak index:", peak0)
print("Strongest RX1 peak index:", peak1)
print("Strongest joint peak index:", peak_joint)
print("Strongest RX0 peak time [ms]:", peak0 / sample_rate * 1000)
print("Strongest RX1 peak time [ms]:", peak1 / sample_rate * 1000)
print("Strongest joint peak time [ms]:", peak_joint / sample_rate * 1000)
print("RX1 - RX0 strongest peak difference [samples]:", peak1 - peak0)

# ============================================================
# DETECT ALL SRS PEAKS
# ============================================================

min_distance_samples = int(sample_rate * MIN_PEAK_DISTANCE_MS * 1e-3)

peaks, props = find_peaks(
    corr_joint,
    height=PEAK_HEIGHT_THRESHOLD,
    distance=min_distance_samples,
)

print()
print("============================================================")
print("ALL DETECTED SRS PEAKS")
print("============================================================")
print("Number of detected SRS peaks:", len(peaks))

for i, p in enumerate(peaks):
    print(
        f"SRS {i+1}: "
        f"sample={p}, "
        f"time={p / sample_rate * 1000:.3f} ms, "
        f"height={props['peak_heights'][i]:.3f}"
    )

# ============================================================
# PHASE DIFFERENCE USING USEFUL OFDM SYMBOL ONLY
# ============================================================

phase_diffs = []
valid_peaks = []
h0_values = []
h1_values = []

print()
print("============================================================")
print("PHASE DIFFERENCE PER DETECTED SRS")
print("USEFUL OFDM SYMBOL ONLY, CP REMOVED")
print("============================================================")

for i, p in enumerate(peaks):
    start = int(p)

    useful_start = start + CP_LEN
    useful_end = useful_start + N_FFT

    if useful_end > len(rx0):
        print(f"SRS {i+1}: skipped because it is too close to buffer end.")
        continue

    rx0_srs_useful = rx0[useful_start:useful_end]
    rx1_srs_useful = rx1[useful_start:useful_end]

    # Estimate complex channel on each RX channel
    h0 = np.vdot(local_useful, rx0_srs_useful)
    h1 = np.vdot(local_useful, rx1_srs_useful)

    # Phase of RX1 relative to RX0
    phase = np.angle(h1 * np.conj(h0))

    phase_diffs.append(phase)
    valid_peaks.append(p)
    h0_values.append(h0)
    h1_values.append(h1)

    print(
        f"SRS {i+1}: "
        f"time={p / sample_rate * 1000:.3f} ms, "
        f"useful_start={useful_start}, "
        f"phase={phase:.5f} rad, "
        f"phase={np.rad2deg(phase):.2f} deg, "
        f"|h0|={np.abs(h0):.3f}, "
        f"|h1|={np.abs(h1):.3f}"
    )

phase_diffs = np.array(phase_diffs, dtype=float)
valid_peaks = np.array(valid_peaks, dtype=int)
h0_values = np.array(h0_values, dtype=np.complex64)
h1_values = np.array(h1_values, dtype=np.complex64)

if len(phase_diffs) > 0:
    avg_phase = np.angle(np.mean(np.exp(1j * phase_diffs)))
    phase_coherence = np.abs(np.mean(np.exp(1j * phase_diffs)))

    print()
    print("============================================================")
    print("AVERAGE PHASE DIFFERENCE")
    print("============================================================")
    print("Valid SRS peaks used:", len(phase_diffs))
    print("Average phase [rad]:", avg_phase)
    print("Average phase [deg]:", np.rad2deg(avg_phase))
    print("Phase coherence R:", phase_coherence)

    if phase_coherence > 0.9:
        print("Phase looks stable.")
    elif phase_coherence > 0.6:
        print("Phase is somewhat stable, but noisy.")
    else:
        print("Phase is unstable. AoA will not be reliable yet.")

else:
    avg_phase = np.nan
    phase_coherence = np.nan
    print("No valid SRS regions found for phase calculation.")

# ============================================================
# AOA INFO FOR LATER
# ============================================================

c = 299_792_458.0
wavelength = c / center_freq

print()
print("============================================================")
print("AOA INFO FOR LATER")
print("============================================================")
print("Wavelength [m]:", wavelength)
print("Half wavelength [m]:", wavelength / 2)

# Uncomment later when you know antenna spacing:
#
# antenna_spacing = wavelength / 2
# arg = avg_phase * wavelength / (2 * np.pi * antenna_spacing)
# arg = np.clip(arg, -1.0, 1.0)
# theta_rad = np.arcsin(arg)
# theta_deg = np.rad2deg(theta_rad)
# print("Estimated AoA [deg]:", theta_deg)

# ============================================================
# PLOT FULL CORRELATION
# ============================================================

t_corr_ms = np.arange(len(corr_joint)) / sample_rate * 1000

plt.figure()
plt.plot(t_corr_ms, corr0_norm, label="RX0 correlation")
plt.plot(t_corr_ms, corr1_norm, label="RX1 correlation")
plt.plot(t_corr_ms, corr_joint, label="Joint correlation", linewidth=2)

for p in peaks:
    plt.axvline(p / sample_rate * 1000, linestyle=":")

plt.xlabel("Time [ms]")
plt.ylabel("Normalized correlation magnitude")
plt.title("SRS detection by correlation")
plt.grid(True)
plt.legend()
plt.show()

# ============================================================
# PLOT ZOOM AROUND STRONGEST JOINT PEAK
# ============================================================

zoom_ms = 2.0
zoom_samps = int(sample_rate * zoom_ms * 1e-3)

z0 = max(0, peak_joint - zoom_samps)
z1 = min(len(corr_joint), peak_joint + zoom_samps)

plt.figure()
plt.plot(t_corr_ms[z0:z1], corr0_norm[z0:z1], label="RX0 correlation")
plt.plot(t_corr_ms[z0:z1], corr1_norm[z0:z1], label="RX1 correlation")
plt.plot(t_corr_ms[z0:z1], corr_joint[z0:z1], label="Joint correlation", linewidth=2)

for p in peaks:
    if z0 <= p < z1:
        plt.axvline(p / sample_rate * 1000, linestyle=":")

plt.xlabel("Time [ms]")
plt.ylabel("Normalized correlation magnitude")
plt.title("Zoom around strongest SRS peak")
plt.grid(True)
plt.legend()
plt.show()

# ============================================================
# PLOT PHASE DIFFERENCE OVER SRS PEAKS
# ============================================================

if len(phase_diffs) > 0:
    valid_peak_times_ms = valid_peaks / sample_rate * 1000

    plt.figure()
    plt.plot(valid_peak_times_ms, np.rad2deg(phase_diffs), marker="o")
    plt.axhline(np.rad2deg(avg_phase), linestyle="--", label="Average phase")
    plt.xlabel("SRS time [ms]")
    plt.ylabel("Phase difference [deg]")
    plt.title("RX1 - RX0 phase difference per detected SRS")
    plt.grid(True)
    plt.legend()
    plt.show()

# ============================================================
# SAVE RESULT
# ============================================================

np.savez(
    "pluto_srs_correlation_and_phase_result_updated.npz",
    rx0=rx0,
    rx1=rx1,
    local_srs_td=local_srs_td,
    local_useful=local_useful,
    r_freq=r,
    sc_idx=sc_idx,
    corr0_norm=corr0_norm,
    corr1_norm=corr1_norm,
    corr_joint=corr_joint,
    peaks=peaks,
    valid_peaks=valid_peaks,
    phase_diffs=phase_diffs,
    avg_phase=avg_phase,
    phase_coherence=phase_coherence,
    h0_values=h0_values,
    h1_values=h1_values,
    sample_rate=sample_rate,
    center_freq=center_freq,
    N_FFT=N_FFT,
    CP_LEN=CP_LEN,
    k0=k0,
    T_per=T_per,
    T_off=T_off,
)

print()
print("Saved result to: pluto_srs_correlation_and_phase_result_updated.npz")
