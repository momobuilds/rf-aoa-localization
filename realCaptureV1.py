import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import adi

from srs_config import (
    SystemParams,
    SRSConfigCommon,
    SRSConfigDedicated,
    generateSRSsequence,
)


def make_srs_ofdm_symbol(cyclic_shift=0, n_fft=512, cp_len=36):
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
        cyclicShift=cyclic_shift,
        srs_AntennaPort=10,
        transmissionCombNum=2,
    )

    r, k0, sc_idx, T_per, T_off = generateSRSsequence(
        cfg_ded,
        cfg_com,
        sys_params,
    )

    n_used = sys_params.N_RB_UL * 12

    X_shifted = np.zeros(n_fft, dtype=np.complex64)
    start = n_fft // 2 - n_used // 2
    fft_bins = start + sc_idx.astype(int)

    X_shifted[fft_bins] = r

    x = np.fft.ifft(np.fft.ifftshift(X_shifted))
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))

    x_cp = np.concatenate([x[-cp_len:], x])

    return x_cp.astype(np.complex64), r, k0, sc_idx, T_per, T_off


def complex_normalized_correlation(rx, ref):
    """
    Returns:
      corr_complex: complex correlation, keeps phase
      metric: normalized magnitude, useful for detection
    """

    rx = np.asarray(rx, dtype=np.complex64)
    ref = np.asarray(ref, dtype=np.complex64)

    corr_complex = correlate(rx, ref, mode="valid", method="fft")

    ref_energy = np.sum(np.abs(ref) ** 2)

    rx_energy = np.convolve(
        np.abs(rx) ** 2,
        np.ones(len(ref)),
        mode="valid",
    )

    metric = np.abs(corr_complex) / np.sqrt(ref_energy * rx_energy + 1e-12)

    return corr_complex, metric


# ------------------------------------------------------------
# 1) Generate local SRS
# ------------------------------------------------------------

local_srs, r, k0, sc_idx, T_per, T_off = make_srs_ofdm_symbol(
    cyclic_shift=0,
    n_fft=512,
    cp_len=36,
)

print("SRS sequence length:", len(r))
print("OFDM SRS length with CP:", len(local_srs))
print("k0:", k0)
print("T_per [ms]:", T_per)
print("T_off:", T_off)


# ------------------------------------------------------------
# 2) Configure real Pluto
# ------------------------------------------------------------

sample_rate = int(3e6)        # Use this only if "3 MHz" means sample rate
center_freq = int(700e6)      # Replace this with the actual RF center frequency
rx_bw = int(3e6)

rx_buffer_size = 2**18        # 262144 samples. Much better than 20.

# For many 2RX AD9361-type setups, adi.ad9361 is safer than adi.Pluto.
# If this does not work on your board, try: sdr = adi.Pluto("ip:192.168.2.1")
sdr = adi.ad9361(uri="ip:192.168.2.1")

sdr.rx_enabled_channels = [0, 1]

sdr.sample_rate = sample_rate
sdr.rx_lo = center_freq
sdr.rx_rf_bandwidth = rx_bw
sdr.rx_buffer_size = rx_buffer_size

sdr.gain_control_mode_chan0 = "manual"
sdr.gain_control_mode_chan1 = "manual"

sdr.rx_hardwaregain_chan0 = 30.0
sdr.rx_hardwaregain_chan1 = 30.0

# Optional: remove old buffers
try:
    sdr.rx_destroy_buffer()
except Exception:
    pass


# ------------------------------------------------------------
# 3) Receive both channels synchronously
# ------------------------------------------------------------

samples = sdr.rx()

rx0 = np.asarray(samples[0], dtype=np.complex64)
rx1 = np.asarray(samples[1], dtype=np.complex64)

rx0 = rx0 - np.mean(rx0)
rx1 = rx1 - np.mean(rx1)

print("RX0 length:", len(rx0))
print("RX1 length:", len(rx1))


# ------------------------------------------------------------
# 4) Correlate both channels with same local SRS
# ------------------------------------------------------------

corr0, metric0 = complex_normalized_correlation(rx0, local_srs)
corr1, metric1 = complex_normalized_correlation(rx1, local_srs)

joint_metric = metric0 + metric1
peak = int(np.argmax(joint_metric))

print("Detected SRS start:", peak)
print("RX0 peak metric:", metric0[peak])
print("RX1 peak metric:", metric1[peak])
print("Joint peak metric:", joint_metric[peak])


# ------------------------------------------------------------
# 5) Phase difference between RX channels
# ------------------------------------------------------------

phase0 = np.angle(corr0[peak])
phase1 = np.angle(corr1[peak])

phase_diff = np.angle(corr1[peak] * np.conj(corr0[peak]))

print("RX0 correlation phase:", phase0)
print("RX1 correlation phase:", phase1)
print("Raw phase difference RX1 - RX0:", phase_diff)


# ------------------------------------------------------------
# 6) Plot detection result
# ------------------------------------------------------------
def print_detection_stats(name, metric):
    peak = int(np.argmax(metric))
    peak_value = metric[peak]
    median_value = np.median(metric)
    mean_value = np.mean(metric)

    ratio_to_median_db = 20 * np.log10(
        (peak_value + 1e-12) / (median_value + 1e-12)
    )

    print(f"\n{name}")
    print("Peak index:", peak)
    print("Peak value:", peak_value)
    print("Median:", median_value)
    print("Mean:", mean_value)
    print("Peak / median [dB]:", ratio_to_median_db)

    return peak


joint_metric = 0.5 * (metric0 + metric1)

peak0 = print_detection_stats("RX0", metric0)
peak1 = print_detection_stats("RX1", metric1)
peak_joint = print_detection_stats("Joint", joint_metric)

plt.figure()
plt.plot(metric0)
plt.axvline(peak0, linestyle="--")
plt.title("RX0 SRS correlation")
plt.xlabel("Sample index")
plt.ylabel("Normalized correlation")
plt.grid(True)

plt.figure()
plt.plot(metric1)
plt.axvline(peak1, linestyle="--")
plt.title("RX1 SRS correlation")
plt.xlabel("Sample index")
plt.ylabel("Normalized correlation")
plt.grid(True)

plt.figure()
plt.plot(joint_metric)
plt.axvline(peak_joint, linestyle="--")
plt.title("Joint SRS correlation")
plt.xlabel("Sample index")
plt.ylabel("Normalized correlation")
plt.grid(True)

plt.show()
