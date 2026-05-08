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
# Make local SRS OFDM symbol
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Normalized correlation
# ------------------------------------------------------------
def normalized_correlation(rx, ref):
    corr = np.correlate(rx, ref, mode="valid")

    ref_energy = np.sum(np.abs(ref) ** 2)

    rx_energy = np.convolve(
        np.abs(rx) ** 2,
        np.ones(len(ref)),
        mode="valid",
    )

    metric = np.abs(corr) / np.sqrt(ref_energy * rx_energy + 1e-12)

    return metric


# ------------------------------------------------------------
# Pluto RX configuration
# ------------------------------------------------------------
sample_rate = 1_000_000
center_freq = 700_000_000
num_samps = 20000

sdr = adi.Pluto("ip:192.168.2.1")

# For dual RX Pluto
sdr.rx_enabled_channels = [0, 1]

sdr.gain_control_mode_chan0 = "manual"
sdr.gain_control_mode_chan1 = "manual"

sdr.rx_hardwaregain_chan0 = 30.0
sdr.rx_hardwaregain_chan1 = 30.0

sdr.rx_lo = int(center_freq)
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_buffer_size = int(num_samps)

print("Pluto configured")
print("RX channels:", sdr.rx_enabled_channels)
print("Sample rate:", sdr.sample_rate)
print("Center frequency:", sdr.rx_lo)
print("RX buffer size:", sdr.rx_buffer_size)


# ------------------------------------------------------------
# Generate local SRS
# ------------------------------------------------------------
local_srs, r, k0, sc_idx, T_per, T_off = make_srs_ofdm_symbol(
    cyclic_shift=0,
    n_fft=512,
    cp_len=36,
)

print("Local SRS length:", len(local_srs))
print("SRS sequence length:", len(r))
print("k0:", k0)
print("First subcarriers:", sc_idx[:5])
print("T_per:", T_per)
print("T_off:", T_off)


# ------------------------------------------------------------
# Receive samples from Pluto
# ------------------------------------------------------------
samples = sdr.rx()

rx0 = samples[0]
rx1 = samples[1]

print("Received RX0 samples:", len(rx0))
print("Received RX1 samples:", len(rx1))
print("RX0 first samples:", rx0[:5])
print("RX1 first samples:", rx1[:5])


# ------------------------------------------------------------
# Correlate RX0 and RX1 with local SRS
# ------------------------------------------------------------
metric0 = normalized_correlation(rx0, local_srs)
metric1 = normalized_correlation(rx1, local_srs)

peak0 = int(np.argmax(metric0))
peak1 = int(np.argmax(metric1))

print("\nRX0 peak index:", peak0)
print("RX0 peak value:", metric0[peak0])

print("\nRX1 peak index:", peak1)
print("RX1 peak value:", metric1[peak1])


# ------------------------------------------------------------
# Plot correlation
# ------------------------------------------------------------
plt.figure()
plt.plot(metric0, label="RX0 correlation")
plt.plot(metric1, label="RX1 correlation")
plt.title("Correlation between Pluto RX samples and local SRS")
plt.xlabel("Sample index")
plt.ylabel("Normalized correlation")
plt.grid(True)
plt.legend()
plt.show()
