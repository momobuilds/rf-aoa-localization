import numpy as np
import matplotlib.pyplot as plt

from srs_config import (
    SystemParams,
    SRSConfigCommon,
    SRSConfigDedicated,
    generateSRSsequence,
)


def make_srs_ofdm_symbol(cyclic_shift=0, n_fft=512, cp_len=36):
    """
    Create one time-domain SRS OFDM symbol.

    1. Generate frequency-domain SRS sequence r
    2. Put r on the correct subcarriers
    3. IFFT to time domain
    4. Add cyclic prefix
    """

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

    # Number of occupied uplink subcarriers
    n_used = sys_params.N_RB_UL * 12

    # Frequency-domain OFDM grid, shifted form:
    # index n_fft//2 is DC, left side negative frequencies, right side positive.
    X_shifted = np.zeros(n_fft, dtype=np.complex64)

    # Put the N_RB_UL*12 used subcarriers in the center of the FFT grid
    start = n_fft // 2 - n_used // 2

    # Map SRS subcarrier indexes into FFT bins
    fft_bins = start + sc_idx.astype(int)

    # Put SRS sequence onto its subcarriers
    X_shifted[fft_bins] = r

    # Convert from shifted spectrum to normal IFFT order
    x = np.fft.ifft(np.fft.ifftshift(X_shifted))

    # Normalize power
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))

    # Add cyclic prefix
    x_cp = np.concatenate([x[-cp_len:], x])

    return x_cp.astype(np.complex64), r, k0, sc_idx, T_per, T_off


def normalized_correlation(rx, ref):
    """
    Sliding normalized correlation.

    rx  = received buffer
    ref = local SRS copy

    Output is close to 1 when rx contains ref.
    """

    corr = np.correlate(rx, ref, mode="valid")

    ref_energy = np.sum(np.abs(ref) ** 2)

    rx_energy = np.convolve(
        np.abs(rx) ** 2,
        np.ones(len(ref)),
        mode="valid",
    )

    corr_norm = np.abs(corr) / np.sqrt(ref_energy * rx_energy + 1e-12)

    return corr_norm


# ------------------------------------------------------------
# 1) Generate local SRS time-domain waveform
# ------------------------------------------------------------

local_srs, r, k0, sc_idx, T_per, T_off = make_srs_ofdm_symbol(
    cyclic_shift=0
)

print("SRS sequence length:", len(r))
print("Time-domain SRS length with CP:", len(local_srs))
print("k0:", k0)
print("First SRS subcarriers:", sc_idx[:5])
print("T_per [ms]:", T_per)
print("T_off:", T_off)


# ------------------------------------------------------------
# 2) Create fake received buffer
# ------------------------------------------------------------

np.random.seed(0)

true_delay = 3500
rx_len = 6000
noise_power = 0.05

# Complex noise: I + jQ
rx = np.sqrt(noise_power / 2) * (
    np.random.randn(rx_len) + 1j * np.random.randn(rx_len)
)

# Simulate wireless channel phase
channel_phase = np.exp(1j * 1.2)

# Insert SRS into received buffer
rx[true_delay:true_delay + len(local_srs)] += 0.8 * channel_phase * local_srs


# ------------------------------------------------------------
# 3) Correlate received buffer with local SRS
# ------------------------------------------------------------

metric = normalized_correlation(rx, local_srs)

detected_delay = int(np.argmax(metric))

print("Expected SRS start:", true_delay)
print("Detected SRS start:", detected_delay)
print("Peak correlation:", metric[detected_delay])


# ------------------------------------------------------------
# 4) Plot result
# ------------------------------------------------------------

plt.figure()
plt.plot(metric)
plt.axvline(true_delay, linestyle="--", label="true start")
plt.axvline(detected_delay, linestyle=":", label="detected start")
plt.title("Correlation between RX buffer and local SRS")
plt.xlabel("Sample index")
plt.ylabel("Normalized correlation")
plt.legend()
plt.grid(True)
plt.show()
