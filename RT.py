import json
import time
import numpy as np
import adi
from scipy.signal import find_peaks

from srs_config import (
    SystemParams,
    SRSConfigCommon,
    SRSConfigDedicated,
    generateSRSsequence,
)

# ============================================================
# REAL-TIME AOA SETTINGS
# ============================================================

sample_rate = int(5e6)
center_freq = int(886.6e6)
rx_bw = int(3.84e6)

# For real-time:
# 0.100 s gives about 10 SRS peaks because SRS period is 10 ms.
# If Raspberry Pi is slow, change to 0.050.
capture_time = 0.050
num_samps = int(sample_rate * capture_time)

# Local OFDM/SRS settings
N_FFT = 512
CP_LEN = 18

# Peak detection settings
MIN_PEAK_DISTANCE_MS = 8.0
PEAK_HEIGHT_THRESHOLD = 0.5

# Pluto gain
RX_GAIN_CH0 = 30.0
RX_GAIN_CH1 = 30.0

# Correlation mode
# Use False because your latest stable result used USE_LEGACY_CORRELATION=False.
USE_LEGACY_CORRELATION = False

# Calibration file from splitter calibration
CALIBRATION_FILE = "srs_splitter_calibration.json"

# Antenna spacing
# IMPORTANT:
# Replace this with your measured antenna spacing.
# Half wavelength at 886.6 MHz is about 0.16907 m.
c = 299_792_458.0
wavelength = c / center_freq
ANTENNA_SPACING_M = wavelength / 2

# Real-time smoothing
# Larger alpha = faster response, more jumpy.
# Smaller alpha = smoother, slower response.
AOA_SMOOTH_ALPHA = 0.25

# Quality thresholds
MIN_VALID_PEAKS = 3
MIN_PHASE_COHERENCE = 0.80
MIN_PEAK_TO_FLOOR_DB = 15.0

# Optional: print every loop
PRINT_EVERY_LOOP = True

# ============================================================
# LOAD CALIBRATION
# ============================================================

def load_calibration_phase(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    calibration_phase_rad = float(data["calibration_phase_rad"])
    calibration_phase_deg = float(data["calibration_phase_deg"])

    print("============================================================")
    print("LOADED SPLITTER CALIBRATION")
    print("============================================================")
    print("Calibration phase [rad]:", calibration_phase_rad)
    print("Calibration phase [deg]:", calibration_phase_deg)

    if "calibration_coherence" in data:
        print("Calibration coherence R:", data["calibration_coherence"])

    return calibration_phase_rad


calibration_phase_rad = load_calibration_phase(CALIBRATION_FILE)

# ============================================================
# SRS CONFIGURATION FROM SUPERVISOR
# ============================================================

sys_params = SystemParams(
    C_RNTI=0x1234,
    N_RB_UL=15,
    PCID=1,
    sequence_hopping_enabled=0,
    FDD=1,
    group_hopping_enabled=0,
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
    config_Index=7,
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

print()
print("============================================================")
print("SRS CONFIG RESULT")
print("============================================================")
print("Frequency-domain SRS length:", len(r))
print("k0:", k0)
print("First SRS subcarrier index:", sc_idx[0])
print("Last SRS subcarrier index:", sc_idx[-1])
print("T_per [ms]:", T_per)
print("T_off [subframes/ms]:", T_off)
print("Expected SRS peaks per capture:", capture_time * 1000 / T_per)

# ============================================================
# CREATE LOCAL TIME-DOMAIN SRS
# ============================================================

def make_time_domain_srs(r, sc_idx, N_RB_UL, n_fft=512, cp_len=18):
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

local_useful = local_srs_td[CP_LEN : CP_LEN + N_FFT]

print()
print("============================================================")
print("LOCAL TIME-DOMAIN SRS")
print("============================================================")
print("N_FFT:", N_FFT)
print("CP_LEN:", CP_LEN)
print("Full local SRS length CP + symbol:", len(local_srs_td))
print("Useful local SRS length:", len(local_useful))

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def matched_correlate(rx, template):
    """
    np.correlate(a, v) conjugates v internally for complex data.
    Therefore np.correlate(rx, template) is the matched-filter form.
    """
    if USE_LEGACY_CORRELATION:
        return np.correlate(rx, template.conj(), mode="valid")
    return np.correlate(rx, template, mode="valid")


def peak_to_floor_db(corr, peak_index, guard=1000):
    mask = np.ones(len(corr), dtype=bool)

    lo = max(0, peak_index - guard)
    hi = min(len(corr), peak_index + guard)
    mask[lo:hi] = False

    floor = np.median(corr[mask]) + 1e-12
    peak = corr[peak_index] + 1e-12

    return 20 * np.log10(peak / floor)


def circular_weighted_average(phases, weights):
    weights = np.asarray(weights, dtype=float)
    phases = np.asarray(phases, dtype=float)

    if len(phases) == 0 or np.sum(weights) <= 0:
        return np.nan, np.nan

    vector = np.sum(weights * np.exp(1j * phases)) / np.sum(weights)
    avg_phase = np.angle(vector)
    coherence = np.abs(vector)

    return avg_phase, coherence


def wrap_phase_rad(phase):
    return np.angle(np.exp(1j * phase))


def estimate_aoa_from_capture(rx0, rx1):
    """
    Estimate AoA from one short capture.
    Returns a dictionary with phase, corrected phase, AoA, and quality values.
    """

    # Remove DC offset
    rx0 = rx0 - np.mean(rx0)
    rx1 = rx1 - np.mean(rx1)

    rx0_rms = np.sqrt(np.mean(np.abs(rx0) ** 2))
    rx1_rms = np.sqrt(np.mean(np.abs(rx1) ** 2))

    rx0_norm = rx0 / (rx0_rms + 1e-12)
    rx1_norm = rx1 / (rx1_rms + 1e-12)

    # Correlation
    corr0_complex = matched_correlate(rx0_norm, local_srs_td)
    corr1_complex = matched_correlate(rx1_norm, local_srs_td)

    corr0 = np.abs(corr0_complex)
    corr1 = np.abs(corr1_complex)

    corr0_norm = corr0 / (np.max(corr0) + 1e-12)
    corr1_norm = corr1 / (np.max(corr1) + 1e-12)

    corr_joint = corr0_norm + corr1_norm
    corr_joint = corr_joint / (np.max(corr_joint) + 1e-12)

    raw_joint = corr0 + corr1
    raw_peak = int(np.argmax(raw_joint))
    raw_quality_db = peak_to_floor_db(raw_joint, raw_peak)

    # Detect SRS peaks
    min_distance_samples = int(sample_rate * MIN_PEAK_DISTANCE_MS * 1e-3)

    peaks, props = find_peaks(
        corr_joint,
        height=PEAK_HEIGHT_THRESHOLD,
        distance=min_distance_samples,
    )

    phase_diffs = []
    weights = []
    valid_peaks = []
    h0_values = []
    h1_values = []

    for p in peaks:
        start = int(p)

        useful_start = start + CP_LEN
        useful_end = useful_start + N_FFT

        if useful_end > len(rx0):
            continue

        rx0_srs_useful = rx0[useful_start:useful_end]
        rx1_srs_useful = rx1[useful_start:useful_end]

        h0 = np.vdot(local_useful, rx0_srs_useful)
        h1 = np.vdot(local_useful, rx1_srs_useful)

        phase = np.angle(h1 * np.conj(h0))
        weight = np.abs(h0) * np.abs(h1)

        phase_diffs.append(phase)
        weights.append(weight)
        valid_peaks.append(p)
        h0_values.append(h0)
        h1_values.append(h1)

    phase_diffs = np.array(phase_diffs, dtype=float)
    weights = np.array(weights, dtype=float)
    valid_peaks = np.array(valid_peaks, dtype=int)
    h0_values = np.array(h0_values, dtype=np.complex64)
    h1_values = np.array(h1_values, dtype=np.complex64)

    if len(phase_diffs) == 0:
        return {
            "valid": False,
            "reason": "No valid SRS peaks",
            "num_detected_peaks": int(len(peaks)),
            "num_valid_peaks": 0,
            "raw_quality_db": float(raw_quality_db),
        }

    measured_phase_rad, phase_coherence = circular_weighted_average(
        phase_diffs,
        weights,
    )

    corrected_phase_rad = wrap_phase_rad(
        measured_phase_rad - calibration_phase_rad
    )

    # AoA conversion
    arg = corrected_phase_rad * wavelength / (2 * np.pi * ANTENNA_SPACING_M)
    arg_clipped = np.clip(arg, -1.0, 1.0)

    theta_rad = np.arcsin(arg_clipped)
    theta_deg = np.rad2deg(theta_rad)

    phase_deg = np.rad2deg(phase_diffs)

    valid_quality = True
    reason = "OK"

    if len(phase_diffs) < MIN_VALID_PEAKS:
        valid_quality = False
        reason = "Too few valid peaks"

    if phase_coherence < MIN_PHASE_COHERENCE:
        valid_quality = False
        reason = "Low phase coherence"

    if raw_quality_db < MIN_PEAK_TO_FLOOR_DB:
        valid_quality = False
        reason = "Low correlation peak-to-floor"

    return {
        "valid": valid_quality,
        "reason": reason,
        "num_detected_peaks": int(len(peaks)),
        "num_valid_peaks": int(len(phase_diffs)),
        "raw_quality_db": float(raw_quality_db),
        "measured_phase_rad": float(measured_phase_rad),
        "measured_phase_deg": float(np.rad2deg(measured_phase_rad)),
        "corrected_phase_rad": float(corrected_phase_rad),
        "corrected_phase_deg": float(np.rad2deg(corrected_phase_rad)),
        "aoa_deg": float(theta_deg),
        "aoa_rad": float(theta_rad),
        "phase_coherence": float(phase_coherence),
        "phase_std_deg": float(np.std(phase_deg)),
        "phase_min_deg": float(np.min(phase_deg)),
        "phase_max_deg": float(np.max(phase_deg)),
        "rx0_rms": float(rx0_rms),
        "rx1_rms": float(rx1_rms),
        "valid_peaks": valid_peaks,
        "phase_diffs": phase_diffs,
        "h0_values": h0_values,
        "h1_values": h1_values,
    }


# ============================================================
# PLUTO SETUP
# ============================================================

print()
print("============================================================")
print("PLUTO REAL-TIME AOA SETUP")
print("============================================================")
print("Capture duration per update [ms]:", capture_time * 1000)
print("Samples per update:", num_samps)
print("Sample rate:", sample_rate)
print("Center frequency:", center_freq)
print("RX bandwidth:", rx_bw)
print("Wavelength [m]:", wavelength)
print("Antenna spacing [m]:", ANTENNA_SPACING_M)
print("Antenna spacing / wavelength:", ANTENNA_SPACING_M / wavelength)
print("USE_LEGACY_CORRELATION:", USE_LEGACY_CORRELATION)

if ANTENNA_SPACING_M > wavelength / 2:
    print()
    print("WARNING:")
    print("Antenna spacing is larger than lambda/2.")
    print("This can cause AoA ambiguity.")

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
# REAL-TIME LOOP
# ============================================================

print()
print("============================================================")
print("STARTING REAL-TIME AOA")
print("============================================================")
print("Press Ctrl+C to stop.")
print()

aoa_smooth = None
loop_count = 0

try:
    while True:
        loop_start = time.time()

        try:
            sdr.rx_destroy_buffer()
        except Exception:
            pass

        samples = sdr.rx()

        rx0 = np.asarray(samples[0], dtype=np.complex64)
        rx1 = np.asarray(samples[1], dtype=np.complex64)

        result = estimate_aoa_from_capture(rx0, rx1)

        loop_count += 1
        elapsed = time.time() - loop_start
        update_rate_hz = 1.0 / elapsed if elapsed > 0 else 0.0

        if result["valid"]:
            aoa_raw = result["aoa_deg"]

            if aoa_smooth is None:
                aoa_smooth = aoa_raw
            else:
                aoa_smooth = (
                    (1.0 - AOA_SMOOTH_ALPHA) * aoa_smooth
                    + AOA_SMOOTH_ALPHA * aoa_raw
                )

            if PRINT_EVERY_LOOP:
                print(
                    f"[{loop_count:05d}] "
                    f"AoA raw={aoa_raw:+7.2f} deg | "
                    f"AoA smooth={aoa_smooth:+7.2f} deg | "
                    f"meas phase={result['measured_phase_deg']:+7.2f} deg | "
                    f"corr phase={result['corrected_phase_deg']:+7.2f} deg | "
                    f"peaks={result['num_valid_peaks']:02d} | "
                    f"R={result['phase_coherence']:.4f} | "
                    f"Q={result['raw_quality_db']:.1f} dB | "
                    f"rate={update_rate_hz:.2f} Hz"
                )

        else:
            if PRINT_EVERY_LOOP:
                print(
                    f"[{loop_count:05d}] "
                    f"INVALID: {result['reason']} | "
                    f"detected peaks={result.get('num_detected_peaks', 0)} | "
                    f"valid peaks={result.get('num_valid_peaks', 0)} | "
                    f"Q={result.get('raw_quality_db', np.nan):.1f} dB | "
                    f"rate={update_rate_hz:.2f} Hz"
                )

except KeyboardInterrupt:
    print()
    print("Stopped real-time AoA.")
