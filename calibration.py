import json
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
# CALIBRATION SETTINGS
# ============================================================

sample_rate = int(5e6)
center_freq = int(886.6e6)
rx_bw = int(3.84e6)

capture_time = 1.0
num_samps = int(sample_rate * capture_time)

# Run several 1-second captures for a stronger calibration
NUM_CALIBRATION_RUNS = 3

# Local OFDM/SRS settings
N_FFT = 512
CP_LEN = 18

# Peak detection settings
MIN_PEAK_DISTANCE_MS = 8.0
PEAK_HEIGHT_THRESHOLD = 0.5

# Pluto gain
RX_GAIN_CH0 = 30.0
RX_GAIN_CH1 = 30.0

# Correct NumPy matched-filter convention
USE_LEGACY_CORRELATION = False

CALIBRATION_OUTPUT_NPZ = "srs_splitter_calibration.npz"
CALIBRATION_OUTPUT_JSON = "srs_splitter_calibration.json"

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

print("============================================================")
print("SPLITTER CALIBRATION MODE")
print("============================================================")
print("The splitter should feed the same signal to RX0 and RX1.")
print("True phase difference should be 0 degrees.")
print("Measured phase will be saved as hardware calibration phase.")
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

total_subcarriers = sys_params.N_RB_UL * 12
effective_scs = sample_rate / N_FFT

print("Total active subcarriers:", total_subcarriers)
print("Effective subcarrier spacing from sample_rate / N_FFT [kHz]:", effective_scs / 1e3)

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
    vector = np.sum(weights * np.exp(1j * phases)) / (np.sum(weights) + 1e-12)
    avg_phase = np.angle(vector)
    coherence = np.abs(vector)
    return avg_phase, coherence


# ============================================================
# PLUTO SETUP
# ============================================================

print()
print("============================================================")
print("PLUTO CAPTURE SETTINGS")
print("============================================================")
print("Capture duration per run [ms]:", capture_time * 1000)
print("Number of calibration runs:", NUM_CALIBRATION_RUNS)
print("Samples per run:", num_samps)
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
# ONE CALIBRATION CAPTURE
# ============================================================

def run_one_calibration_capture(run_index):
    print()
    print("============================================================")
    print(f"CALIBRATION RUN {run_index}")
    print("============================================================")

    try:
        sdr.rx_destroy_buffer()
    except Exception:
        pass

    samples = sdr.rx()

    rx0 = np.asarray(samples[0], dtype=np.complex64)
    rx1 = np.asarray(samples[1], dtype=np.complex64)

    # Remove DC offset
    rx0 = rx0 - np.mean(rx0)
    rx1 = rx1 - np.mean(rx1)

    rx0_rms = np.sqrt(np.mean(np.abs(rx0) ** 2))
    rx1_rms = np.sqrt(np.mean(np.abs(rx1) ** 2))

    print("RX0 length:", len(rx0))
    print("RX1 length:", len(rx1))
    print("RX0 max abs:", np.max(np.abs(rx0)))
    print("RX1 max abs:", np.max(np.abs(rx1)))
    print("RX0 RMS:", rx0_rms)
    print("RX1 RMS:", rx1_rms)

    rx0_norm = rx0 / (rx0_rms + 1e-12)
    rx1_norm = rx1 / (rx1_rms + 1e-12)

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

    print("Raw joint peak index:", raw_peak)
    print("Raw joint peak time [ms]:", raw_peak / sample_rate * 1000)
    print("Raw joint peak-to-floor [dB]:", raw_quality_db)

    min_distance_samples = int(sample_rate * MIN_PEAK_DISTANCE_MS * 1e-3)

    peaks, props = find_peaks(
        corr_joint,
        height=PEAK_HEIGHT_THRESHOLD,
        distance=min_distance_samples,
    )

    print("Detected SRS peaks:", len(peaks))

    if T_per is not None:
        expected_num_peaks = int(round((capture_time * 1000) / T_per))
        print("Expected approximate SRS peaks:", expected_num_peaks)

    if len(peaks) > 1:
        intervals_ms = np.diff(peaks) / sample_rate * 1000
        print("Peak interval mean [ms]:", np.mean(intervals_ms))
        print("Peak interval std [ms]:", np.std(intervals_ms))
        print("Peak interval min [ms]:", np.min(intervals_ms))
        print("Peak interval max [ms]:", np.max(intervals_ms))

    phase_diffs = []
    weights = []
    h0_values = []
    h1_values = []
    valid_peaks = []

    for i, p in enumerate(peaks):
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
        h0_values.append(h0)
        h1_values.append(h1)
        valid_peaks.append(p)

    phase_diffs = np.array(phase_diffs, dtype=float)
    weights = np.array(weights, dtype=float)
    h0_values = np.array(h0_values, dtype=np.complex64)
    h1_values = np.array(h1_values, dtype=np.complex64)
    valid_peaks = np.array(valid_peaks, dtype=int)

    if len(phase_diffs) == 0:
        raise RuntimeError("No valid SRS regions found for calibration phase calculation.")

    avg_phase = np.angle(np.mean(np.exp(1j * phase_diffs)))
    phase_coherence = np.abs(np.mean(np.exp(1j * phase_diffs)))

    avg_phase_weighted, phase_coherence_weighted = circular_weighted_average(
        phase_diffs,
        weights,
    )

    phase_deg = np.rad2deg(phase_diffs)

    print()
    print("PHASE RESULT FOR THIS RUN")
    print("Valid SRS peaks used:", len(phase_diffs))
    print("Circular average phase [rad]:", avg_phase)
    print("Circular average phase [deg]:", np.rad2deg(avg_phase))
    print("Phase coherence R:", phase_coherence)
    print("Weighted average phase [rad]:", avg_phase_weighted)
    print("Weighted average phase [deg]:", np.rad2deg(avg_phase_weighted))
    print("Weighted phase coherence R:", phase_coherence_weighted)
    print("Phase standard deviation [deg]:", np.std(phase_deg))
    print("Phase min [deg]:", np.min(phase_deg))
    print("Phase max [deg]:", np.max(phase_deg))
    print("Phase range [deg]:", np.max(phase_deg) - np.min(phase_deg))

    return {
        "run_index": run_index,
        "num_detected_peaks": int(len(peaks)),
        "num_valid_phase_peaks": int(len(phase_diffs)),
        "raw_quality_db": float(raw_quality_db),
        "avg_phase_rad": float(avg_phase),
        "avg_phase_deg": float(np.rad2deg(avg_phase)),
        "phase_coherence": float(phase_coherence),
        "avg_phase_weighted_rad": float(avg_phase_weighted),
        "avg_phase_weighted_deg": float(np.rad2deg(avg_phase_weighted)),
        "phase_coherence_weighted": float(phase_coherence_weighted),
        "phase_std_deg": float(np.std(phase_deg)),
        "phase_min_deg": float(np.min(phase_deg)),
        "phase_max_deg": float(np.max(phase_deg)),
        "phase_range_deg": float(np.max(phase_deg) - np.min(phase_deg)),
        "phase_diffs": phase_diffs,
        "weights": weights,
        "valid_peaks": valid_peaks,
        "h0_values": h0_values,
        "h1_values": h1_values,
    }


# ============================================================
# RUN CALIBRATION
# ============================================================

all_phase_diffs = []
all_weights = []
all_run_results = []

for run_index in range(1, NUM_CALIBRATION_RUNS + 1):
    result = run_one_calibration_capture(run_index)

    all_run_results.append(result)
    all_phase_diffs.append(result["phase_diffs"])
    all_weights.append(result["weights"])

all_phase_diffs = np.concatenate(all_phase_diffs)
all_weights = np.concatenate(all_weights)

calibration_phase_rad, calibration_coherence = circular_weighted_average(
    all_phase_diffs,
    all_weights,
)

calibration_phase_deg = np.rad2deg(calibration_phase_rad)

all_phase_deg = np.rad2deg(all_phase_diffs)

print()
print("============================================================")
print("FINAL SPLITTER CALIBRATION RESULT")
print("============================================================")
print("Total SRS phase measurements used:", len(all_phase_diffs))
print("Calibration phase [rad]:", calibration_phase_rad)
print("Calibration phase [deg]:", calibration_phase_deg)
print("Calibration coherence R:", calibration_coherence)
print("All phase standard deviation [deg]:", np.std(all_phase_deg))
print("All phase min [deg]:", np.min(all_phase_deg))
print("All phase max [deg]:", np.max(all_phase_deg))
print("All phase range [deg]:", np.max(all_phase_deg) - np.min(all_phase_deg))

print()
print("Use this value later as:")
print("Delta_phi_cal =", calibration_phase_rad, "rad")
print("Delta_phi_cal =", calibration_phase_deg, "deg")

# ============================================================
# SAVE CALIBRATION
# ============================================================

run_summary = []

for result in all_run_results:
    run_summary.append({
        "run_index": result["run_index"],
        "num_detected_peaks": result["num_detected_peaks"],
        "num_valid_phase_peaks": result["num_valid_phase_peaks"],
        "raw_quality_db": result["raw_quality_db"],
        "avg_phase_rad": result["avg_phase_rad"],
        "avg_phase_deg": result["avg_phase_deg"],
        "phase_coherence": result["phase_coherence"],
        "avg_phase_weighted_rad": result["avg_phase_weighted_rad"],
        "avg_phase_weighted_deg": result["avg_phase_weighted_deg"],
        "phase_coherence_weighted": result["phase_coherence_weighted"],
        "phase_std_deg": result["phase_std_deg"],
        "phase_min_deg": result["phase_min_deg"],
        "phase_max_deg": result["phase_max_deg"],
        "phase_range_deg": result["phase_range_deg"],
    })

np.savez(
    CALIBRATION_OUTPUT_NPZ,
    calibration_phase_rad=calibration_phase_rad,
    calibration_phase_deg=calibration_phase_deg,
    calibration_coherence=calibration_coherence,
    all_phase_diffs=all_phase_diffs,
    all_weights=all_weights,
    all_phase_deg=all_phase_deg,
    sample_rate=sample_rate,
    center_freq=center_freq,
    rx_bw=rx_bw,
    capture_time=capture_time,
    NUM_CALIBRATION_RUNS=NUM_CALIBRATION_RUNS,
    N_FFT=N_FFT,
    CP_LEN=CP_LEN,
    k0=k0,
    sc_idx=sc_idx,
    T_per=T_per,
    T_off=T_off,
    N_RB_UL=sys_params.N_RB_UL,
    PCID=sys_params.PCID,
    C_RNTI=sys_params.C_RNTI,
    config_Index=cfg_ded.config_Index,
    USE_LEGACY_CORRELATION=USE_LEGACY_CORRELATION,
)

json_data = {
    "calibration_phase_rad": float(calibration_phase_rad),
    "calibration_phase_deg": float(calibration_phase_deg),
    "calibration_coherence": float(calibration_coherence),
    "all_phase_std_deg": float(np.std(all_phase_deg)),
    "all_phase_min_deg": float(np.min(all_phase_deg)),
    "all_phase_max_deg": float(np.max(all_phase_deg)),
    "all_phase_range_deg": float(np.max(all_phase_deg) - np.min(all_phase_deg)),
    "total_phase_measurements": int(len(all_phase_diffs)),
    "sample_rate": int(sample_rate),
    "center_freq": int(center_freq),
    "rx_bw": int(rx_bw),
    "capture_time": float(capture_time),
    "num_calibration_runs": int(NUM_CALIBRATION_RUNS),
    "N_FFT": int(N_FFT),
    "CP_LEN": int(CP_LEN),
    "T_per": int(T_per) if T_per is not None else None,
    "T_off": int(T_off) if T_off is not None else None,
    "N_RB_UL": int(sys_params.N_RB_UL),
    "PCID": int(sys_params.PCID),
    "C_RNTI": int(sys_params.C_RNTI),
    "config_Index": int(cfg_ded.config_Index),
    "run_summary": run_summary,
}

with open(CALIBRATION_OUTPUT_JSON, "w") as f:
    json.dump(json_data, f, indent=4)

print()
print("Saved calibration NPZ to:", CALIBRATION_OUTPUT_NPZ)
print("Saved calibration JSON to:", CALIBRATION_OUTPUT_JSON)
