import json
import time
import math
import threading
import queue
import tkinter as tk

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
# USER SETTINGS
# ============================================================

sample_rate = int(5e6)
center_freq = int(886.6e6)
rx_bw = int(3.84e6)

# Faster real-time update
# 50 ms gives about 5 SRS peaks because SRS period is 10 ms.
capture_time = 0.050
num_samps = int(sample_rate * capture_time)

# Local OFDM / SRS settings
N_FFT = 512
CP_LEN = 18

# Peak detection settings
MIN_PEAK_DISTANCE_MS = 8.0
PEAK_HEIGHT_THRESHOLD = 0.5

# Pluto gain
RX_GAIN_CH0 = 30.0
RX_GAIN_CH1 = 30.0

# Your latest good result used False
USE_LEGACY_CORRELATION = False

# Calibration file from splitter calibration
CALIBRATION_FILE = "srs_splitter_calibration.json"

# Antenna spacing
c = 299_792_458.0
wavelength = c / center_freq

# Replace this with measured antenna spacing if different
ANTENNA_SPACING_M = wavelength / 2

# Quality thresholds
MIN_VALID_PEAKS = 3
MIN_PHASE_COHERENCE = 0.80
MIN_PEAK_TO_FLOOR_DB = 15.0

# UI smoothing
AOA_SMOOTH_ALPHA = 0.35

# UI refresh time
UI_REFRESH_MS = 200

# Arrow display limit
MAX_DISPLAY_ANGLE_DEG = 60.0


# ============================================================
# LOAD CALIBRATION
# ============================================================

def load_calibration_phase(filename):
    try:
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

        print()
        return calibration_phase_rad

    except Exception as e:
        print("============================================================")
        print("WARNING")
        print("============================================================")
        print("Could not load calibration file:", filename)
        print("Reason:", str(e))
        print("Using calibration phase = 0 rad")
        print()
        return 0.0


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
print("Wavelength [m]:", wavelength)
print("Antenna spacing [m]:", ANTENNA_SPACING_M)
print()


# ============================================================
# LOCAL TIME-DOMAIN SRS
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

local_useful = local_srs_td[CP_LEN:CP_LEN + N_FFT]

print("============================================================")
print("LOCAL TIME-DOMAIN SRS")
print("============================================================")
print("N_FFT:", N_FFT)
print("CP_LEN:", CP_LEN)
print("Full local SRS length:", len(local_srs_td))
print("Useful local SRS length:", len(local_useful))
print()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def matched_correlate(rx, template):
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
    phases = np.asarray(phases, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if len(phases) == 0 or np.sum(weights) <= 0:
        return np.nan, np.nan

    vector = np.sum(weights * np.exp(1j * phases)) / np.sum(weights)
    avg_phase = np.angle(vector)
    coherence = np.abs(vector)

    return avg_phase, coherence


def wrap_phase_rad(phase):
    return np.angle(np.exp(1j * phase))


def estimate_aoa_from_capture(rx0, rx1):
    # Remove DC offset
    rx0 = rx0 - np.mean(rx0)
    rx1 = rx1 - np.mean(rx1)

    rx0_rms = np.sqrt(np.mean(np.abs(rx0) ** 2))
    rx1_rms = np.sqrt(np.mean(np.abs(rx1) ** 2))

    rx0_norm = rx0 / (rx0_rms + 1e-12)
    rx1_norm = rx1 / (rx1_rms + 1e-12)

    # Correlate with local SRS
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

    phase_diffs = np.array(phase_diffs, dtype=float)
    weights = np.array(weights, dtype=float)

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

    arg = corrected_phase_rad * wavelength / (2 * np.pi * ANTENNA_SPACING_M)
    arg = np.clip(arg, -1.0, 1.0)

    theta_rad = np.arcsin(arg)
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
        reason = "Low correlation quality"

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
        "rx0_rms": float(rx0_rms),
        "rx1_rms": float(rx1_rms),
    }


# ============================================================
# PLUTO SETUP
# ============================================================

print("============================================================")
print("PLUTO SETUP")
print("============================================================")
print("Capture duration [ms]:", capture_time * 1000)
print("Samples:", num_samps)
print("Sample rate:", sample_rate)
print("Center freq:", center_freq)
print("RX bandwidth:", rx_bw)
print("USE_LEGACY_CORRELATION:", USE_LEGACY_CORRELATION)
print()

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

# Destroy once before starting, not every loop
try:
    sdr.rx_destroy_buffer()
except Exception:
    pass


# ============================================================
# BACKGROUND AOA WORKER
# ============================================================

class AoAWorker(threading.Thread):
    def __init__(self, result_queue, stop_event):
        super().__init__(daemon=True)
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.aoa_smooth = None
        self.loop_count = 0

    def push_result(self, result):
        # Keep only newest result
        try:
            while True:
                self.result_queue.get_nowait()
        except queue.Empty:
            pass

        self.result_queue.put(result)

    def run(self):
        while not self.stop_event.is_set():
            loop_start = time.time()

            # Do NOT destroy the RX buffer every loop.
            # This makes real-time mode much faster.
            samples = sdr.rx()

            rx0 = np.asarray(samples[0], dtype=np.complex64)
            rx1 = np.asarray(samples[1], dtype=np.complex64)

            result = estimate_aoa_from_capture(rx0, rx1)

            self.loop_count += 1

            if result["valid"]:
                aoa_raw = result["aoa_deg"]

                if self.aoa_smooth is None:
                    self.aoa_smooth = aoa_raw
                else:
                    self.aoa_smooth = (
                        (1.0 - AOA_SMOOTH_ALPHA) * self.aoa_smooth
                        + AOA_SMOOTH_ALPHA * aoa_raw
                    )

                result["aoa_smooth_deg"] = float(self.aoa_smooth)
            else:
                result["aoa_smooth_deg"] = None

            elapsed = time.time() - loop_start
            result["update_rate_hz"] = 1.0 / elapsed if elapsed > 0 else 0.0
            result["loop_count"] = self.loop_count

            self.push_result(result)


# ============================================================
# TKINTER UI
# ============================================================

class AoAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AoA Direction Finder")
        self.root.geometry("900x700")
        self.root.configure(bg="#0b1220")

        self.result_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.worker = AoAWorker(self.result_queue, self.stop_event)

        self.build_ui()
        self.worker.start()

        self.root.after(UI_REFRESH_MS, self.update_ui)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_ui(self):
        title = tk.Label(
            self.root,
            text="AoA Direction Finder",
            font=("Helvetica", 26, "bold"),
            fg="white",
            bg="#0b1220",
        )
        title.pack(pady=12)

        subtitle = tk.Label(
            self.root,
            text="Fast real-time arrow UI using PlutoSDR SRS AoA",
            font=("Helvetica", 12),
            fg="#9fb3c8",
            bg="#0b1220",
        )
        subtitle.pack()

        main_frame = tk.Frame(self.root, bg="#0b1220")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        left_frame = tk.Frame(main_frame, bg="#0b1220")
        left_frame.pack(side="left", fill="both", expand=True)

        right_frame = tk.Frame(main_frame, bg="#0b1220")
        right_frame.pack(side="right", fill="y", padx=20)

        self.canvas_size = 500
        self.canvas = tk.Canvas(
            left_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="#111827",
            highlightthickness=0,
        )
        self.canvas.pack(pady=10)

        self.cx = self.canvas_size / 2
        self.cy = self.canvas_size / 2
        self.radius = 180

        self.draw_static_gauge()

        self.status_label = tk.Label(
            left_frame,
            text="WAITING FOR SIGNAL",
            font=("Helvetica", 22, "bold"),
            fg="#fbbf24",
            bg="#0b1220",
        )
        self.status_label.pack(pady=10)

        self.angle_label = tk.Label(
            left_frame,
            text="AoA: --.-°",
            font=("Helvetica", 28, "bold"),
            fg="white",
            bg="#0b1220",
        )
        self.angle_label.pack()

        self.info_vars = {
            "Smoothed AoA": tk.StringVar(value="--"),
            "Raw AoA": tk.StringVar(value="--"),
            "Corrected Phase": tk.StringVar(value="--"),
            "Measured Phase": tk.StringVar(value="--"),
            "Coherence R": tk.StringVar(value="--"),
            "Valid Peaks": tk.StringVar(value="--"),
            "Quality [dB]": tk.StringVar(value="--"),
            "Update Rate": tk.StringVar(value="--"),
            "Status": tk.StringVar(value="Starting..."),
        }

        for key, var in self.info_vars.items():
            self.make_info_card(right_frame, key, var)

        footer = tk.Label(
            self.root,
            text="0° = straight ahead | + angle = right | - angle = left",
            font=("Helvetica", 11),
            fg="#9fb3c8",
            bg="#0b1220",
        )
        footer.pack(pady=8)

    def make_info_card(self, parent, title, variable):
        frame = tk.Frame(parent, bg="#111827", bd=0, relief="flat")
        frame.pack(fill="x", pady=6)

        label_title = tk.Label(
            frame,
            text=title,
            font=("Helvetica", 11, "bold"),
            fg="#93c5fd",
            bg="#111827",
            anchor="w",
            width=16,
        )
        label_title.pack(side="left", padx=10, pady=10)

        label_value = tk.Label(
            frame,
            textvariable=variable,
            font=("Helvetica", 11),
            fg="white",
            bg="#111827",
            anchor="e",
            width=16,
        )
        label_value.pack(side="right", padx=10, pady=10)

    def draw_static_gauge(self):
        self.canvas.delete("all")

        # Outer circle
        self.canvas.create_oval(
            self.cx - self.radius,
            self.cy - self.radius,
            self.cx + self.radius,
            self.cy + self.radius,
            outline="#334155",
            width=3,
        )

        # Center dot
        self.canvas.create_oval(
            self.cx - 12,
            self.cy - 12,
            self.cx + 12,
            self.cy + 12,
            fill="#38bdf8",
            outline="",
        )

        # Main labels
        self.canvas.create_text(
            self.cx,
            self.cy - self.radius - 25,
            text="0°",
            fill="white",
            font=("Helvetica", 12, "bold"),
        )

        self.canvas.create_text(
            self.cx - self.radius - 25,
            self.cy,
            text="-90°",
            fill="#9fb3c8",
            font=("Helvetica", 10),
        )

        self.canvas.create_text(
            self.cx + self.radius + 25,
            self.cy,
            text="+90°",
            fill="#9fb3c8",
            font=("Helvetica", 10),
        )

        # Center guide line
        self.canvas.create_line(
            self.cx,
            self.cy,
            self.cx,
            self.cy - self.radius,
            fill="#1f2937",
            dash=(4, 4),
            width=2,
        )

        # Tick marks
        for deg in range(-60, 61, 15):
            theta = math.radians(deg)

            x1 = self.cx + (self.radius - 10) * math.sin(theta)
            y1 = self.cy - (self.radius - 10) * math.cos(theta)

            x2 = self.cx + (self.radius + 5) * math.sin(theta)
            y2 = self.cy - (self.radius + 5) * math.cos(theta)

            self.canvas.create_line(x1, y1, x2, y2, fill="#475569", width=2)

        # Arrow
        self.arrow = self.canvas.create_line(
            self.cx,
            self.cy,
            self.cx,
            self.cy - self.radius + 30,
            fill="#22c55e",
            width=10,
            arrow=tk.LAST,
            arrowshape=(18, 22, 8),
            capstyle=tk.ROUND,
        )

    def update_arrow(self, angle_deg):
        display_angle = max(
            -MAX_DISPLAY_ANGLE_DEG,
            min(MAX_DISPLAY_ANGLE_DEG, angle_deg),
        )

        theta = math.radians(display_angle)
        length = self.radius - 30

        x = self.cx + length * math.sin(theta)
        y = self.cy - length * math.cos(theta)

        self.canvas.coords(self.arrow, self.cx, self.cy, x, y)

        if abs(display_angle) < 5:
            color = "#22c55e"
        elif abs(display_angle) < 20:
            color = "#f59e0b"
        else:
            color = "#ef4444"

        self.canvas.itemconfig(self.arrow, fill=color)

    def angle_to_status(self, angle_deg):
        if angle_deg is None:
            return "WAITING FOR SIGNAL", "#fbbf24"

        if abs(angle_deg) < 5:
            return "ON TARGET", "#22c55e"
        elif angle_deg > 0:
            return "MOVE RIGHT", "#38bdf8"
        else:
            return "MOVE LEFT", "#38bdf8"

    def update_ui(self):
        latest = None

        try:
            while True:
                latest = self.result_queue.get_nowait()
        except queue.Empty:
            pass

        if latest is not None:
            if latest["valid"]:
                aoa_smooth = latest["aoa_smooth_deg"]
                aoa_raw = latest["aoa_deg"]
                corrected_phase = latest["corrected_phase_deg"]
                measured_phase = latest["measured_phase_deg"]
                coherence = latest["phase_coherence"]
                peaks = latest["num_valid_peaks"]
                quality = latest["raw_quality_db"]
                rate = latest["update_rate_hz"]

                self.update_arrow(aoa_smooth)

                status_text, status_color = self.angle_to_status(aoa_smooth)
                self.status_label.config(text=status_text, fg=status_color)
                self.angle_label.config(text=f"AoA: {aoa_smooth:+.1f}°")

                self.info_vars["Smoothed AoA"].set(f"{aoa_smooth:+.2f}°")
                self.info_vars["Raw AoA"].set(f"{aoa_raw:+.2f}°")
                self.info_vars["Corrected Phase"].set(f"{corrected_phase:+.2f}°")
                self.info_vars["Measured Phase"].set(f"{measured_phase:+.2f}°")
                self.info_vars["Coherence R"].set(f"{coherence:.4f}")
                self.info_vars["Valid Peaks"].set(f"{peaks}")
                self.info_vars["Quality [dB]"].set(f"{quality:.2f}")
                self.info_vars["Update Rate"].set(f"{rate:.2f} Hz")
                self.info_vars["Status"].set("Tracking")

            else:
                self.status_label.config(text="NO RELIABLE SIGNAL", fg="#ef4444")
                self.angle_label.config(text="AoA: --.-°")
                self.update_arrow(0)

                self.info_vars["Smoothed AoA"].set("--")
                self.info_vars["Raw AoA"].set("--")
                self.info_vars["Corrected Phase"].set("--")
                self.info_vars["Measured Phase"].set("--")
                self.info_vars["Coherence R"].set("--")
                self.info_vars["Valid Peaks"].set(str(latest.get("num_valid_peaks", 0)))
                self.info_vars["Quality [dB]"].set(
                    f"{latest.get('raw_quality_db', np.nan):.2f}"
                )
                self.info_vars["Update Rate"].set(
                    f"{latest.get('update_rate_hz', 0.0):.2f} Hz"
                )
                self.info_vars["Status"].set(latest.get("reason", "Invalid"))

        self.root.after(UI_REFRESH_MS, self.update_ui)

    def on_close(self):
        self.stop_event.set()
        self.root.destroy()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = AoAApp(root)
    root.mainloop()
