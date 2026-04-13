# Direction Finding

Small PlutoSDR experiment scripts for checking receiver configuration and capturing a short block of IQ samples in Python.

## Contents

- `checkRX.py`: connects to the PlutoSDR and prints the available RX channel names and the currently enabled RX channels.
- `gettingstart.py`: minimal connection test that sets the sample rate and performs a basic `rx()` call.
- `receving.py`: configures two RX channels, tunes the radio, captures samples, and prints the returned IQ data.

## Requirements

- Python 3
- `pyadi-iio` (`import adi`)
- `numpy`
- An ADALM-PLUTO / PlutoSDR reachable at `ip:192.168.2.1`

Install Python packages with:

```bash
pip install pyadi-iio numpy
```

## Hardware Assumptions

These scripts are currently hard-coded to use:

- PlutoSDR IP: `192.168.2.1`
- Center frequency: `700e6`
- Sample rate: `1e6` or `2.5e6`, depending on the script
- Two receive channels in `receving.py`

If your PlutoSDR uses a different IP, update:

```python
adi.Pluto('ip:192.168.2.1')
```

If your device does not expose two RX channels, `receving.py` will need to be adjusted before it will run successfully.

## Usage

Run the scripts directly with Python:

```bash
python checkRX.py
python gettingstart.py
python receving.py
```

Suggested order:

1. Run `checkRX.py` to verify the device is reachable and inspect RX channel availability.
2. Run `gettingstart.py` to confirm a basic receive call works.
3. Run `receving.py` to capture and print sample data from both receive channels.

## What `receving.py` Does

The script:

- enables RX channels `0` and `1`
- sets manual gain mode on both channels
- applies `30 dB` hardware gain to both channels
- tunes the LO to `700 MHz`
- sets sample rate and RF bandwidth to `1 MHz`
- captures `20` samples per receive call
- prints the full sample array and the first few samples from each channel

## Notes

- The filename `receving.py` appears to be a typo of "receiving", but this README keeps the current filename as-is.
- These scripts are basic experiments and do not yet include error handling, logging, or signal processing for actual direction-finding estimation.
