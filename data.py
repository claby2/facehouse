# @title Data retrieval
import os
import requests
import numpy as np
from scipy import signal

fname = "faceshouses.npz"
url = "https://osf.io/argh7/download"

if not os.path.isfile(fname):
    try:
        r = requests.get(url)
    except requests.ConnectionError:
        print("!!! Failed to download data !!!")
    else:
        if r.status_code != requests.codes.ok:
            print("!!! Failed to download data !!!")
        else:
            with open(fname, "wb") as fid:
                fid.write(r.content)

# alldat[patient_id][experiment_id]
alldat = np.load(fname, allow_pickle=True)["dat"]


# Experiment 1:
# dat1['V']: continuous voltage data (time by channels)
# dat1['srate']: acquisition rate (1000 Hz). All stimulus times are in units of this.
# dat1['t_on']: time of stimulus onset in data samples
# dat1['t_off']: time of stimulus offset, always 400 samples after t_on
# dat1['stim_id]: identity of stimulus from 1-100, with 1-50 being houses and 51-100 being faces
# dat1['locs]: 3D electrode positions on the brain surface

# Experiment 2:
# dat2['V]: continuous voltage data (time by channels)
# dat2['srate']: acquisition rate (1000 Hz). All stimulus times are in units of this.
# dat2['t_on']: time of stimulus onset in data samples
# dat2['t_off']: time of stimulus offset, always 1000 samples after t_on, with no inter-stimulus interval
# dat2['stim_id]: identity of stimulus from 1-600 (not really useful, since we don't know which ones are the same house/face)
# dat2['stim_cat']: stimulus category (1 = house, 2 = face)
# dat2['stim_noise']: percent noise from 0 to 100
# dat2['key_press']: when the subject thought the image was a face
# dat2['categories']: categories legend (1 = house, 2 = face)
# dat2['locs]: 3D electrode positions on the brain surface


def convert_to_power(voltage_data: np.ndarray) -> np.ndarray:
    """Converts continuous voltage data into a continuous array of power data.
    Applies high-pass and low-pass filtering and normalizes across time."""
    voltage_thirty_two_bit = voltage_data.astype("float32")  # Convert to 32-bit floats
    b, a = signal.butter(3, [50], btype="high", fs=1000)  # type: ignore
    V_high_passed = signal.filtfilt(b, a, voltage_thirty_two_bit, 0)
    power_high_passed = np.abs(V_high_passed) ** 2
    b, a = signal.butter(3, [10], btype="low", fs=1000)  # type: ignore
    power_hi_lo_passed = signal.filtfilt(b, a, power_high_passed, 0)
    power_hi_lo_passed_norm = power_hi_lo_passed / power_hi_lo_passed.mean(0)
    return power_hi_lo_passed_norm
