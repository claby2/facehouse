import data
import numpy as np
import matplotlib.pyplot as plt


def extract_epochs(dat, trange=np.arange(-200, 400)):
    """
    Extract stimulus-aligned epochs from multichannel power data.
    """
    V = data.convert_to_power(dat["V"])
    _, nchan = V.shape
    nstim = len(dat["t_on"])

    ts = dat["t_on"][:, np.newaxis] + trange
    V_epochs = np.reshape(V[ts, :], (nstim, 600, nchan))
    return V_epochs


def _visualize(dat: np.ndarray) -> None:
    trange = np.arange(-200, 400)
    V_epochs = extract_epochs(dat, trange)

    if "stim_cat" in dat:
        # exp2: 1=house, 2=face
        is_house = dat["stim_cat"].squeeze() == 1
        is_face = dat["stim_cat"].squeeze() == 2
    else:
        # exp1: stim_id 1–50 = house, 51–100 = face
        is_house = dat["stim_id"].squeeze() <= 50
        is_face = dat["stim_id"].squeeze() > 50

    V_house = V_epochs[is_house].mean(0)
    V_face = V_epochs[is_face].mean(0)

    num_electrodes = V_house.shape[1]
    plt.figure(figsize=(20, 10))
    for j in range(num_electrodes):
        plt.subplot(6, 10, j + 1)
        plt.plot(trange, V_house[:, j], label="house")
        plt.plot(trange, V_face[:, j], label="face")
        plt.title(f"ch{j}")
        plt.xticks([-200, 0, 200])
        plt.ylim([0, 4])
        if j == 0:
            plt.legend()
    plt.tight_layout()
    plt.show()


def visualize(patient: int, experiment: int) -> None:
    """
    Visualize per‐channel average timecourses for houses vs faces
    for either experiment.
    """
    dat = data.alldat[patient][experiment]
    _visualize(dat)


def visualize_channel(patient: int, experiment: int, channel: int) -> None:
    """
    Show a heatmap of all trials (sorted by house→face) for one channel.
    """
    dat = data.alldat[patient][experiment]
    V_epochs = extract_epochs(dat)

    # choose the right sort key
    if "stim_cat" in dat:
        key = dat["stim_cat"].squeeze()
    else:
        key = dat["stim_id"].squeeze()

    isort = np.argsort(key)
    mat = V_epochs[isort, :, channel].astype("float32")

    plt.figure(figsize=(6, 8))
    plt.imshow(mat, aspect="auto", vmin=0, vmax=7, cmap="magma")
    plt.xlabel("Time (samples relative to onset)")
    plt.ylabel("Sorted trials (house→face)")
    plt.colorbar(label="Normalized power")
    plt.show()


# visualize_channel(1, 0, 46)


def compute_selectivity_index(
    dat, noisy=False, pre=200, post=400, resp_start=400, resp_end=None
):
    """
    Compute a d-prime selectivity index for each channel,
    handling both clean (exp1) and noisy (exp2) data.
    """
    trange = np.arange(-pre, post)
    V_epochs = extract_epochs(dat, trange)
    _, nepoch, nchan = V_epochs.shape

    baseline = V_epochs[:, :pre, :].mean(axis=1, keepdims=True)
    V_bc = V_epochs - baseline

    if resp_end is None:
        resp_end = nepoch
    resp = V_bc[:, resp_start:resp_end, :].mean(axis=1)

    if noisy:
        mask_house = dat["stim_cat"].squeeze() == 1
        mask_face = dat["stim_cat"].squeeze() == 2
    else:
        mask_house = dat["stim_id"].squeeze() <= 50
        mask_face = dat["stim_id"].squeeze() > 50

    resp_h = resp[mask_house]
    resp_f = resp[mask_face]

    dprimes = np.zeros(nchan, dtype=np.float32)
    for j in range(nchan):
        xh = resp_h[:, j]
        xf = resp_f[:, j]

        mu_h, mu_f = xh.mean(), xf.mean()
        var_h, var_f = xh.var(ddof=1), xf.var(ddof=1)
        denom = np.sqrt(0.5 * (var_h + var_f))
        dprimes[j] = (mu_f - mu_h) / denom if denom > 0 else np.nan

    return dprimes


def visualize_selectivity_index(patient: int, experiment: int) -> None:
    """
    Visualize the selectivity index for a given patient in a given experiment.
    """
    dat = data.alldat[patient][experiment]
    dps = compute_selectivity_index(dat, noisy=(experiment == 1))

    plt.figure(figsize=(12, 4))
    plt.bar(np.arange(len(dps)), dps)
    plt.xlabel("Channel")
    plt.ylabel("d′ (face vs house)")
    plt.axhline(0, color="k", lw=1)
    plt.show()


def identify_selective(patient: int, experiment: int) -> int:
    """
    Returns the electrode channel number of the most selective channel for a given patient in a given experiment.
    """
    dat = data.alldat[patient][experiment]
    dps = compute_selectivity_index(dat, noisy=(experiment == 1))
    return int(np.argmax(dps))


def print_selective_channels() -> None:
    # For each patient, print the most selective channel for both experiments
    # Ideally, these should be the same...
    for patient in range(7):
        print(f"Patient {patient}:")
        exp1 = identify_selective(patient, 0)
        exp2 = identify_selective(patient, 1)
        print(f"  Experiment 1: {exp1}")
        print(f"  Experiment 2: {exp2}")


# print_selective_channels()
