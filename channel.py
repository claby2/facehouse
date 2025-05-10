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

    V_house = (V_epochs[dat["stim_id"] <= 50]).mean(0)
    V_face = (V_epochs[dat["stim_id"] > 50]).mean(0)
    num_electrodes = V_house.shape[1]
    assert V_house.shape == V_face.shape
    plt.figure(figsize=(20, 10))
    for j in range(num_electrodes):
        plt.subplot(6, 10, j + 1)
        plt.plot(trange, V_house[:, j])
        plt.plot(trange, V_face[:, j])
        plt.title("ch%d" % j)
        plt.xticks([-200, 0, 200])
        plt.ylim([0, 4])
    plt.show()


def visualize(patient: int, experiment: int) -> None:
    """
    Visualize the electrode channel activity for a given patient in a given experiment.
    Activity for houses and faces is shown in different colors.
    """
    _visualize(data.alldat[patient][experiment])


def visualize_channel(patient: int, experiment: int, channel: int) -> None:
    dat1 = data.alldat[patient][experiment]
    isort = np.argsort(dat1["stim_id"])
    V_epochs = extract_epochs(dat1)
    plt.imshow(
        V_epochs[isort, :, channel].astype("float32"),
        aspect="auto",
        vmax=7,
        vmin=0,
        cmap="magma",
    )
    plt.colorbar()
    plt.show()


# visualize_channel(1, 0, 46)


def compute_selectivity_index(
    dat, noisy=False, pre=200, post=400, resp_start=400, resp_end=None
):
    """
    Compute a d-prime selectivity index for each channel,
    handling both clean (exp1) and noisy (exp2) data.
    """
    # 1) epochs: (nTrials, pre+post, nChan)
    trange = np.arange(-pre, post)
    V_epochs = extract_epochs(dat, trange)
    nTrials, nepoch, nchan = V_epochs.shape

    # 2) baseline‑correct
    baseline = V_epochs[:, :pre, :].mean(axis=1, keepdims=True)
    V_bc = V_epochs - baseline

    # 3) response window
    if resp_end is None:
        resp_end = nepoch
    # → (nTrials, nChan)
    resp = V_bc[:, resp_start:resp_end, :].mean(axis=1)

    # 4) pick which trials are houses vs faces
    if noisy:
        mask_house = dat["stim_cat"].squeeze() == 1
        mask_face = dat["stim_cat"].squeeze() == 2
    else:
        mask_house = dat["stim_id"].squeeze() <= 50
        mask_face = dat["stim_id"].squeeze() > 50

    # 5) pull out resp for each condition
    resp_h = resp[mask_house]  # shape = (#houseTrials, nChan)
    resp_f = resp[mask_face]  # shape = (#faceTrials,  nChan)

    # 6) compute d′ channel by channel
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


# For each patient, print the most selective channel for both experiments
# Ideally, these should be the same...
for patient in range(7):
    print(f"Patient {patient}:")
    exp1 = identify_selective(patient, 0)
    exp2 = identify_selective(patient, 1)
    print(f"  Experiment 1: {exp1}")
    print(f"  Experiment 2: {exp2}")
