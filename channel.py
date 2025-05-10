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


def compute_selectivity_index(dat, pre=200, post=400, resp_start=400, resp_end=600):
    """
    Compute a d-prime selectivity index for each channel.
    """
    # 1) Grab your epochs: shape = (nTrials, pre+post, nchan)
    trange = np.arange(-pre, post)
    V_epochs = extract_epochs(dat, trange)
    _, nepoch, nchan = V_epochs.shape

    # 2) baseline‑correct: subtract mean over pre‑stim window [0:pre]
    baseline = V_epochs[:, :pre, :].mean(axis=1, keepdims=True)
    V_bc = V_epochs - baseline

    # 3) pick a response window
    if resp_end is None:
        resp_end = nepoch
    # average over that window → (nTrials, nchan)
    resp = V_bc[:, resp_start:resp_end, :].mean(axis=1)

    # 4) split face vs house
    is_house = dat["stim_id"] <= 50
    is_face = dat["stim_id"] > 50

    # 5) compute d′ per channel
    dprimes = np.zeros(nchan, dtype=np.float32)
    for j in range(nchan):
        xh = resp[is_house, j]
        xf = resp[is_face, j]
        mu_h, mu_f = xh.mean(), xf.mean()
        var_h, var_f = xh.var(ddof=1), xf.var(ddof=1)
        denom = np.sqrt(0.5 * (var_h + var_f)) or np.nan
        dprimes[j] = (mu_f - mu_h) / denom

    return dprimes


def visualize_selectivity_index(patient: int, experiment: int) -> None:
    """
    Visualize the selectivity index for a given patient in a given experiment.
    """
    dat = data.alldat[patient][experiment]
    dps = compute_selectivity_index(dat)

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
    dps = compute_selectivity_index(dat)
    return int(np.argmax(dps))
