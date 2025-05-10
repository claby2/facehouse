import numpy as np
import data


def calculate_base_accuracy(
    patient: int, min_noise: float = 0, max_noise: float = 100
) -> float:
    """
    For experiment 2 only, compute overall percent correct
    (hits + correct rejections) for trials whose noise%
    lies between min_noise and max_noise (inclusive).
    """
    dat = data.alldat[patient][1]  # experiment 2

    stim_cat = dat["stim_cat"].squeeze()
    stim_noise = dat["stim_noise"].squeeze()
    t_on = dat["t_on"].squeeze()
    t_off = dat["t_off"].squeeze()
    presses = np.asarray(dat["key_press"]).ravel()

    noise_mask = (stim_noise >= min_noise) & (stim_noise <= max_noise)

    if not noise_mask.any():
        return np.nan

    cats = stim_cat[noise_mask]
    ons = t_on[noise_mask]
    offs = t_off[noise_mask]
    n_trials = len(cats)

    responded_face = np.array(
        [np.any((presses >= ons[i]) & (presses <= offs[i])) for i in range(n_trials)]
    )

    is_face = cats == 2
    is_house = cats == 1
    correct = (is_face & responded_face) | (is_house & ~responded_face)

    return correct.sum() / n_trials


# print_base_accuracies()
