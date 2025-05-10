import numpy as np
import data
import channel
from channel import extract_epochs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def predict_mistakes(
    dat, channel, pre=200, post=400, resp_start=400, resp_end=None, cv=5
):
    trange = np.arange(-pre, post)
    V_epochs = extract_epochs(dat, trange)
    nTrials, _, _ = V_epochs.shape

    baseline = V_epochs[:, :pre, :].mean(axis=1, keepdims=True)
    V_bc = V_epochs - baseline

    if resp_end is None:
        resp_end = V_bc.shape[1]

    feat = V_bc[:, resp_start:resp_end, channel].mean(axis=1)  # shape (nTrials,)

    t_on = dat["t_on"].squeeze()
    t_off = dat["t_off"].squeeze()
    presses = np.asarray(dat["key_press"]).ravel()
    responded_face = np.array(
        [np.any((presses >= t_on[i]) & (presses <= t_off[i])) for i in range(nTrials)]
    )
    true_face = dat["stim_cat"].squeeze() == 2

    mistakes = responded_face != true_face

    X = feat.reshape(-1, 1)
    y = mistakes.astype(int)

    clf = LogisticRegression(solver="liblinear")
    aucs = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    mean_auc = aucs.mean()

    return mean_auc


patient = 0
for patient in range(7):
    print(f"Patient {patient}:")
    dat = data.alldat[patient][1]
    chan = channel.identify_selective(patient, 1)
    print(predict_mistakes(dat, chan))
