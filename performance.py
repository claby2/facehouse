import numpy as np
import data


def calculate_accuracy(patient: int):
    dat = data.alldat[patient][1]  # Always do experiment 2

    face_guesses = np.zeros(dat["stim_cat"].shape, dtype=bool)
    for i in range(len(dat["stim_cat"])):
        face_guesses[i] = np.any(
            (dat["key_press"] >= dat["t_on"][i]) & (dat["key_press"] <= dat["t_off"][i])
        )
    accuracy = np.sum((dat["stim_cat"] == 2) & face_guesses) / len(face_guesses)
    return accuracy


# Print the accuracy for each patient
for patient in range(7):
    accuracy = calculate_accuracy(patient)
    print(f"Patient {patient}: {accuracy:.2f}")
