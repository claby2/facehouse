import data
import channel
import predict
import performance

NUM_PATIENTS = 7


def q1():
    """
    Visualize the electrode channels for a given patient in experiment 0
    """
    patient = input("Enter patient number (0-6): ")
    channel.visualize(int(patient), 0)


def q3():
    """
    NOTE: This tries to combine Q2 and Q3
    But also, Q3 is not done super correctly as noise bins are not taken into account...
    Do not think this matters too much though.

    For each patient, print the most selective channel for both experiments
    Ideally, these should be the same...
    """
    for patient in range(NUM_PATIENTS):
        print(f"Patient {patient}:")
        exp1 = channel.identify_selective(patient, 0)
        exp2 = channel.identify_selective(patient, 1)
        print(f"  Experiment 1: {exp1}")
        print(f"  Experiment 2: {exp2}")


def q4():
    """
    Calculate the baseline accuracy for each patient in experiment 2
    """
    for patient in range(NUM_PATIENTS):
        acc = performance.calculate_base_accuracy(patient)
        print(f"Patient {patient}: {acc*100:5.1f}%")


def q6():
    """
    Goal: for each patient in experiment 2, we try train a logistic regression
    model to predict whether the subject will make a mistake or not based on the
    power data from the most selective channel.


    If the accuracies are above 0.5, it probably means "the neural activity
    might be directly reflecting the noisy perception."
    """
    patient = 0
    for patient in range(NUM_PATIENTS):
        print(f"Patient {patient}:")
        dat = data.alldat[patient][1]
        chan = channel.identify_selective(patient, 1)
        print(predict.predict_mistakes(dat, chan))


# Feel free to uncomment the following lines to run the functions
# q1()
# q3()
# q4()
# q6()
