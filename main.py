import channel

patient = 0

# Get the most selective channel in experiment 1 for this patient
selective_channel = channel.identify_selective(patient, 1)
print(
    f"Most selective channel in experiment 1 for patient {patient}: {selective_channel}"
)

channel.visualize(patient, 1)
channel.visualize_channel(patient, 1, selective_channel)
