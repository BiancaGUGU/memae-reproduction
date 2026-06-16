from datasets.ucsd_dataset import UCSDPed2Dataset

# dataset = UCSDPed2Dataset(
#     "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train"
# )
dataset = UCSDPed2Dataset(
    "/home/bianca/memae-reproduction/data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train"
)

print(
    "Number of frames:",
    len(dataset)
)

sample = dataset[0]

print(
    "Frame shape:",
    sample.shape
)