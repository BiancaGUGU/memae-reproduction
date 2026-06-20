import os

for memory in [
    200,
    500,
    1000
]:

    print()

    print(
        f"Running memory={memory}"
    )

    os.system(
        f"sed -i 's/MEMORY_SIZE = .*/MEMORY_SIZE = {memory}/' train_ucsd_extended.py"
    )

    os.system(
        "python train_ucsd_extended.py"
    )