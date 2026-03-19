import os
import pickle
import numpy as np
from pathlib import Path

RAW_DATASET_DIR = Path("data/ServerMachineDataset")
PER_MACHINE_DIR = Path("data/per_machine")


def process_machine_file(filename: str) -> None:
    machine_name = Path(filename).stem

    for split in ["train", "test", "test_label"]:
        input_path = RAW_DATASET_DIR / split / filename
        output_path = PER_MACHINE_DIR / f"{machine_name}_{split}.pkl"

        array = np.genfromtxt(input_path, dtype=np.float32, delimiter=",")
        print(f"Loaded {input_path} with shape {array.shape}")
        with open(output_path, "wb") as f:
            pickle.dump(array, f)
        print(f"Saved to {output_path}")


def prepare_machine_data() -> None:
    PER_MACHINE_DIR.mkdir(parents=True, exist_ok=True)

    train_dir = RAW_DATASET_DIR / "train"
    filenames = sorted(f for f in os.listdir(train_dir) if f.endswith(".txt"))

    for filename in filenames:
        process_machine_file(filename)


if __name__ == "__main__":
    prepare_machine_data()