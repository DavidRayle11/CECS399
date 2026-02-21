import argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
import os


def main(nirs_path, behav_path, subject_number):

    # -------------------------------
    # Resolve project root properly
    # -------------------------------
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(project_root, "data", "processed_nirs")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------
    # 1) Load NIRS file
    # -------------------------------
    data = loadmat(nirs_path)

    t = data["t"].flatten()
    aux = data["aux"]

    print("Loaded:", nirs_path)
    print("Time points:", t.shape)
    print("AUX shape:", aux.shape)

    # -------------------------------
    # 2) Extract AUX trigger channel
    # -------------------------------
    aux0 = aux[:, 0]

    threshold = 1.0
    digital = aux0 > threshold
    onsets = np.where(digital[1:] & ~digital[:-1])[0] + 1

    print("Detected AUX events:", len(onsets))

    # HARD STOP if no triggers
    if len(onsets) == 0:
        raise ValueError(
            f"No AUX triggers detected. Cannot time-lock this run."
        )

    # -------------------------------
    # 3) Load Behavioral Data
    # -------------------------------
    behav = pd.read_excel(behav_path)

    # Ensure correct dtype
    behav["SubNum"] = behav["SubNum"].astype(int)

    behav_sub = behav[behav["SubNum"] == subject_number]

    print("Behavioral trials (filtered):", len(behav_sub))

    if len(behav_sub) == 0:
        raise ValueError(
            f"No behavioral trials found for SubNum {subject_number}"
        )

    # -------------------------------
    # 4) Sanity Check
    # -------------------------------
    if len(onsets) != len(behav_sub):
        print("WARNING: AUX events and behavioral trials do not match!")
        print("AUX events:", len(onsets))
        print("Behavioral trials:", len(behav_sub))
        print("Proceeding with minimum length to avoid crash.")

    min_len = min(len(onsets), len(behav_sub))

    # -------------------------------
    # 5) Build S Matrix
    # -------------------------------
    n_times = len(t)
    S_new = np.zeros((n_times, 2))

    for i in range(min_len):
        onset = onsets[i]
        condition = int(behav_sub.iloc[i]["TType"])

        if condition == 1:
            S_new[onset, 0] = 1
        elif condition == 2:
            S_new[onset, 1] = 1
        else:
            print(f"Unknown TType at trial {i}: {condition}")

    print("S matrix shape:", S_new.shape)

    # -------------------------------
    # 6) Replace S in structure
    # -------------------------------
    data["s"] = S_new

    # -------------------------------
    # 7) Save new file
    # -------------------------------
    base = os.path.basename(nirs_path)
    name, _ = os.path.splitext(base)
    output_path = os.path.join(output_dir, f"{name}_updated.nirs")

    savemat(output_path, data, appendmat=False)

    print("Saved:", output_path)


# -------------------------------
# Command Line Interface
# -------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert AUX to S matrix")

    parser.add_argument("--nirs", required=True)
    parser.add_argument("--behav", required=True)
    parser.add_argument("--sub", required=True, type=int)

    args = parser.parse_args()

    main(args.nirs, args.behav, args.sub)