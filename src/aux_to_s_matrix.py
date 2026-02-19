import argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
import os


def main(nirs_path, behav_path, subject_number):

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

    # -------------------------------
    # 3) Load Behavioral Data
    # -------------------------------
    behav = pd.read_excel(behav_path)

    # Filter to subject
    behav_sub = behav[behav["SubNum"] == subject_number]

    print("Behavioral trials (filtered):", len(behav_sub))

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
    #    Column 0 = Shape (TType == 1)
    #    Column 1 = Color (TType == 2)
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

    output_dir = os.path.join("..", "data", "processed_nirs")
    
    # Create folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}_updated.nirs")

    savemat(output_path, data)

    print("Saved:", output_path)


# -------------------------------
# Command Line Interface
# -------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert AUX to S matrix")

    parser.add_argument("--nirs", required=True, help="Path to .nirs file")
    parser.add_argument("--behav", required=True, help="Path to behavioral Excel file")
    parser.add_argument("--sub", required=True, type=int, help="Subject number")

    args = parser.parse_args()

    main(args.nirs, args.behav, args.sub)
