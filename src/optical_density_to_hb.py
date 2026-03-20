import os
import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import mne
from mne.preprocessing.nirs import optical_density, beer_lambert_law

from collections import defaultdict

# load files and return data
def load_nirs(file_path):
    data = loadmat(file_path, appendmat=False)
    return data

# Convert to MNE Raw
def nirs_to_raw(data):
    d= data["d"]                # time x channels
    t = data["t"].flatten()     
    ml = data["ml"]             # measurement list
    SD = data["SD"][0, 0]       # MATLAB struct

    # sample frequency
    sfreq = 1.0 / np.mean(np.diff(t))

    # transpose data -> channels x time
    data_ch = d.T

    for i in range(data_ch.shape[0]):
        if np.all(data_ch[i] == 0):
            print(f"Warning: Channel {i} is flat")

    # wavelenghts
    wavelengths = SD["Lambda"].flatten()

    # Source/Detector positions
    src_pos = SD["SrcPos"].astype(float)
    det_pos = SD["DetPos"].astype(float)

    # print("Max src_pos:", np.max(src_pos))

    # Convert mm → meters if needed
    max_val = np.max(src_pos)

    if max_val > 100:  # clearly mm (e.g., 150 mm)
        print("Converting optode positions from mm to meters")
        src_pos = src_pos / 1000.0
        det_pos = det_pos / 1000.0

    elif max_val > 10:  # likely cm (e.g., 15 cm)
        print("Converting optode positions from cm to meters")
        src_pos = src_pos / 100.0
        det_pos = det_pos / 100.0

    else:
        print("Optode positions already in meters")

    src_pos = np.asarray(src_pos, dtype=float)
    det_pos = np.asarray(det_pos, dtype=float)

    channels = []

    for i in range(ml.shape[0]):
        src = int(ml[i, 0]) - 1
        det = int(ml[i, 1]) - 1
        wl_idx = int(ml[i, 3]) - 1
        wavelength = wavelengths[wl_idx]

        if src < 0 or src >= len(src_pos):
            raise ValueError(f"Source index out of bounds at row {i}: {src}")
        if det < 0 or det >= len(det_pos):
            raise ValueError(f"Detector index out of bounds at row {i}: {det}")
        if wl_idx < 0 or wl_idx >= len(wavelengths):
            raise ValueError(f"Wavelength index out of bounds at row {i}: {wl_idx}")

        key = (src, det)
        channels.append((key, wavelength, i))

    grouped = defaultdict(list)

    for key, wl, idx in channels:
        grouped[key].append((wl, idx))

    new_order = []

    for key in sorted(grouped):
        sorted_pair = sorted(grouped[key], key=lambda x: x[0])

        if len(sorted_pair) == 2:
            new_order.extend([idx for _, idx in sorted_pair])
        else:
            print("Skipping incomplete pair:", key)

    # Apply reordering to data
    data_ch = data_ch[new_order, :]

    # Rebuild channel metadata AFTER ordering
    ch_names = []
    ch_types = []
    locs = []

    for idx in new_order:
        src = int(ml[idx, 0]) - 1
        det = int(ml[idx, 1]) - 1
        wl_idx = int(ml[idx, 3]) - 1

        if wl_idx >= len(wavelengths):
            raise ValueError("Wavelength index out of bounds")

        wavelength = wavelengths[wl_idx]

        name = f"S{src+1}_D{det+1} {int(wavelength)}"
        ch_names.append(name)
        ch_types.append("fnirs_cw_amplitude")

        loc = np.zeros(12)
        loc[0:3] = src_pos[src]
        loc[3:6] = det_pos[det]

        dist = np.linalg.norm(src_pos[src] - det_pos[det])
        loc[6] = dist

        loc[9] = wavelength

        locs.append(loc)
    
    dists = [loc[6] for loc in locs]

    print("Distance stats:")
    print("Min:", np.min(dists))
    print("Max:", np.max(dists))
    print("Mean:", np.mean(dists))

    # print("Num channels:", len(ch_names))
    # print("Data shape:", data_ch.shape)
    
    
    # Create MNE info
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=ch_types
    )

    # Create Raw object
    raw = mne.io.RawArray(data_ch, info)
    for i, loc in enumerate(locs):
        raw.info["chs"][i]["loc"] = loc

    return raw

# Optical density
def convert_to_optical_density(raw):
    return optical_density(raw)
    
# Beer-Lambert (HbO / HbR)
def convert_to_hemoglobin(raw_od):
    return beer_lambert_law(raw_od)

# Save output
def save_raw(raw, output_path):
    raw.save(output_path, overwrite=True)

def add_stim_channel(raw, data):
    sfreq = raw.info["sfreq"]

    if "s" not in data:
        return raw

    stim = np.asarray(data["s"])

    # Build MNE-style stim channel:
    # 0 = no event, 1..N = condition code
    if stim.ndim == 2 and stim.shape[1] > 1:
        stim_channel = np.zeros(stim.shape[0], dtype=int)

        for cond_idx in range(stim.shape[1]):
            stim_channel[stim[:, cond_idx] > 0] = cond_idx + 1
    else:
        stim_channel = stim.flatten().astype(int)

    stim_raw = mne.io.RawArray(
        stim_channel[np.newaxis, :],
        mne.create_info(["STIM"], sfreq, ch_types=["stim"])
    )

    raw.add_channels([stim_raw], force_update_info=True)
    return raw

def process_all_files(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.rglob("*.nirs"))
    

    if len(files) == 0:
        print("No .nirs files found.")
        return

    print(f"Found {len(files)} .nirs files.")

    for file in files:
        print(f"\nProcessing: {file.name}")

        try:
            data = load_nirs(file)
            raw = nirs_to_raw(data)

            # print("Sample channel names:")
            # print(raw.ch_names[:10])

            raw_od = convert_to_optical_density(raw)
            raw_hb = convert_to_hemoglobin(raw_od)

            raw_hb = add_stim_channel(raw_hb, data)

            # Preserve subfolder structure
            relative_path = file.relative_to(input_dir)
            output_file = output_dir / relative_path.with_suffix(".fif")

            # Add suffix
            output_file = output_file.with_name(output_file.stem + "_hb_raw.fif")

            output_file.parent.mkdir(parents=True, exist_ok=True)

            save_raw(raw_hb, output_file)
            print(f"Saved: {output_file}")

        except Exception as e:
            print(f"ERROR processing {file.name}: {e}")

def main(input_dir=None, output_dir=None):
    project_root = Path(__file__).resolve().parents[1]

    # default to processed_nirs/
    if input_dir is None:
        input_dir = project_root / "data" / "processed_nirs"
    if output_dir is None:
        output_dir = project_root / "data" / "processed_hb"

    print("Input dir:", input_dir)
    print("Output dir:", output_dir)

    process_all_files(input_dir, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .nirs -> Optical Density -> Hemoglobin"
    )

    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()
    main(args.input_dir, args.output_dir)