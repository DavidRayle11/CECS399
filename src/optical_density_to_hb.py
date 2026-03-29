import os
import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import mne
from mne.preprocessing.nirs import optical_density, beer_lambert_law, source_detector_distances
from mne.channels import make_dig_montage

from collections import defaultdict

# load files and return data
def load_nirs(file_path):
    data = loadmat(file_path, appendmat=False)
    return data

def nirs_to_raw(data):
    """
    Convert a Homer2 .nirs structure into an MNE Raw object
    with valid continuous-wave fNIRS channel metadata.

    Expected fields in `data`:
        d   : time x channels intensity data
        t   : time vector
        SD  : Homer probe structure
        ml  : measurement list (optional if SD["MeasList"] is present)

    Returns
    -------
    raw : mne.io.RawArray
        Raw CW-amplitude fNIRS data ready for optical density conversion.
    """
    # -------------------------------
    # 1) Extract core fields
    # -------------------------------
    d = np.asarray(data["d"], dtype=float)          # time x channels
    t = np.asarray(data["t"]).flatten()

    if "SD" not in data:
        raise ValueError("Missing SD structure in .nirs file")

    SD = data["SD"][0, 0]

    # Prefer MeasList from SD if available
    if "MeasList" in SD.dtype.names:
        ml = np.asarray(SD["MeasList"], dtype=int)
    elif "ml" in data:
        ml = np.asarray(data["ml"], dtype=int)
    else:
        raise ValueError("No measurement list found in SD['MeasList'] or data['ml']")

    if "Lambda" not in SD.dtype.names:
        raise ValueError("Missing SD['Lambda'] in .nirs file")

    wavelengths = np.asarray(SD["Lambda"]).flatten().astype(float)

    if "SrcPos" not in SD.dtype.names or "DetPos" not in SD.dtype.names:
        raise ValueError("Missing SrcPos or DetPos in SD structure")

    src_pos = np.asarray(SD["SrcPos"], dtype=float)
    det_pos = np.asarray(SD["DetPos"], dtype=float)

    # -------------------------------
    # 2) Sampling rate
    # -------------------------------
    if len(t) < 2:
        raise ValueError("Time vector must contain at least 2 samples")

    sfreq = 1.0 / np.mean(np.diff(t))

    # -------------------------------
    # 3) Reformat intensity data
    # -------------------------------
    data_ch = d.T  # channels x time

    if data_ch.shape[0] != ml.shape[0]:
        raise ValueError(
            f"Channel count mismatch: data has {data_ch.shape[0]} channels, "
            f"measurement list has {ml.shape[0]} rows"
        )

    for i in range(data_ch.shape[0]):
        if np.all(data_ch[i] == 0):
            print(f"Warning: Channel {i} is flat")

    # -------------------------------
    # 4) Convert optode positions to meters
    # -------------------------------
    max_val = max(np.max(np.abs(src_pos)), np.max(np.abs(det_pos)))

    if max_val > 100:
        print("Converting optode positions from mm to meters")
        src_pos = src_pos / 1000.0
        det_pos = det_pos / 1000.0
    elif max_val > 10:
        print("Converting optode positions from cm to meters")
        src_pos = src_pos / 100.0
        det_pos = det_pos / 100.0
    else:
        print("Optode positions already in meters")

    # -------------------------------
    # 5) Build and validate channel defs
    # -------------------------------
    channel_defs = []

    for i in range(ml.shape[0]):
        src = int(ml[i, 0]) - 1
        det = int(ml[i, 1]) - 1
        wl_idx = int(ml[i, 3]) - 1

        if src < 0 or src >= len(src_pos):
            raise ValueError(f"Invalid source index at ml row {i}: {src+1}")
        if det < 0 or det >= len(det_pos):
            raise ValueError(f"Invalid detector index at ml row {i}: {det+1}")
        if wl_idx < 0 or wl_idx >= len(wavelengths):
            raise ValueError(f"Invalid wavelength index at ml row {i}: {wl_idx+1}")

        wavelength = float(wavelengths[wl_idx])

        if np.isnan(wavelength):
            raise ValueError(f"Wavelength is NaN at ml row {i}")

        channel_defs.append({
            "src": src,
            "det": det,
            "wavelength": wavelength,
            "orig_idx": i,
        })

    # -------------------------------
    # 6) Reorder channels into source-detector pairs
    #    so each pair has wavelengths together
    # -------------------------------
    grouped = defaultdict(list)

    for ch in channel_defs:
        grouped[(ch["src"], ch["det"])].append(ch)

    ordered_channels = []

    for key in sorted(grouped):
        pair = sorted(grouped[key], key=lambda x: x["wavelength"])

        if len(pair) != 2:
            print(f"Skipping incomplete pair: S{key[0]+1}_D{key[1]+1}")
            continue

        ordered_channels.extend(pair)

    if len(ordered_channels) == 0:
        raise ValueError("No valid source-detector wavelength pairs found")

    new_order = [ch["orig_idx"] for ch in ordered_channels]
    data_ch = data_ch[new_order, :]

    # -------------------------------
    # 7) Build channel names/types
    # -------------------------------
    ch_names = [
        f"S{ch['src']+1}_D{ch['det']+1} {int(round(ch['wavelength']))}"
        for ch in ordered_channels
    ]
    ch_types = ["fnirs_cw_amplitude"] * len(ch_names)

    # -------------------------------
    # 8) Create info and explicitly fill loc
    # -------------------------------
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=ch_types
    )

    for i, ch in enumerate(ordered_channels):
        s_pos = src_pos[ch["src"]]
        d_pos = det_pos[ch["det"]]
        wl = float(ch["wavelength"])

        # Reset loc to known state
        info["chs"][i]["loc"][:] = np.nan

        # MNE fNIRS convention
        # loc[3:6] = source position
        # loc[6:9] = detector position
        # loc[9]   = wavelength
        info["chs"][i]["loc"][3:6] = s_pos
        info["chs"][i]["loc"][6:9] = d_pos
        info["chs"][i]["loc"][9] = wl

    # -------------------------------
    # 9) Create raw object
    # -------------------------------
    raw = mne.io.RawArray(data_ch, info)

    # -------------------------------
    # 10) Add montage for probe geometry display
    # -------------------------------
    dig_dict = {}

    for i, pos in enumerate(src_pos):
        dig_dict[f"S{i+1}"] = pos

    for i, pos in enumerate(det_pos):
        dig_dict[f"D{i+1}"] = pos

    montage = make_dig_montage(ch_pos=dig_dict, coord_frame="head")
    raw.set_montage(montage, on_missing="ignore")

    # -------------------------------
    # 11) Debug checks
    # -------------------------------
    encoded_wls = np.array([ch["loc"][9] for ch in raw.info["chs"]], dtype=float)
    print("Encoded wavelengths (first 10):", encoded_wls[:10])
    print("Unique wavelengths:", np.unique(np.round(encoded_wls, 3)))

    from mne.preprocessing.nirs import source_detector_distances
    dists = source_detector_distances(raw.info)

    print("\nChannels with large source-detector distances:")
    for ch_name, dist in zip(raw.ch_names, dists):
        if dist > 0.10:
            print(f"{ch_name}: {dist:.4f} m")

    print("MNE distance stats:")
    print("Min:", np.min(dists))
    print("Max:", np.max(dists))
    print("Mean:", np.mean(dists))

    return raw

# Optical density
def convert_to_optical_density(raw):
    return optical_density(raw)
    
# Beer-Lambert (HbO / HbR)
def convert_to_hemoglobin(raw_od):
    return beer_lambert_law(raw_od)

def drop_long_distance_channels(raw, max_dist=0.10):
    dists = source_detector_distances(raw.info)
    bads = [ch for ch, dist in zip(raw.ch_names, dists) if dist > max_dist]

    n_before = len(raw.ch_names)

    if bads:
        print(f"\nDropping {len(bads)} long-distance channels (>{max_dist:.2f} m):")
        for ch in bads:
            print(" ", ch)

    raw = raw.copy().drop_channels(bads)

    n_after = len(raw.ch_names)
    print(f"Channels kept: {n_after}/{n_before} ({100*n_after/n_before:.1f}%)")

    return raw

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
            raw = drop_long_distance_channels(raw, max_dist=0.10)

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