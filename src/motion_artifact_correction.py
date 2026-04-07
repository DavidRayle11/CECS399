import os
from pathlib import Path
import argparse

import numpy as np
from scipy.io import loadmat

import mne
from mne.preprocessing.nirs import optical_density
from mne.preprocessing.nirs import source_detector_distances
from mne.preprocessing.nirs import temporal_derivative_distribution_repair


# -------------------------------
# Project paths
# -------------------------------
def get_project_root():
    return Path(__file__).resolve().parent.parent


# -------------------------------
# Load Homer2 .nirs file
# -------------------------------
def load_nirs(file_path):
    return loadmat(file_path, appendmat=False)


# -------------------------------
# Convert Homer2 structure to MNE Raw
# -------------------------------
def nirs_to_raw(data, distance_warning_threshold=0.10):
    """
    Build an MNE Raw object from a Homer2 .nirs MATLAB structure.

    Expected fields:
      d   : time x channels intensity data
      t   : time vector
      s   : stimulus matrix
      ml  : measurement list
      SD  : probe structure
    """
    d = data["d"]                      # time x channels
    t = data["t"].flatten()
    s = data.get("s", None)
    ml = data["ml"]
    SD = data["SD"][0, 0]

    # -------------------------------
    # Sampling frequency
    # -------------------------------
    if len(t) < 2:
        raise ValueError("Time vector is too short to compute sampling rate.")

    sfreq = 1.0 / np.mean(np.diff(t))

    # -------------------------------
    # Channel data -> channels x time
    # -------------------------------
    data_ch = d.T

    # -------------------------------
    # Flat channel warning
    # -------------------------------
    flat_channels = []
    for i in range(data_ch.shape[0]):
        if np.allclose(data_ch[i], data_ch[i, 0]):
            flat_channels.append(i)

    if flat_channels:
        print(f"WARNING: {len(flat_channels)} flat channel(s) detected.")
        print("Flat channel indices:", flat_channels)

    # -------------------------------
    # Probe geometry
    # -------------------------------
    wavelengths = np.array(SD["Lambda"]).flatten().astype(float)
    src_pos = np.array(SD["SrcPos"], dtype=float)
    det_pos = np.array(SD["DetPos"], dtype=float)

    # Heuristic: if positions are large, they are probably in mm
    max_coord = max(np.abs(src_pos).max(), np.abs(det_pos).max())
    if max_coord > 10:
        print("Converting optode positions from mm to meters")
        src_pos = src_pos / 1000.0
        det_pos = det_pos / 1000.0
    else:
        print("Optode positions appear to already be in meters")

    # -------------------------------
    # Build channel metadata
    # -------------------------------
    ch_names = []
    ch_types = []
    info = []

    # ml can appear with extra columns; first 4 are usually:
    # [source, detector, data_type, wavelength_index]
    # Using column 0,1,3 here.
    for row in ml:
        src_idx = int(row[0]) - 1
        det_idx = int(row[1]) - 1
        wl_idx = int(row[3]) - 1

        if wl_idx < 0 or wl_idx >= len(wavelengths):
            raise ValueError(
                f"Invalid wavelength index {wl_idx + 1} for measurement list row {row}"
            )

        wl = float(wavelengths[wl_idx])
        ch_name = f"S{src_idx + 1}_D{det_idx + 1} {int(wl)}"

        ch_names.append(ch_name)
        ch_types.append("fnirs_cw_amplitude")

        info.append(
            {
                "src_idx": src_idx,
                "det_idx": det_idx,
                "wavelength": wl,
            }
        )

    # -------------------------------
    # Create MNE Info / Raw
    # -------------------------------
    mne_info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=ch_types
    )

    raw = mne.io.RawArray(data_ch, mne_info)

    # -------------------------------
    # Fill wavelength + geometry into info["chs"]
    # -------------------------------
    for idx, ch in enumerate(raw.info["chs"]):
        src_idx = info[idx]["src_idx"]
        det_idx = info[idx]["det_idx"]
        wl = info[idx]["wavelength"]

        ch["loc"][:] = 0.0
        ch["loc"][3:6] = src_pos[src_idx]
        ch["loc"][6:9] = det_pos[det_idx]
        ch["loc"][9] = wl

    encoded_wavelengths = np.array([ch["loc"][9] for ch in raw.info["chs"]])
    print("Encoded wavelengths (first 10):", encoded_wavelengths[:10])
    print("Unique wavelengths:", np.unique(encoded_wavelengths))

    # -------------------------------
    # Source-detector distance checks
    # -------------------------------
    distances = source_detector_distances(raw.info)

    print("\nSource-detector distance stats:")
    print(f"  Min : {distances.min():.4f} m")
    print(f"  Max : {distances.max():.4f} m")
    print(f"  Mean: {distances.mean():.4f} m")

    large_dist_idx = np.where(distances > distance_warning_threshold)[0]
    if len(large_dist_idx) > 0:
        print("\nWARNING: Channels with large source-detector distances:")
        for idx in large_dist_idx:
            print(f"  {raw.ch_names[idx]}: {distances[idx]:.4f} m")
        print(
            f"WARNING: {len(large_dist_idx)} channel(s) exceed "
            f"{distance_warning_threshold:.2f} m"
        )

    # -------------------------------
    # Preserve stimulus timing as annotations
    # -------------------------------
    # This keeps event information with the saved FIF file.
    if s is not None:
        s = np.array(s)
        if s.ndim == 1:
            s = s[:, np.newaxis]

        onsets = []
        durations = []
        descriptions = []

        for cond_idx in range(s.shape[1]):
            event_samples = np.where(s[:, cond_idx] > 0)[0]
            for sample_idx in event_samples:
                onsets.append(sample_idx / sfreq)
                durations.append(0.0)
                descriptions.append(f"Condition_{cond_idx + 1}")

        if len(onsets) > 0:
            annotations = mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions
            )
            raw.set_annotations(annotations)
            print(f"Added {len(onsets)} event annotation(s) from s matrix.")
        else:
            print("No events found in s matrix to convert into annotations.")

    return raw


# -------------------------------
# Optical density + motion correction
# -------------------------------
def correct_motion_artifacts(raw):
    """
    Convert continuous-wave intensity to optical density and apply
    motion artifact correction using TDDR.

    TDDR is a practical and standard choice for student/research pipelines
    because it targets abrupt motion-related changes while preserving slower
    hemodynamic trends better than simple clipping or manual rejection.
    """
    print("\nConverting intensity to optical density...")
    raw_od = optical_density(raw)

    print("Applying motion artifact correction (TDDR) on optical density...")
    raw_od_corr = temporal_derivative_distribution_repair(raw_od)

    return raw_od_corr


# -------------------------------
# Save output
# -------------------------------
def save_corrected_output(raw_od_corr, input_path, output_root):
    input_path = Path(input_path)
    folder_name = input_path.parent.name
    base_name = input_path.stem

    output_dir = output_root / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{base_name}_motion_corrected_raw.fif"

    raw_od_corr.save(output_file, overwrite=True)
    print("Saved:", output_file)

    return output_file


# -------------------------------
# Process one file
# -------------------------------
def process_single_file(nirs_path, output_root, distance_warning_threshold=0.10):
    print("\n" + "=" * 60)
    print("Processing:", nirs_path)

    data = load_nirs(nirs_path)
    raw = nirs_to_raw(
        data,
        distance_warning_threshold=distance_warning_threshold
    )
    raw_od_corr = correct_motion_artifacts(raw)
    save_corrected_output(raw_od_corr, nirs_path, output_root)


# -------------------------------
# Batch processing
# -------------------------------
def process_all_files(distance_warning_threshold=0.10):
    project_root = get_project_root()
    input_root = project_root / "data" / "processed_nirs"
    output_root = project_root / "data" / "motion_corrected_od"

    folders = ["S1", "S2", "S3"]

    for folder in folders:
        folder_path = input_root / folder

        if not folder_path.exists():
            print(f"WARNING: Folder not found, skipping: {folder_path}")
            continue

        print(f"\nScanning folder: {folder_path}")

        for file_name in os.listdir(folder_path):
            if not file_name.endswith(".nirs"):
                continue

            nirs_path = folder_path / file_name

            try:
                process_single_file(
                    nirs_path=nirs_path,
                    output_root=output_root,
                    distance_warning_threshold=distance_warning_threshold
                )
            except Exception as e:
                print(f"ERROR processing {file_name}: {e}")


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Apply motion artifact correction to Homer2 .nirs files"
    )

    parser.add_argument(
        "--nirs",
        type=str,
        default=None,
        help="Path to one .nirs file. If omitted, process all files in processed_nirs/S1,S2,S3"
    )

    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.10,
        help="Warn if source-detector distance exceeds this value in meters"
    )

    args = parser.parse_args()

    project_root = get_project_root()
    output_root = project_root / "data" / "motion_corrected_od"

    if args.nirs is not None:
        process_single_file(
            nirs_path=Path(args.nirs),
            output_root=output_root,
            distance_warning_threshold=args.distance_threshold
        )
    else:
        process_all_files(
            distance_warning_threshold=args.distance_threshold
        )


if __name__ == "__main__":
    main()