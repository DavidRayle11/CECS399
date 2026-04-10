import argparse
from pathlib import Path

import mne
from mne.preprocessing.nirs import beer_lambert_law, source_detector_distances


# -------------------------------
# Load motion-corrected OD FIF
# -------------------------------
def load_motion_corrected_od(file_path):
    print(f"Loading motion-corrected OD file: {file_path}")
    raw_od = mne.io.read_raw_fif(file_path, preload=True)
    return raw_od


# -------------------------------
# Beer-Lambert (HbO / HbR)
# -------------------------------
def convert_to_hemoglobin(raw_od):
    print("Converting optical density to hemoglobin concentration...")
    return beer_lambert_law(raw_od)


# -------------------------------
# Drop long-distance channels
# -------------------------------
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
    print(f"Channels kept: {n_after}/{n_before} ({100 * n_after / n_before:.1f}%)")

    return raw


# -------------------------------
# Save output
# -------------------------------
def save_raw(raw, output_path):
    raw.save(output_path, overwrite=True)


# -------------------------------
# Process all files
# -------------------------------
def process_all_files(input_dir, output_dir, max_dist=0.10):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.rglob("*_motion_corrected_raw.fif"))

    if len(files) == 0:
        print("No motion-corrected .fif files found.")
        return

    print(f"Found {len(files)} motion-corrected OD file(s).")

    for file in files:
        print(f"\nProcessing: {file.name}")

        try:
            raw_od = load_motion_corrected_od(file)

            # Optional distance-based cleanup before Hb conversion
            raw_od = drop_long_distance_channels(raw_od, max_dist=max_dist)

            raw_hb = convert_to_hemoglobin(raw_od)

            # Preserve subfolder structure
            relative_path = file.relative_to(input_dir)
            output_file = output_dir / relative_path

            # Rename suffix clearly for Hb output
            output_name = output_file.name.replace(
                "_motion_corrected_raw.fif",
                "_hb_raw.fif"
            )
            output_file = output_file.with_name(output_name)

            output_file.parent.mkdir(parents=True, exist_ok=True)

            save_raw(raw_hb, output_file)
            print(f"Saved: {output_file}")

        except Exception as e:
            print(f"ERROR processing {file.name}: {e}")


# -------------------------------
# Main
# -------------------------------
def main(input_dir=None, output_dir=None, max_dist=0.10):
    project_root = Path(__file__).resolve().parents[1]

    # New default input is motion-corrected OD
    if input_dir is None:
        input_dir = project_root / "data" / "motion_corrected_od"

    if output_dir is None:
        output_dir = project_root / "data" / "processed_hb"

    print("Input dir:", input_dir)
    print("Output dir:", output_dir)

    process_all_files(input_dir, output_dir, max_dist=max_dist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert motion-corrected optical density FIF -> hemoglobin FIF"
    )

    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-dist", type=float, default=0.10)

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.max_dist)