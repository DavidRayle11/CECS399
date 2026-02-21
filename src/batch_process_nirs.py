import os
import re
from aux_to_s_matrix import main as process_single_file


def compute_subnum(folder_name, filename):

    match = re.match(r"[A-Za-z](\d+)_run\d{2}\.nirs", filename)
    if not match:
        raise ValueError(f"Filename format incorrect: {filename}")

    digits = match.group(1)

    if folder_name in ["S1", "S2"]:
        if len(digits) != 3:
            raise ValueError(
                f"Unexpected digit length in {filename} for folder {folder_name}"
            )
        folder_digit = folder_name[1]
        subnum = int(folder_digit + digits)
        print(f"  → 3-digit ID detected (S1/S2). Normalized to {subnum}")
        return subnum

    elif folder_name == "S3":
        if len(digits) == 4:
            subnum = int(digits)
            print(f"  → 4-digit ID detected (S3). Using {subnum}")
            return subnum
        elif len(digits) == 3:
            subnum = int("3" + digits)
            print(f"  → 3-digit ID detected (S3). Normalized to {subnum}")
            return subnum
        else:
            raise ValueError(
                f"Unexpected digit length in {filename} for folder S3"
            )

    else:
        raise ValueError(f"Unknown folder: {folder_name}")


def main_batch():

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    raw_root = os.path.join(project_root, "data", "raw_nirs")
    processed_root = os.path.join(project_root, "data", "processed_nirs")
    behavioral_path = os.path.join(project_root, "data", "TC_BehavioralData.xlsx")

    exclusion_log_path = os.path.join(project_root, "excluded_runs.txt")

    # Reset exclusion log at start of batch
    if os.path.exists(exclusion_log_path):
        os.remove(exclusion_log_path)
        print("Previous excluded_runs.txt removed. Starting fresh log.")

    folders = ["S1", "S2", "S3"]
    #folders = ["S1"]               # used in testing

    for folder in folders:

        raw_folder = os.path.join(raw_root, folder)
        processed_folder = os.path.join(processed_root, folder)

        os.makedirs(processed_folder, exist_ok=True)

        print(f"\nProcessing folder: {folder}")

        for file in os.listdir(raw_folder):

            if not file.endswith(".nirs"):
                continue

            nirs_path = os.path.join(raw_folder, file)

            try:
                subnum = compute_subnum(folder, file)

                print(f"\nFile: {file}")
                print(f"Computed SubNum: {subnum}")

                process_single_file(
                    nirs_path=nirs_path,
                    behav_path=behavioral_path,
                    subject_number=subnum
                )

                # Move file into correct subfolder
                base_name = os.path.splitext(file)[0]
                updated_name = f"{base_name}_updated.nirs"

                temp_output_path = os.path.join(
                    project_root, "data", "processed_nirs", updated_name
                )

                final_output_path = os.path.join(processed_folder, updated_name)

                if os.path.exists(temp_output_path):
                    os.replace(temp_output_path, final_output_path)
                    print(f"Moved to: {final_output_path}")
                else:
                    print("WARNING: Expected output file not found.")

            except Exception as e:
                print(f"EXCLUDED: {file} → {e}")

                with open(exclusion_log_path, "a") as log_file:
                    log_file.write(
                        f"{file} | Folder: {folder} | Error: {e}\n"
                    )


if __name__ == "__main__":
    main_batch()