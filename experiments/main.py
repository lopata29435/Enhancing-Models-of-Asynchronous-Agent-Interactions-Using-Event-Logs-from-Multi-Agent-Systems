import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', 'src'))
sys.path.append(HERE)

import batch_runner
import batch_runner

SUPPORTED_DATASET = "IP-1"

def _copy_reference_model(dataset_name: str, data_dir: str, output_dir: str) -> None:
    data_path = os.path.join(data_dir, dataset_name)
    output_subdir = os.path.join(output_dir, dataset_name)
    os.makedirs(output_subdir, exist_ok=True)

    for ref_file in glob.glob(os.path.join(data_path, "*ref*.pnml")):
        dest_file = os.path.join(output_subdir, os.path.basename(ref_file))
        shutil.copy2(ref_file, dest_file)
        print(f"[INFO] Copied reference model to: {dest_file}")

def run_synthesis_for_ip1(data_dir: str = "data", output_dir: str = "output") -> None:
    _copy_reference_model(SUPPORTED_DATASET, data_dir, output_dir)

    print("\n--- Starting synthesis + HYBRID precision evaluation for IP-1 ---")
    batch_runner.process_dataset(SUPPORTED_DATASET, data_dir, output_dir)

def _prompt_user() -> str:
    print("\n" + "=" * 30)
    print(f"Experiments runner - pinned to {SUPPORTED_DATASET}")
    print("=" * 30)
    print(f"1. Run {SUPPORTED_DATASET}")
    print("Q. Quit")
    while True:
        choice = input(
            f"\nPress enter (or 1) to run {SUPPORTED_DATASET}, 'q' to quit: "
        ).strip().lower()
        if choice in ("", "1"):
            return SUPPORTED_DATASET
        if choice == "q":
            return ""
        print("Unrecognised input. Try again.")

def main() -> None:
    non_interactive = (
        "--run" in sys.argv or "-y" in sys.argv or not sys.stdin.isatty()
    )

    while True:
        dataset = SUPPORTED_DATASET if non_interactive else _prompt_user()
        if not dataset:
            print("Exiting...")
            break
        if dataset != SUPPORTED_DATASET:
            print(
                f"This experiments entry point only supports "
                f"{SUPPORTED_DATASET}; got {dataset}."
            )
            break

        run_synthesis_for_ip1()
        print("\n" + "=" * 50)

        if non_interactive:
            break

if __name__ == "__main__":
    main()
