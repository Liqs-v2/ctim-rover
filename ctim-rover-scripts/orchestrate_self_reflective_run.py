import argparse
import subprocess
import os

from perform_self_reflections import perform_self_reflections_and_cleanup_output
from sort_run_outputs_by_outcome import update_current_status_and_remaining_tasks

def main(retries: int):
    """
    Runs the E2E self-reflection pipeline with ACR. Assumes that the environments have been set up using the
    SWE-bench harness as specified by AutoCodeRover. The task list must be named 'remaining_tasks.txt' and be placed
    in 'auto-code-rover/conf/remaining_tasks.txt'

    Args:
        retries: Number of reflection retries to perform
    """

    for attempt in range(retries):
        with open("/opt/auto-code-rover/conf/remaining_tasks.txt", "r") as f:
            remaining_tasks = f.read().splitlines()
            if not remaining_tasks:
                print("All tasks resolved. Exiting.")
                break

        subprocess.run([
            "/opt/conda/envs/auto-code-rover/bin/python3", "app/main.py",
            "swe-bench",
            "--model", "gpt-4o",
            "--setup-map", "../SWE-bench/setup_result/setup_map.json",
            "--tasks-map", "../SWE-bench/setup_result/tasks_map.json",
            "--output-dir", "output",
            "--task-list-file", "/opt/auto-code-rover/conf/remaining_tasks.txt",
            "--enable-validation",
            "--use-reflections",
            "--num-processes", "8"
        ], cwd=os.getcwd())

        update_current_status_and_remaining_tasks()

        if attempt < retries - 1:
            perform_self_reflections_and_cleanup_output(cleanup_mode=False)
        else:
            perform_self_reflections_and_cleanup_output(cleanup_mode=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process command line arguments.")
    parser.add_argument('--retries', type=int, default=3,
                        help="Number of self reflection retries the system gets for unresolved samples.")

    args = parser.parse_args()

    main(retries=args.retries)
