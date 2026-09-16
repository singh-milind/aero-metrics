import subprocess
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


if __name__ == "__main__":

    # RUN COMPLETE DVC PIPELINE

    subprocess.run(
        ["dvc", "repro"],
        cwd=ROOT_DIR,
        check=True,
    )