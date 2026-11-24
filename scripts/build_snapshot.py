"""Entry point for GitHub Actions and the command line.

    python scripts/build_snapshot.py          # real data (needs FRED_API_KEY)
    python scripts/build_snapshot.py --demo   # synthetic data, no keys or network
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from macronav.pipeline import main  # noqa: E402

if __name__ == "__main__":
    snap = main(demo="--demo" in sys.argv)
    print(snap["narrative"]["headline"])
    problems = snap["data_status"]["problems"]
    if problems:
        print(f"{len(problems)} data problem(s):")
        for k, v in problems.items():
            print(f"  {k}: {v}")
