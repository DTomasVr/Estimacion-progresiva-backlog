"""Downloads the SiP dataset (Jones and Cullum, 2019) from its original source.

SiP is published under CC BY-NC-ND 4.0 (no commercial use, no modified versions), so this
repository does not include its rows nor any file derived from them. This script fetches the
two original files into the folder the other scripts read them from:

    python src/00_download_sip.py

  data/raw/jones/SiP/Sip-task-info.csv    tasks with their estimated and actual hours
  data/raw/jones/SiP/est-act-dates.csv    estimate and completion dates
  data/raw/jones/SiP/LICENSE.md           the dataset licence, kept next to the data

Source: https://github.com/Derek-Jones/SiP_dataset
"""
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/Derek-Jones/SiP_dataset/master/"
FILES = ["Sip-task-info.csv", "est-act-dates.csv", "LICENSE.md"]
OUT = Path("data/raw/jones/SiP")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = OUT / name
        if dest.exists():
            print(f"already there: {dest}")
            continue
        urllib.request.urlretrieve(BASE + name, dest)
        print(f"downloaded {dest} ({dest.stat().st_size:,} bytes)")
    print("SiP is CC BY-NC-ND 4.0: use it for research only and do not redistribute it or any derived file.")


if __name__ == "__main__":
    sys.exit(main())
