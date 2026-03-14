import os
import subprocess
import pandas as pd
from tqdm import tqdm
import sys
import glob

CSV_FILE = "data/balanced_train_segments.csv"
OUTPUT_DIR = "data/audioset_5000"
TARGET_CLIPS = 5000

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Reading AudioSet metadata...")

df = pd.read_csv(
    CSV_FILE,
    skiprows=3,
    names=["YTID", "start", "end", "labels"],
    skipinitialspace=True
)

df["start"] = df["start"].astype(float)
df["end"] = df["end"].astype(float)

print("Total rows:", len(df))

subset = df.sample(TARGET_CLIPS)

print("Selected subset:", len(subset))


def download_clip(ytid, start, end, output_path):

    url = f"https://www.youtube.com/watch?v={ytid}"
    template = f"{ytid}.%(ext)s"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "-f",
            "bestaudio",
            "-o",
            template,
            url
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    files = glob.glob(f"{ytid}.*")

    if not files:
        print("Failed to see file")
        return

    temp_file = files[0]

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            temp_file,
            "-ss",
            str(start),
            "-to",
            str(end),
            "-ar",
            "16000",
            "-ac",
            "1",
            output_path
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    os.remove(temp_file)


success = 0

for _, row in tqdm(subset.iterrows(), total=len(subset)):

    ytid = row["YTID"]
    start = row["start"]
    end = row["end"]

    out_file = os.path.join(OUTPUT_DIR, f"{ytid}.wav")

    if os.path.exists(out_file):
        success += 1
        continue

    download_clip(ytid, start, end, out_file)

    if os.path.exists(out_file):
        success += 1

    if success >= TARGET_CLIPS:
        break

print("Downloaded clips:", success)
print("Saved to:", OUTPUT_DIR)