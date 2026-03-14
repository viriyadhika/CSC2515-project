import os
import subprocess
import pandas as pd
from tqdm import tqdm
import sys
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

CSV_FILE = "data/balanced_train_segments.csv"
OUTPUT_DIR = "data/audioset_5000"
TARGET_CLIPS = 5000
MAX_WORKERS = 8

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
    template = f"{ytid}_{int(start)}.%(ext)s"

    # download audio using yt_dlp
    subprocess.run(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "-f",
            "bestaudio",
            "--quiet",
            "--no-warnings",
            "-o",
            template,
            url
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    files = glob.glob(f"{ytid}_{int(start)}.*")

    if not files:
        return False

    temp_file = files[0]

    # trim and convert audio
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

    try:
        os.remove(temp_file)
    except:
        pass

    return os.path.exists(output_path)


def process_row(row):

    ytid = row["YTID"]
    start = row["start"]
    end = row["end"]

    out_file = os.path.join(OUTPUT_DIR, f"{ytid}_{int(start)}.wav")

    if os.path.exists(out_file):
        return True

    return download_clip(ytid, start, end, out_file)


success = 0

print(f"Starting downloads with {MAX_WORKERS} workers...")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

    futures = [
        executor.submit(process_row, row)
        for _, row in subset.iterrows()
    ]

    for future in tqdm(as_completed(futures), total=len(futures)):

        if future.result():
            success += 1

        if success >= TARGET_CLIPS:
            break

print("Downloaded clips:", success)
print("Saved to:", OUTPUT_DIR)

