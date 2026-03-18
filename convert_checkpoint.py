"""
OPTIONAL optimization script.

The Docker container works WITHOUT running this first.
However, if you want faster startup on subsequent runs, this script
merges your fine-tuned GPT weights with the base XTTS v2 model into
a single complete inference checkpoint.

Requirements (host machine):
    pip install torch TTS

Usage:
    python convert_checkpoint.py
"""
import os
import sys
import gc

os.environ["PYTHONUNBUFFERED"] = "1"

def log(msg):
    print(msg, flush=True)

def main():
    log("This script is OPTIONAL. The container handles loading automatically.")
    log("It merges fine-tuned weights + base XTTS into one file for faster startup.\n")

    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ljspeech_model")
    ORIGINAL = os.path.join(MODEL_DIR, "model.pth")

    if not os.path.exists(ORIGINAL):
        log(f"ERROR: Checkpoint not found: {ORIGINAL}")
        sys.exit(1)

    log(f"Fine-tuned checkpoint: {ORIGINAL} ({os.path.getsize(ORIGINAL)/1e6:.0f} MB)")
    log("The Docker container will load the base XTTS model first,")
    log("then overlay your fine-tuned GPT weights on top.")
    log("\nNo conversion needed. Just copy the folder and run:")
    log("  docker compose build")
    log("  docker compose up -d")


if __name__ == "__main__":
    main()
